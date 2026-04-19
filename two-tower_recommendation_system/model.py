"""Two-tower (dual encoder) architecture for neural retrieval.

User tower:
- Sequence encoder over past interacted items
- Event-type embedding and time-decay feature integration
- Transformer encoder + attention pooling

Item tower:
- Item ID embedding
- Multi-modal projections for image + text embeddings
- Categorical embeddings (category, brand)

Both towers output vectors in the same latent space for dot-product retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


PAD_ITEM_ID = 0
PAD_EVENT_ID = 0


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class AttentionPooling(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(d_model))

    def forward(self, sequence: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        # sequence: [B, L, D], padding_mask: [B, L] True where padded
        scores = torch.matmul(sequence, self.query)
        scores = scores.masked_fill(padding_mask, -1e9)
        attn = torch.softmax(scores, dim=1)
        pooled = torch.bmm(attn.unsqueeze(1), sequence).squeeze(1)
        return pooled


class UserTower(nn.Module):
    """Encodes a user's recent interaction sequence into a user embedding."""

    def __init__(
        self,
        item_input_dim: int,
        event_vocab_size: int,
        output_dim: int,
        user_feat_dim: int = 0,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 100,
        event_emb_dim: int = 16,
        time_emb_dim: int = 16,
    ):
        super().__init__()
        self.user_feat_dim = user_feat_dim

        self.event_embedding = nn.Embedding(
            num_embeddings=event_vocab_size,
            embedding_dim=event_emb_dim,
            padding_idx=PAD_EVENT_ID,
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.input_projection = nn.Linear(
            item_input_dim + event_emb_dim + time_emb_dim,
            d_model,
        )
        self.position = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pooling = AttentionPooling(d_model=d_model)

        if user_feat_dim > 0:
            self.user_feat_mlp = nn.Sequential(
                nn.Linear(user_feat_dim, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            fusion_in = d_model + d_model // 2
        else:
            self.user_feat_mlp = None
            fusion_in = d_model

        self.output = nn.Sequential(
            nn.Linear(fusion_in, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim),
        )

    def forward(
        self,
        seq_item_vectors: torch.Tensor,
        seq_event_ids: torch.Tensor,
        seq_time_deltas_hours: torch.Tensor,
        seq_padding_mask: torch.Tensor,
        user_dense_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # seq_item_vectors: [B, L, item_input_dim]
        # seq_event_ids: [B, L]
        # seq_time_deltas_hours: [B, L]
        # seq_padding_mask: [B, L] True where padded
        event_vec = self.event_embedding(seq_event_ids)

        # log1p stabilizes long-tail recency distances
        time_input = torch.log1p(seq_time_deltas_hours).unsqueeze(-1)
        time_vec = self.time_mlp(time_input)

        x = torch.cat([seq_item_vectors, event_vec, time_vec], dim=-1)
        x = self.input_projection(x)
        x = self.position(x)

        x = self.encoder(x, src_key_padding_mask=seq_padding_mask)
        pooled = self.pooling(x, seq_padding_mask)

        if self.user_feat_mlp is not None and user_dense_features is not None:
            dense_vec = self.user_feat_mlp(user_dense_features)
            pooled = torch.cat([pooled, dense_vec], dim=-1)

        out = self.output(pooled)
        return F.normalize(out, p=2, dim=-1)


class ItemTower(nn.Module):
    """Encodes item multi-modal/categorical features into an item embedding."""

    def __init__(
        self,
        num_items: int,
        image_dim: int,
        text_dim: int,
        num_categories: int,
        num_brands: int,
        output_dim: int,
        d_model: int = 256,
        id_emb_dim: int = 64,
        cat_emb_dim: int = 16,
        brand_emb_dim: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.item_id_embedding = nn.Embedding(num_items + 1, id_emb_dim, padding_idx=PAD_ITEM_ID)
        self.category_embedding = nn.Embedding(num_categories + 1, cat_emb_dim, padding_idx=0)
        self.brand_embedding = nn.Embedding(num_brands + 1, brand_emb_dim, padding_idx=0)

        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        fusion_in = d_model + d_model + id_emb_dim + cat_emb_dim + brand_emb_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim),
        )

    def forward(
        self,
        item_ids: torch.Tensor,
        image_emb: torch.Tensor,
        text_emb: torch.Tensor,
        category_ids: torch.Tensor,
        brand_ids: torch.Tensor,
    ) -> torch.Tensor:
        id_vec = self.item_id_embedding(item_ids)
        cat_vec = self.category_embedding(category_ids)
        brand_vec = self.brand_embedding(brand_ids)
        img_vec = self.image_proj(image_emb)
        txt_vec = self.text_proj(text_emb)

        x = torch.cat([img_vec, txt_vec, id_vec, cat_vec, brand_vec], dim=-1)
        out = self.fusion(x)
        return F.normalize(out, p=2, dim=-1)


@dataclass
class TwoTowerBatch:
    seq_item_vectors: torch.Tensor
    seq_event_ids: torch.Tensor
    seq_time_deltas_hours: torch.Tensor
    seq_padding_mask: torch.Tensor
    user_dense_features: Optional[torch.Tensor]

    pos_item_ids: torch.Tensor
    pos_item_image_emb: torch.Tensor
    pos_item_text_emb: torch.Tensor
    pos_item_category_ids: torch.Tensor
    pos_item_brand_ids: torch.Tensor


class TwoTowerModel(nn.Module):
    """Container module with convenience methods for training/retrieval."""

    def __init__(self, user_tower: UserTower, item_tower: ItemTower, temperature: float = 0.07):
        super().__init__()
        self.user_tower = user_tower
        self.item_tower = item_tower
        self.temperature = temperature

    def encode_user(
        self,
        seq_item_vectors: torch.Tensor,
        seq_event_ids: torch.Tensor,
        seq_time_deltas_hours: torch.Tensor,
        seq_padding_mask: torch.Tensor,
        user_dense_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.user_tower(
            seq_item_vectors=seq_item_vectors,
            seq_event_ids=seq_event_ids,
            seq_time_deltas_hours=seq_time_deltas_hours,
            seq_padding_mask=seq_padding_mask,
            user_dense_features=user_dense_features,
        )

    def encode_item(
        self,
        item_ids: torch.Tensor,
        image_emb: torch.Tensor,
        text_emb: torch.Tensor,
        category_ids: torch.Tensor,
        brand_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.item_tower(
            item_ids=item_ids,
            image_emb=image_emb,
            text_emb=text_emb,
            category_ids=category_ids,
            brand_ids=brand_ids,
        )

    def forward(self, batch: TwoTowerBatch) -> tuple[torch.Tensor, torch.Tensor]:
        user_emb = self.encode_user(
            seq_item_vectors=batch.seq_item_vectors,
            seq_event_ids=batch.seq_event_ids,
            seq_time_deltas_hours=batch.seq_time_deltas_hours,
            seq_padding_mask=batch.seq_padding_mask,
            user_dense_features=batch.user_dense_features,
        )
        pos_item_emb = self.encode_item(
            item_ids=batch.pos_item_ids,
            image_emb=batch.pos_item_image_emb,
            text_emb=batch.pos_item_text_emb,
            category_ids=batch.pos_item_category_ids,
            brand_ids=batch.pos_item_brand_ids,
        )
        return user_emb, pos_item_emb

    def inbatch_infonce_loss(self, user_emb: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        # user_emb: [B, D], item_emb: [B, D]
        logits = torch.matmul(user_emb, item_emb.transpose(0, 1)) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)

    @staticmethod
    def dot_similarity(user_emb: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        return torch.sum(user_emb * item_emb, dim=-1)
