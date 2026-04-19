"""Training pipeline for Two-Tower retrieval model.

Includes:
- PyTorch training loop with in-batch InfoNCE loss
- Event-weighted sample contribution
- Temporal split utility
- Retrieval metrics on validation (Recall@K, NDCG@K)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from model import ItemTower, TwoTowerModel, UserTower
from dataset import (
    IndexMaps,
    ItemFeatureTensors,
    TwoTowerTrainDataset,
    build_index_maps,
    build_item_feature_tensors,
    collate_train_batch,
)


@dataclass
class TrainConfig:
    embedding_dim: int = 128
    d_model: int = 256
    max_seq_len: int = 50
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 8
    num_workers: int = 0
    temperature: float = 0.07
    min_history: int = 2
    lookback_days: int = 180
    include_view_targets: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class TwoTowerTrainer:
    def __init__(
        self,
        model: TwoTowerModel,
        item_features: ItemFeatureTensors,
        idx_maps: IndexMaps,
        config: TrainConfig,
    ):
        self.model = model.to(config.device)
        self.item_features = item_features
        self.idx_maps = idx_maps
        self.config = config
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

    def _move_batch(self, batch: dict) -> dict:
        out = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                out[k] = v.to(self.config.device)
            else:
                out[k] = v
        return out

    def _encode_positive_items(self, item_idx: torch.Tensor) -> torch.Tensor:
        item_idx_cpu = item_idx.detach().cpu()
        image = self.item_features.image_embeddings[item_idx_cpu].to(self.config.device)
        text = self.item_features.text_embeddings[item_idx_cpu].to(self.config.device)
        cat = self.item_features.category_ids[item_idx_cpu].to(self.config.device)
        brand = self.item_features.brand_ids[item_idx_cpu].to(self.config.device)
        return self.model.encode_item(item_idx, image, text, cat, brand)

    def _weighted_inbatch_loss(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
        sample_weights: torch.Tensor,
    ) -> torch.Tensor:
        logits = torch.matmul(user_emb, item_emb.transpose(0, 1)) / self.model.temperature
        targets = torch.arange(logits.size(0), device=logits.device)
        per_sample_loss = torch.nn.functional.cross_entropy(
            logits,
            targets,
            reduction="none",
        )
        weighted = per_sample_loss * sample_weights
        return weighted.mean()

    def fit(self, train_loader: DataLoader, val_loader: DataLoader | None = None) -> None:
        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            losses: List[float] = []
            for batch in train_loader:
                batch = self._move_batch(batch)
                user_emb = self.model.encode_user(
                    seq_item_vectors=batch["seq_item_vectors"],
                    seq_event_ids=batch["seq_event_ids"],
                    seq_time_deltas_hours=batch["seq_time_deltas_hours"],
                    seq_padding_mask=batch["seq_padding_mask"],
                    user_dense_features=batch["user_dense_features"],
                )
                pos_item_emb = self._encode_positive_items(batch["target_item_idx"])
                loss = self._weighted_inbatch_loss(
                    user_emb,
                    pos_item_emb,
                    batch["target_weight"],
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                self.optimizer.step()

                losses.append(float(loss.item()))

            avg_loss = float(np.mean(losses)) if losses else 0.0
            print(f"Epoch {epoch:02d}/{self.config.epochs} | train_loss={avg_loss:.4f}")

            if val_loader is not None:
                metrics = self.evaluate(val_loader, ks=(10, 50, 100))
                pretty = " | ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
                print(f"  Val: {pretty}")

    @torch.no_grad()
    def compute_item_embedding_table(self, batch_size: int = 4096) -> Tuple[np.ndarray, List[str]]:
        self.model.eval()
        all_vecs: List[np.ndarray] = []

        item_ids = self.item_features.item_ids.numpy()
        valid_item_idx = item_ids[item_ids > 0]

        for start in range(0, len(valid_item_idx), batch_size):
            idx_np = valid_item_idx[start : start + batch_size]
            idx_t = torch.tensor(idx_np, dtype=torch.long, device=self.config.device)
            img = self.item_features.image_embeddings[idx_np].to(self.config.device)
            txt = self.item_features.text_embeddings[idx_np].to(self.config.device)
            cat = self.item_features.category_ids[idx_np].to(self.config.device)
            brand = self.item_features.brand_ids[idx_np].to(self.config.device)

            vec = self.model.encode_item(idx_t, img, txt, cat, brand)
            all_vecs.append(vec.detach().cpu().numpy())

        matrix = np.concatenate(all_vecs, axis=0)
        ids = [self.idx_maps.idx_to_item[int(i)] for i in valid_item_idx]
        return matrix, ids

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader, ks: tuple[int, ...] = (10, 50)) -> Dict[str, float]:
        self.model.eval()

        item_matrix, _ = self.compute_item_embedding_table()
        # item index in this matrix maps to real item index via offset +1.
        # To score against all items, matrix rows correspond to item_idx 1..N.

        recalls = {k: [] for k in ks}
        ndcgs = {k: [] for k in ks}

        for batch in val_loader:
            batch = self._move_batch(batch)

            user_emb = self.model.encode_user(
                seq_item_vectors=batch["seq_item_vectors"],
                seq_event_ids=batch["seq_event_ids"],
                seq_time_deltas_hours=batch["seq_time_deltas_hours"],
                seq_padding_mask=batch["seq_padding_mask"],
                user_dense_features=batch["user_dense_features"],
            ).detach().cpu().numpy()

            targets = batch["target_item_idx"].detach().cpu().numpy()
            scores = np.matmul(user_emb, item_matrix.T)

            for i in range(scores.shape[0]):
                target_idx = int(targets[i])
                target_col = target_idx - 1
                if target_col < 0 or target_col >= scores.shape[1]:
                    continue

                row = scores[i]
                order = np.argsort(row)[::-1]

                for k in ks:
                    topk = order[:k]
                    hit = float(target_col in topk)
                    recalls[k].append(hit)

                    if hit:
                        rank_pos = np.where(topk == target_col)[0][0] + 1
                        dcg = 1.0 / np.log2(rank_pos + 1)
                    else:
                        dcg = 0.0
                    ndcgs[k].append(float(dcg))

        metrics: Dict[str, float] = {}
        for k in ks:
            metrics[f"Recall@{k}"] = float(np.mean(recalls[k])) if recalls[k] else 0.0
            metrics[f"NDCG@{k}"] = float(np.mean(ndcgs[k])) if ndcgs[k] else 0.0
        return metrics


def temporal_split_events(events: pd.DataFrame, val_fraction: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    events = events.copy()
    events["timestamp"] = pd.to_datetime(events["timestamp"])
    cutoff = events["timestamp"].quantile(1.0 - val_fraction)
    train_df = events[events["timestamp"] < cutoff].copy()
    val_df = events[events["timestamp"] >= cutoff].copy()
    return train_df, val_df


def build_model(
    maps: IndexMaps,
    item_features: ItemFeatureTensors,
    config: TrainConfig,
) -> TwoTowerModel:
    image_dim = item_features.image_embeddings.shape[1]
    text_dim = item_features.text_embeddings.shape[1]

    user_tower = UserTower(
        item_input_dim=image_dim + text_dim,
        event_vocab_size=5,
        output_dim=config.embedding_dim,
        user_feat_dim=3,
        d_model=config.d_model,
        max_seq_len=config.max_seq_len,
    )
    item_tower = ItemTower(
        num_items=len(maps.item_to_idx),
        image_dim=image_dim,
        text_dim=text_dim,
        num_categories=len(maps.category_to_idx),
        num_brands=len(maps.brand_to_idx),
        output_dim=config.embedding_dim,
        d_model=config.d_model,
    )
    return TwoTowerModel(user_tower=user_tower, item_tower=item_tower, temperature=config.temperature)


def train_two_tower(
    products: pd.DataFrame,
    events: pd.DataFrame,
    image_embeddings: Dict[str, np.ndarray],
    text_embeddings: Dict[str, np.ndarray],
    config: TrainConfig,
) -> tuple[TwoTowerModel, TwoTowerTrainer, dict]:
    maps = build_index_maps(products)
    item_features = build_item_feature_tensors(products, image_embeddings, text_embeddings, maps)

    train_events, val_events = temporal_split_events(events)

    train_ds = TwoTowerTrainDataset(
        events=train_events,
        products=products,
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
        maps=maps,
        max_seq_len=config.max_seq_len,
        min_history=config.min_history,
        lookback_days=config.lookback_days,
        include_view_targets=config.include_view_targets,
    )
    val_ds = TwoTowerTrainDataset(
        events=val_events,
        products=products,
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
        maps=maps,
        max_seq_len=config.max_seq_len,
        min_history=config.min_history,
        lookback_days=config.lookback_days,
        include_view_targets=config.include_view_targets,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_train_batch,
        pin_memory=(config.device.startswith("cuda")),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_train_batch,
        pin_memory=(config.device.startswith("cuda")),
    )

    model = build_model(maps, item_features, config)
    trainer = TwoTowerTrainer(model, item_features, maps, config)
    trainer.fit(train_loader, val_loader)

    final_metrics = trainer.evaluate(val_loader, ks=(10, 50, 100))
    return model, trainer, final_metrics


def save_two_tower_checkpoint(
    path: str,
    model: TwoTowerModel,
    maps: IndexMaps,
    config: TrainConfig,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "maps": maps,
        "config": config,
    }
    torch.save(payload, path)


def load_two_tower_checkpoint(path: str, model: TwoTowerModel) -> dict:
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model_state_dict"])
    return payload
