"""Integration helpers to plug Two-Tower retrieval into an existing ranker stage.

Expected flow:
1) Compute neural user embedding from user interaction sequence.
2) Retrieve top-N candidates via FAISS/brute-force on learned item embeddings.
3) Build ranker feature rows (existing handcrafted features + neural_similarity).
4) Score with LightGBM/XGBoost ranker and return top-K recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from model import TwoTowerModel
from retrieval import ItemRetriever
from dataset import EVENT_TYPE_TO_ID


EVENT_PRIORITY = {"purchase": 3.0, "add_to_cart": 2.0, "click": 1.0, "view": 0.25}


@dataclass
class UserSequencePayload:
    seq_item_vectors: torch.Tensor
    seq_event_ids: torch.Tensor
    seq_time_deltas_hours: torch.Tensor
    seq_padding_mask: torch.Tensor
    user_dense_features: torch.Tensor


class TwoTowerRankerIntegrator:
    """Drop-in orchestrator for neural retrieval + tree-based ranking."""

    def __init__(
        self,
        model: TwoTowerModel,
        retriever: ItemRetriever,
        products: pd.DataFrame,
        user_events: pd.DataFrame,
        image_embeddings: Dict[str, np.ndarray],
        text_embeddings: Dict[str, np.ndarray],
        ranker_model,
        feature_builder_fn,
        max_seq_len: int = 50,
        lookback_days: int = 180,
        device: str = "cpu",
    ):
        """
        feature_builder_fn signature:
            (user_id: str, candidate_df: pd.DataFrame) -> pd.DataFrame
        It should return all non-neural ranking features required by ranker_model.
        """
        self.model = model.to(device)
        self.model.eval()
        self.retriever = retriever
        self.products = products.copy().drop_duplicates(subset=["item_id"]).set_index("item_id")
        self.user_events = user_events.copy()
        self.user_events["timestamp"] = pd.to_datetime(self.user_events["timestamp"])
        self.user_events["item_id"] = self.user_events["item_id"].astype(str)

        self.image_embeddings = image_embeddings
        self.text_embeddings = text_embeddings
        self.ranker_model = ranker_model
        self.feature_builder_fn = feature_builder_fn
        self.max_seq_len = max_seq_len
        self.lookback_days = lookback_days
        self.device = device

    @torch.no_grad()
    def _encode_user(self, user_payload: UserSequencePayload) -> np.ndarray:
        user_emb = self.model.encode_user(
            seq_item_vectors=user_payload.seq_item_vectors.to(self.device),
            seq_event_ids=user_payload.seq_event_ids.to(self.device),
            seq_time_deltas_hours=user_payload.seq_time_deltas_hours.to(self.device),
            seq_padding_mask=user_payload.seq_padding_mask.to(self.device),
            user_dense_features=user_payload.user_dense_features.to(self.device),
        )
        return user_emb.squeeze(0).detach().cpu().numpy().astype(np.float32)

    def _build_user_payload(self, user_id: str) -> UserSequencePayload:
        hist = self.user_events[self.user_events["user_id"] == user_id].sort_values("timestamp")
        if hist.empty:
            raise ValueError(f"No interaction history for user_id={user_id}")

        now = hist["timestamp"].max()
        cutoff = now - pd.Timedelta(days=self.lookback_days)
        hist = hist[hist["timestamp"] >= cutoff].tail(self.max_seq_len)

        seq_vecs = []
        seq_event_ids = []
        seq_time_deltas = []

        for _, row in hist.iterrows():
            item_id = str(row["item_id"])
            if item_id not in self.image_embeddings or item_id not in self.text_embeddings:
                continue

            vec = np.concatenate([
                np.asarray(self.image_embeddings[item_id], dtype=np.float32),
                np.asarray(self.text_embeddings[item_id], dtype=np.float32),
            ])
            delta_h = max((now - row["timestamp"]).total_seconds() / 3600.0, 0.0)

            seq_vecs.append(vec)
            seq_event_ids.append(EVENT_TYPE_TO_ID.get(str(row["event_type"]), EVENT_TYPE_TO_ID["view"]))
            seq_time_deltas.append(delta_h)

        if len(seq_vecs) == 0:
            raise ValueError(f"No valid embedding-backed history for user_id={user_id}")

        seq_arr = np.stack(seq_vecs)
        l = seq_arr.shape[0]

        seq_item_vectors = torch.tensor(seq_arr, dtype=torch.float32).unsqueeze(0)
        seq_event_ids_t = torch.tensor(seq_event_ids, dtype=torch.long).unsqueeze(0)
        seq_time_t = torch.tensor(seq_time_deltas, dtype=torch.float32).unsqueeze(0)
        seq_padding_mask = torch.zeros((1, l), dtype=torch.bool)

        avg_spend = self._avg_spend(hist)
        user_dense = torch.tensor([[avg_spend, float(l), float(seq_time_deltas[-1])]], dtype=torch.float32)

        return UserSequencePayload(
            seq_item_vectors=seq_item_vectors,
            seq_event_ids=seq_event_ids_t,
            seq_time_deltas_hours=seq_time_t,
            seq_padding_mask=seq_padding_mask,
            user_dense_features=user_dense,
        )

    def _avg_spend(self, hist: pd.DataFrame) -> float:
        purchases = hist[hist["event_type"] == "purchase"]
        if purchases.empty:
            return 0.0
        merged = purchases.merge(
            self.products[["price"]],
            left_on="item_id",
            right_index=True,
            how="left",
        )
        return float(merged["price"].fillna(0.0).mean())

    def _exclude_purchased(self, user_id: str) -> set:
        hist = self.user_events[self.user_events["user_id"] == user_id]
        return set(hist.loc[hist["event_type"] == "purchase", "item_id"].astype(str).tolist())

    def recommend(
        self,
        user_id: str,
        top_n_retrieve: int = 200,
        top_k: int = 10,
    ) -> pd.DataFrame:
        """recommend(user_id): neural retrieval -> ranker scoring -> top recommendations."""
        payload = self._build_user_payload(user_id)
        user_emb = self._encode_user(payload)

        exclude = self._exclude_purchased(user_id)
        retrieved = self.retriever.search(
            user_embedding=user_emb,
            top_k=top_n_retrieve,
            exclude_item_ids=exclude,
        )

        candidate_df = pd.DataFrame(retrieved, columns=["item_id", "neural_similarity"])
        if candidate_df.empty:
            return pd.DataFrame(columns=[
                "rank", "item_id", "product_name", "category", "price", "neural_similarity", "ranking_score"
            ])

        # Build existing ranker features (price/popularity/brand match/etc.)
        feature_df = self.feature_builder_fn(user_id, candidate_df)
        feature_df = feature_df.copy()
        feature_df["neural_similarity"] = candidate_df["neural_similarity"].values

        # IMPORTANT: ranker input columns are the feature dataframe columns by contract.
        rank_scores = self.ranker_model.predict(feature_df)
        out = candidate_df.copy()
        out["ranking_score"] = rank_scores

        out = out.sort_values("ranking_score", ascending=False).head(top_k).reset_index(drop=True)
        out["rank"] = np.arange(1, len(out) + 1)

        enriched = out.merge(
            self.products[["product_name", "category", "price"]],
            left_on="item_id",
            right_index=True,
            how="left",
        )

        return enriched[[
            "rank",
            "item_id",
            "product_name",
            "category",
            "price",
            "neural_similarity",
            "ranking_score",
        ]]


def append_neural_similarity_feature(
    features_df: pd.DataFrame,
    user_embedding: np.ndarray,
    candidate_item_embeddings: np.ndarray,
    col_name: str = "neural_similarity",
) -> pd.DataFrame:
    """Utility for batch feature generation in existing ranking pipelines."""
    sim = np.sum(user_embedding[None, :] * candidate_item_embeddings, axis=1)
    out = features_df.copy()
    out[col_name] = sim.astype(np.float32)
    return out
