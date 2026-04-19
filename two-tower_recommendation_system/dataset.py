"""Dataset and feature tensor builders for Two-Tower training.

This module prepares:
- Item-side tensors (image/text/category/brand) indexed by item integer IDs
- Sequence training examples from user event logs
- Collate function with padding and mask creation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


EVENT_TYPE_TO_ID = {
    "pad": 0,
    "click": 1,
    "add_to_cart": 2,
    "purchase": 3,
    "view": 4,
}

EVENT_TYPE_WEIGHT = {
    "click": 1.0,
    "add_to_cart": 2.0,
    "purchase": 3.0,
    "view": 0.25,
}


@dataclass
class ItemFeatureTensors:
    item_ids: torch.Tensor
    image_embeddings: torch.Tensor
    text_embeddings: torch.Tensor
    category_ids: torch.Tensor
    brand_ids: torch.Tensor


@dataclass
class IndexMaps:
    item_to_idx: Dict[str, int]
    idx_to_item: Dict[int, str]
    category_to_idx: Dict[str, int]
    brand_to_idx: Dict[str, int]



def build_index_maps(products: pd.DataFrame) -> IndexMaps:
    products = products.copy()
    products["item_id"] = products["item_id"].astype(str)
    products["category"] = products["category"].fillna("unknown").astype(str)
    products["brand"] = products["brand"].fillna("unknown").astype(str)

    unique_items = products["item_id"].drop_duplicates().tolist()
    item_to_idx = {item_id: i + 1 for i, item_id in enumerate(unique_items)}
    idx_to_item = {i + 1: item_id for i, item_id in enumerate(unique_items)}

    categories = sorted(products["category"].unique().tolist())
    brands = sorted(products["brand"].unique().tolist())

    category_to_idx = {"unknown": 1}
    for c in categories:
        if c not in category_to_idx:
            category_to_idx[c] = len(category_to_idx) + 1

    brand_to_idx = {"unknown": 1}
    for b in brands:
        if b not in brand_to_idx:
            brand_to_idx[b] = len(brand_to_idx) + 1

    return IndexMaps(
        item_to_idx=item_to_idx,
        idx_to_item=idx_to_item,
        category_to_idx=category_to_idx,
        brand_to_idx=brand_to_idx,
    )


def build_item_feature_tensors(
    products: pd.DataFrame,
    image_embeddings: Dict[str, np.ndarray],
    text_embeddings: Dict[str, np.ndarray],
    maps: IndexMaps,
) -> ItemFeatureTensors:
    """Build aligned item tensors indexed by integer item index.

    Index 0 is reserved as padding; real items start at 1.
    """
    products = products.copy()
    products["item_id"] = products["item_id"].astype(str)
    products = products.drop_duplicates(subset=["item_id"]).set_index("item_id")

    max_idx = max(maps.item_to_idx.values()) if maps.item_to_idx else 0

    sample_item = next(iter(image_embeddings.keys()))
    image_dim = len(image_embeddings[sample_item])
    sample_item_text = next(iter(text_embeddings.keys()))
    text_dim = len(text_embeddings[sample_item_text])

    item_ids = torch.arange(0, max_idx + 1, dtype=torch.long)
    img = torch.zeros((max_idx + 1, image_dim), dtype=torch.float32)
    txt = torch.zeros((max_idx + 1, text_dim), dtype=torch.float32)
    category_ids = torch.zeros(max_idx + 1, dtype=torch.long)
    brand_ids = torch.zeros(max_idx + 1, dtype=torch.long)

    for item_id, idx in maps.item_to_idx.items():
        if item_id not in products.index:
            continue
        row = products.loc[item_id]

        if item_id in image_embeddings:
            img[idx] = torch.tensor(image_embeddings[item_id], dtype=torch.float32)
        if item_id in text_embeddings:
            txt[idx] = torch.tensor(text_embeddings[item_id], dtype=torch.float32)

        cat = str(row.get("category", "unknown"))
        brand = str(row.get("brand", "unknown"))
        category_ids[idx] = maps.category_to_idx.get(cat, maps.category_to_idx["unknown"])
        brand_ids[idx] = maps.brand_to_idx.get(brand, maps.brand_to_idx["unknown"])

    return ItemFeatureTensors(
        item_ids=item_ids,
        image_embeddings=img,
        text_embeddings=txt,
        category_ids=category_ids,
        brand_ids=brand_ids,
    )


class TwoTowerTrainDataset(Dataset):
    """Builds (user history -> next interacted item) training pairs.

    Target events are click/add_to_cart/purchase by default.
    Views can be included via include_view_targets=True.
    """

    def __init__(
        self,
        events: pd.DataFrame,
        products: pd.DataFrame,
        image_embeddings: Dict[str, np.ndarray],
        text_embeddings: Dict[str, np.ndarray],
        maps: IndexMaps,
        max_seq_len: int = 50,
        min_history: int = 2,
        lookback_days: int = 180,
        include_view_targets: bool = False,
    ):
        super().__init__()
        self.events = events.copy()
        self.events["timestamp"] = pd.to_datetime(self.events["timestamp"])
        self.events["item_id"] = self.events["item_id"].astype(str)
        self.events = self.events.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

        self.products = products.copy().drop_duplicates(subset=["item_id"]).set_index("item_id")
        self.image_embeddings = image_embeddings
        self.text_embeddings = text_embeddings
        self.maps = maps

        self.max_seq_len = max_seq_len
        self.min_history = min_history
        self.lookback_days = lookback_days
        self.include_view_targets = include_view_targets

        self.item_input_dim = self._infer_item_input_dim()
        self.samples = self._build_samples()

    def _infer_item_input_dim(self) -> int:
        sample = next(iter(self.image_embeddings.values()))
        return len(sample) + len(next(iter(self.text_embeddings.values())))

    def _event_allowed_as_target(self, event_type: str) -> bool:
        if self.include_view_targets:
            return event_type in {"view", "click", "add_to_cart", "purchase"}
        return event_type in {"click", "add_to_cart", "purchase"}

    def _build_samples(self) -> List[dict]:
        samples: List[dict] = []
        for user_id, grp in self.events.groupby("user_id"):
            grp = grp.sort_values("timestamp").reset_index(drop=True)
            for i in range(len(grp)):
                target = grp.iloc[i]
                target_item_id = target["item_id"]
                target_type = target["event_type"]
                target_ts = target["timestamp"]

                if not self._event_allowed_as_target(target_type):
                    continue
                if target_item_id not in self.maps.item_to_idx:
                    continue

                hist = grp.iloc[:i]
                if hist.empty:
                    continue

                cutoff = target_ts - pd.Timedelta(days=self.lookback_days)
                hist = hist[hist["timestamp"] >= cutoff]
                if len(hist) < self.min_history:
                    continue

                hist = hist.iloc[-self.max_seq_len:]

                hist_item_vectors = []
                hist_item_ids = []
                hist_event_ids = []
                hist_time_deltas = []

                for _, h in hist.iterrows():
                    item_id = h["item_id"]
                    if item_id not in self.image_embeddings or item_id not in self.text_embeddings:
                        continue
                    if item_id not in self.maps.item_to_idx:
                        continue
                    evt = str(h["event_type"])
                    delta_hours = max((target_ts - h["timestamp"]).total_seconds() / 3600.0, 0.0)

                    item_vec = np.concatenate([
                        np.asarray(self.image_embeddings[item_id], dtype=np.float32),
                        np.asarray(self.text_embeddings[item_id], dtype=np.float32),
                    ])
                    hist_item_vectors.append(item_vec)
                    hist_item_ids.append(self.maps.item_to_idx[item_id])
                    hist_event_ids.append(EVENT_TYPE_TO_ID.get(evt, EVENT_TYPE_TO_ID["view"]))
                    hist_time_deltas.append(delta_hours)

                if len(hist_item_vectors) < self.min_history:
                    continue

                avg_spend = self._user_avg_spend(hist)
                recent_activity_count = float(len(hist))
                recency_hours = hist_time_deltas[-1] if hist_time_deltas else 0.0
                user_dense_features = np.array(
                    [avg_spend, recent_activity_count, recency_hours],
                    dtype=np.float32,
                )

                samples.append(
                    {
                        "user_id": user_id,
                        "seq_item_vectors": np.stack(hist_item_vectors),
                        "seq_item_ids": np.asarray(hist_item_ids, dtype=np.int64),
                        "seq_event_ids": np.asarray(hist_event_ids, dtype=np.int64),
                        "seq_time_deltas_hours": np.asarray(hist_time_deltas, dtype=np.float32),
                        "user_dense_features": user_dense_features,
                        "target_item_idx": self.maps.item_to_idx[target_item_id],
                        "target_event_type": str(target_type),
                        "target_weight": float(EVENT_TYPE_WEIGHT.get(str(target_type), 1.0)),
                        "target_timestamp": target_ts,
                    }
                )
        return samples

    def _user_avg_spend(self, hist: pd.DataFrame) -> float:
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

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


def collate_train_batch(batch: List[dict]) -> dict:
    """Pad variable-length sequence fields and stack item targets."""
    batch_size = len(batch)
    seq_lens = [len(x["seq_event_ids"]) for x in batch]
    max_len = max(seq_lens)

    item_input_dim = batch[0]["seq_item_vectors"].shape[-1]

    seq_item_vectors = torch.zeros((batch_size, max_len, item_input_dim), dtype=torch.float32)
    seq_event_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    seq_time_deltas = torch.zeros((batch_size, max_len), dtype=torch.float32)
    seq_padding_mask = torch.ones((batch_size, max_len), dtype=torch.bool)

    user_dense_features = torch.zeros((batch_size, 3), dtype=torch.float32)

    target_item_idx = torch.zeros(batch_size, dtype=torch.long)
    target_weight = torch.ones(batch_size, dtype=torch.float32)

    user_ids: List[str] = []

    for i, row in enumerate(batch):
        l = len(row["seq_event_ids"])
        seq_item_vectors[i, :l] = torch.tensor(row["seq_item_vectors"], dtype=torch.float32)
        seq_event_ids[i, :l] = torch.tensor(row["seq_event_ids"], dtype=torch.long)
        seq_time_deltas[i, :l] = torch.tensor(row["seq_time_deltas_hours"], dtype=torch.float32)
        seq_padding_mask[i, :l] = False

        user_dense_features[i] = torch.tensor(row["user_dense_features"], dtype=torch.float32)
        target_item_idx[i] = int(row["target_item_idx"])
        target_weight[i] = float(row["target_weight"])
        user_ids.append(str(row["user_id"]))

    return {
        "seq_item_vectors": seq_item_vectors,
        "seq_event_ids": seq_event_ids,
        "seq_time_deltas_hours": seq_time_deltas,
        "seq_padding_mask": seq_padding_mask,
        "user_dense_features": user_dense_features,
        "target_item_idx": target_item_idx,
        "target_weight": target_weight,
        "user_ids": user_ids,
    }
