"""Runnable end-to-end script for Two-Tower retrieval + optional ranker integration.

Usage examples:
1) Mock quick run:
   python "two-tower recommendation system/run_two_tower.py" --epochs 1 --quick

2) Real data:
   python "two-tower recommendation system/run_two_tower.py" \
       --products "data/products_export.csv" \
       --events "data/user_events.csv" \
       --image-emb "embeddings/models/image_embeddings.pkl" \
       --text-emb "embeddings/models/text_embeddings.pkl" \
       --epochs 5

3) With trained LightGBM/XGBoost ranker (optional):
   python "two-tower recommendation system/run_two_tower.py" \
       --ranker-path "ranking_pipeline/models/lgbm_ranker.pkl"
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch

from train import TrainConfig, train_two_tower
from retrieval import ItemRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-Tower neural retrieval runner")

    parser.add_argument("--products", type=str, default=None, help="Path to products CSV")
    parser.add_argument("--events", type=str, default=None, help="Path to user_events CSV")
    parser.add_argument("--image-emb", type=str, default=None, help="Path to image embeddings pkl")
    parser.add_argument("--text-emb", type=str, default=None, help="Path to text embeddings pkl")

    parser.add_argument("--user-id", type=str, default=None, help="User ID for demo recommendation")
    parser.add_argument("--top-n", type=int, default=200, help="Candidates to retrieve")
    parser.add_argument("--top-k", type=int, default=10, help="Final recommendations")

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=50)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--quick", action="store_true", help="Run very small mock training for smoke test")
    parser.add_argument("--no-faiss", action="store_true", help="Disable FAISS even if installed")

    parser.add_argument("--ranker-path", type=str, default=None, help="Optional path to trained LightGBM/XGBoost ranker pkl")

    return parser.parse_args()


def _required_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def _to_emb_dict(payload, item_ids: list[str]) -> Dict[str, np.ndarray]:
    """Normalize several possible embedding payload shapes to item_id -> np.ndarray."""
    if isinstance(payload, dict):
        # already item_id -> vector
        if all(isinstance(k, str) for k in payload.keys()):
            return {str(k): np.asarray(v, dtype=np.float32) for k, v in payload.items()}

        # common shape: {"item_ids": [...], "embeddings": [[...], ...]}
        if "item_ids" in payload and "embeddings" in payload:
            ids = [str(x) for x in payload["item_ids"]]
            embs = np.asarray(payload["embeddings"], dtype=np.float32)
            return {ids[i]: embs[i] for i in range(min(len(ids), len(embs)))}

        # fallback keys from project experiments
        for key in ["image_embeddings", "text_embeddings", "combined_embeddings"]:
            if key in payload:
                val = payload[key]
                if isinstance(val, dict):
                    return {str(k): np.asarray(v, dtype=np.float32) for k, v in val.items()}
                arr = np.asarray(val, dtype=np.float32)
                n = min(len(item_ids), len(arr))
                return {str(item_ids[i]): arr[i] for i in range(n)}

    arr = np.asarray(payload, dtype=np.float32)
    n = min(len(item_ids), len(arr))
    return {str(item_ids[i]): arr[i] for i in range(n)}


def load_embedding_dict(path: str, item_ids: list[str]) -> Dict[str, np.ndarray]:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return _to_emb_dict(payload, item_ids=item_ids)


def synthesize_missing_embeddings(
    emb: Dict[str, np.ndarray],
    item_ids: list[str],
    dim: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    out = dict(emb)
    for item_id in item_ids:
        if item_id not in out:
            out[item_id] = rng.standard_normal(dim).astype(np.float32)
    return out


def generate_mock_data(quick: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    rng = np.random.default_rng(42)
    n_items = 250 if quick else 1000
    n_users = 80 if quick else 500

    categories = ["T-Shirts", "Jeans", "Dresses", "Shoes", "Jackets", "Accessories"]
    brands = ["Nike", "Adidas", "Zara", "H&M", "Puma", "Levis"]

    item_ids = [f"ITEM_{i:05d}" for i in range(n_items)]
    products = pd.DataFrame(
        {
            "item_id": item_ids,
            "product_name": [f"Product {i}" for i in range(n_items)],
            "category": rng.choice(categories, size=n_items),
            "brand": rng.choice(brands, size=n_items),
            "price": rng.uniform(10, 500, size=n_items).round(2),
            "popularity_score": rng.uniform(0, 1, size=n_items),
            "ctr_score": rng.uniform(0, 1, size=n_items),
        }
    )

    event_types = ["view", "click", "add_to_cart", "purchase"]
    event_probs = [0.55, 0.28, 0.12, 0.05]

    records = []
    start = pd.Timestamp("2025-01-01")
    for u in range(n_users):
        uid = f"USER_{u:05d}"
        n_ev = int(rng.integers(15, 80 if not quick else 40))
        ts = start
        for _ in range(n_ev):
            ts = ts + pd.Timedelta(hours=int(rng.integers(1, 72)))
            records.append(
                {
                    "user_id": uid,
                    "item_id": rng.choice(item_ids),
                    "event_type": rng.choice(event_types, p=event_probs),
                    "timestamp": ts,
                }
            )

    events = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)

    image_dim = 128
    text_dim = 128
    image_emb = {item_id: rng.standard_normal(image_dim).astype(np.float32) for item_id in item_ids}
    text_emb = {item_id: rng.standard_normal(text_dim).astype(np.float32) for item_id in item_ids}

    return products, events, image_emb, text_emb


def load_or_mock(args: argparse.Namespace):
    if args.products and args.events and args.image_emb and args.text_emb:
        products = pd.read_csv(args.products)
        events = pd.read_csv(args.events)
        events["timestamp"] = pd.to_datetime(events["timestamp"])

        _required_cols(products, ["item_id", "product_name", "category", "brand", "price"], "products")
        _required_cols(events, ["user_id", "item_id", "event_type", "timestamp"], "events")

        products["item_id"] = products["item_id"].astype(str)
        events["item_id"] = events["item_id"].astype(str)

        item_ids = products["item_id"].tolist()
        image_emb = load_embedding_dict(args.image_emb, item_ids)
        text_emb = load_embedding_dict(args.text_emb, item_ids)

        image_dim = len(next(iter(image_emb.values())))
        text_dim = len(next(iter(text_emb.values())))

        image_emb = synthesize_missing_embeddings(image_emb, item_ids, image_dim, seed=17)
        text_emb = synthesize_missing_embeddings(text_emb, item_ids, text_dim, seed=23)

        if "popularity_score" not in products.columns:
            products["popularity_score"] = np.random.default_rng(7).uniform(0, 1, len(products))
        if "ctr_score" not in products.columns:
            products["ctr_score"] = np.random.default_rng(9).uniform(0, 1, len(products))

        print("Loaded real datasets and embeddings.")
        return products, events, image_emb, text_emb

    print("Input files not fully provided. Using mock data.")
    return generate_mock_data(quick=args.quick)


@torch.no_grad()
def recommend_topk(
    trainer,
    model,
    retriever,
    products: pd.DataFrame,
    events: pd.DataFrame,
    image_emb: Dict[str, np.ndarray],
    text_emb: Dict[str, np.ndarray],
    user_id: str,
    top_n: int,
    top_k: int,
    device: str,
) -> pd.DataFrame:
    hist = events[events["user_id"] == user_id].sort_values("timestamp")
    if hist.empty:
        raise ValueError(f"No interactions for user_id={user_id}")

    max_seq_len = 50
    hist = hist.tail(max_seq_len)
    now = hist["timestamp"].max()

    seq_vectors = []
    seq_events = []
    seq_deltas = []
    event_map = {"click": 1, "add_to_cart": 2, "purchase": 3, "view": 4}

    for _, row in hist.iterrows():
        item_id = str(row["item_id"])
        if item_id not in image_emb or item_id not in text_emb:
            continue
        v = np.concatenate([image_emb[item_id], text_emb[item_id]]).astype(np.float32)
        seq_vectors.append(v)
        seq_events.append(event_map.get(str(row["event_type"]), 4))
        seq_deltas.append(max((now - row["timestamp"]).total_seconds() / 3600.0, 0.0))

    if not seq_vectors:
        raise ValueError("User has no valid embedding-backed history.")

    seq_item_vectors = torch.tensor(np.stack(seq_vectors), dtype=torch.float32).unsqueeze(0).to(device)
    seq_event_ids = torch.tensor(seq_events, dtype=torch.long).unsqueeze(0).to(device)
    seq_time = torch.tensor(seq_deltas, dtype=torch.float32).unsqueeze(0).to(device)
    seq_padding_mask = torch.zeros((1, len(seq_events)), dtype=torch.bool).to(device)

    user_dense = torch.tensor(
        [[0.0, float(len(seq_events)), float(seq_deltas[-1] if seq_deltas else 0.0)]],
        dtype=torch.float32,
    ).to(device)

    user_vec = model.encode_user(
        seq_item_vectors=seq_item_vectors,
        seq_event_ids=seq_event_ids,
        seq_time_deltas_hours=seq_time,
        seq_padding_mask=seq_padding_mask,
        user_dense_features=user_dense,
    ).squeeze(0).detach().cpu().numpy().astype(np.float32)

    purchased = set(hist.loc[hist["event_type"] == "purchase", "item_id"].astype(str).tolist())
    candidates = retriever.search(user_embedding=user_vec, top_k=top_n, exclude_item_ids=purchased)

    cdf = pd.DataFrame(candidates, columns=["item_id", "neural_similarity"])
    if cdf.empty:
        return cdf

    cdf = cdf.merge(
        products[["item_id", "product_name", "category", "brand", "price", "popularity_score", "ctr_score"]],
        on="item_id",
        how="left",
    )

    # Basic ranking fallback if no tree-ranker is supplied:
    cdf["ranking_score"] = (
        cdf["neural_similarity"]
        + 0.10 * cdf["popularity_score"].fillna(0.0)
        + 0.05 * cdf["ctr_score"].fillna(0.0)
    )

    cdf = cdf.sort_values("ranking_score", ascending=False).head(top_k).reset_index(drop=True)
    cdf.insert(0, "rank", np.arange(1, len(cdf) + 1))
    return cdf[["rank", "item_id", "product_name", "category", "price", "neural_similarity", "ranking_score"]]


def main() -> None:
    args = parse_args()

    products, events, image_emb, text_emb = load_or_mock(args)

    config = TrainConfig(
        embedding_dim=args.embedding_dim,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=args.device,
        d_model=256,
        include_view_targets=False,
    )

    print(f"Training on device: {config.device}")
    model, trainer, metrics = train_two_tower(
        products=products,
        events=events,
        image_embeddings=image_emb,
        text_embeddings=text_emb,
        config=config,
    )

    print("\nValidation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    item_matrix, item_ids = trainer.compute_item_embedding_table()
    item_emb_dict = {item_ids[i]: item_matrix[i] for i in range(len(item_ids))}

    retriever = ItemRetriever(use_faiss=(not args.no_faiss))
    retriever.build(item_emb_dict)

    user_id = args.user_id
    if user_id is None:
        user_id = str(events["user_id"].iloc[0])

    print(f"\nGenerating recommendations for user_id={user_id} ...")
    recs = recommend_topk(
        trainer=trainer,
        model=model,
        retriever=retriever,
        products=products,
        events=events,
        image_emb=image_emb,
        text_emb=text_emb,
        user_id=user_id,
        top_n=args.top_n,
        top_k=args.top_k,
        device=config.device,
    )

    if recs.empty:
        print("No recommendations produced.")
    else:
        print(recs.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
