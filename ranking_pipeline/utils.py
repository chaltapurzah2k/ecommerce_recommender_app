"""
utils.py — Shared utilities for the recommendation ranking pipeline.

Handles:
- Mock data generation (when real files are unavailable)
- Embedding loading / saving
- Metric computation (Precision@K, Recall@K, MAP@K, NDCG@K, HR@K)
- Logging setup
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import normalize

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str = "ranker") -> logging.Logger:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=logging.INFO,
    )
    return logging.getLogger(name)


logger = get_logger("utils")


# ---------------------------------------------------------------------------
# Event weights
# ---------------------------------------------------------------------------

EVENT_WEIGHTS = {
    "purchase": 3,
    "add_to_cart": 2,
    "click": 1,
    "view": 0,
}

LABEL_MAP = EVENT_WEIGHTS  # alias


# ---------------------------------------------------------------------------
# Mock data generation
# ---------------------------------------------------------------------------

CATEGORIES = ["T-Shirts", "Jeans", "Dresses", "Jackets", "Shoes", "Accessories", "Shorts", "Sweaters"]
BRANDS = ["Zara", "H&M", "Nike", "Adidas", "Levi's", "Gucci", "Puma", "Uniqlo"]


def generate_mock_products(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic products dataframe."""
    rng = np.random.default_rng(seed)
    item_ids = [f"ITEM_{i:04d}" for i in range(n)]
    records = []
    for item_id in item_ids:
        records.append({
            "item_id": item_id,
            "product_name": f"{rng.choice(BRANDS)} {rng.choice(CATEGORIES)} {item_id[-4:]}",
            "category": rng.choice(CATEGORIES),
            "brand": rng.choice(BRANDS),
            "price": round(float(rng.uniform(10, 500)), 2),
            "image_url": f"https://placeholder.com/{item_id}.jpg",
            "description": f"A stylish {rng.choice(CATEGORIES).lower()} from {rng.choice(BRANDS)}.",
            "in_stock": rng.choice([True, True, True, False]),  # ~75% in stock
            "popularity_score": round(float(rng.uniform(0, 1)), 4),
            "ctr_score": round(float(rng.uniform(0, 1)), 4),
        })
    return pd.DataFrame(records)


def generate_mock_user_events(
    products: pd.DataFrame,
    n_users: int = 200,
    events_per_user: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic user interaction events."""
    rng = np.random.default_rng(seed)
    item_ids = products["item_id"].tolist()
    event_types = list(EVENT_WEIGHTS.keys())
    event_probs = [0.05, 0.15, 0.30, 0.50]  # purchase, cart, click, view

    records = []
    base_time = datetime(2025, 1, 1)
    for u in range(n_users):
        user_id = f"USER_{u:04d}"
        n_events = int(rng.integers(5, events_per_user + 1))
        for _ in range(n_events):
            records.append({
                "user_id": user_id,
                "item_id": rng.choice(item_ids),
                "event_type": rng.choice(event_types, p=event_probs),
                "timestamp": base_time + timedelta(
                    days=int(rng.integers(0, 365)),
                    hours=int(rng.integers(0, 24)),
                ),
            })
    df = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
    return df


def generate_mock_embeddings(
    item_ids: list,
    dim: int = 128,
    seed: int = 42,
) -> dict:
    """Return a dict mapping item_id -> L2-normalised embedding vector."""
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((len(item_ids), dim)).astype(np.float32)
    normed = normalize(raw, norm="l2")
    return {item_id: normed[i] for i, item_id in enumerate(item_ids)}


# ---------------------------------------------------------------------------
# Embedding I/O
# ---------------------------------------------------------------------------

def save_embeddings(embeddings: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(embeddings, f)
    logger.info("Saved embeddings to %s", path)


def load_embeddings(path: str) -> dict:
    with open(path, "rb") as f:
        embeddings = pickle.load(f)
    logger.info("Loaded %d embeddings from %s", len(embeddings), path)
    return embeddings


def combine_embeddings(
    image_emb: dict,
    text_emb: dict,
    image_weight: float = 0.5,
    text_weight: float = 0.5,
) -> dict:
    """
    Combine image and text embeddings via weighted sum.
    Only items present in both dicts are included.
    """
    combined = {}
    common_ids = set(image_emb) & set(text_emb)
    for item_id in common_ids:
        vec = image_weight * image_emb[item_id] + text_weight * text_emb[item_id]
        norm = np.linalg.norm(vec)
        combined[item_id] = vec / norm if norm > 0 else vec
    logger.info("Combined embeddings for %d items", len(combined))
    return combined


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------

def _dcg(relevances: list) -> float:
    return sum(
        (2**rel - 1) / np.log2(rank + 2)
        for rank, rel in enumerate(relevances)
    )


def ndcg_at_k(recommended: list, relevant_scores: dict, k: int = 10) -> float:
    """
    recommended: ordered list of item_ids (predicted top-k)
    relevant_scores: dict[item_id -> relevance_score]
    """
    rec_k = recommended[:k]
    dcg = _dcg([relevant_scores.get(i, 0) for i in rec_k])
    ideal = _dcg(sorted(relevant_scores.values(), reverse=True)[:k])
    return dcg / ideal if ideal > 0 else 0.0


def precision_at_k(recommended: list, relevant_set: set, k: int = 10) -> float:
    hits = sum(1 for i in recommended[:k] if i in relevant_set)
    return hits / k


def recall_at_k(recommended: list, relevant_set: set, k: int = 10) -> float:
    hits = sum(1 for i in recommended[:k] if i in relevant_set)
    return hits / len(relevant_set) if relevant_set else 0.0


def average_precision_at_k(recommended: list, relevant_set: set, k: int = 10) -> float:
    hits, score = 0, 0.0
    for rank, item in enumerate(recommended[:k], start=1):
        if item in relevant_set:
            hits += 1
            score += hits / rank
    return score / min(len(relevant_set), k) if relevant_set else 0.0


def hit_rate_at_k(recommended: list, relevant_set: set, k: int = 10) -> bool:
    return any(i in relevant_set for i in recommended[:k])


def evaluate_rankings(
    results: list[dict],
    k: int = 10,
) -> dict:
    """
    results: list of dicts, each with keys:
        'recommended' (list of item_ids)
        'relevant_scores' (dict item_id -> label)
        'relevant_set' (set of item_ids with label > 0)
    Returns averaged metrics.
    """
    metrics = {"precision": [], "recall": [], "map": [], "ndcg": [], "hit_rate": []}
    for r in results:
        rec = r["recommended"]
        rel_set = r["relevant_set"]
        rel_scores = r["relevant_scores"]
        metrics["precision"].append(precision_at_k(rec, rel_set, k))
        metrics["recall"].append(recall_at_k(rec, rel_set, k))
        metrics["map"].append(average_precision_at_k(rec, rel_set, k))
        metrics["ndcg"].append(ndcg_at_k(rec, rel_scores, k))
        metrics["hit_rate"].append(float(hit_rate_at_k(rec, rel_set, k)))

    return {f"{m}@{k}": round(np.mean(v), 4) for m, v in metrics.items()}
