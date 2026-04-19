"""
candidate_generator.py — Stage 1: Retrieve top-N candidate items per user.

Strategy:
1. Collect recent interacted items for the user (cart > click > view, within window).
2. Build a query embedding = weighted mean of interacted item embeddings.
3. Use FAISS (if available) or cosine similarity (numpy fallback) to retrieve top-N.
4. Remove already-purchased items and deduplicate.
5. Optionally filter out-of-stock items.
6. Cold-start fallback: return popular items when there is no interaction history.
"""

import numpy as np
import pandas as pd
from typing import Optional
from utils import get_logger, EVENT_WEIGHTS

logger = get_logger("candidate_generator")

# Try to import FAISS; fall back to exact cosine similarity if unavailable.
try:
    import faiss
    _FAISS_AVAILABLE = True
    logger.info("FAISS is available — using ANN index for retrieval.")
except ImportError:
    _FAISS_AVAILABLE = False
    logger.warning("FAISS not found — using numpy cosine similarity (slower).")


# ---------------------------------------------------------------------------
# Interaction weight used when building query vector
# ---------------------------------------------------------------------------

INTERACTION_WEIGHT = {
    "purchase": 3.0,
    "add_to_cart": 2.0,
    "click": 1.0,
    "view": 0.5,
}


# ---------------------------------------------------------------------------
# FAISS index helpers
# ---------------------------------------------------------------------------

def build_faiss_index(embeddings: dict) -> tuple:
    """
    Build an inner-product FAISS index from embeddings dict.
    Vectors must already be L2-normalised (inner product == cosine similarity).

    Returns (index, ordered_item_ids).
    """
    item_ids = list(embeddings.keys())
    matrix = np.stack([embeddings[i] for i in item_ids]).astype(np.float32)
    dim = matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)
    return index, item_ids


# ---------------------------------------------------------------------------
# CandidateGenerator
# ---------------------------------------------------------------------------

class CandidateGenerator:
    """
    Generates top-N candidate item_ids for a given user.

    Parameters
    ----------
    embeddings : dict[item_id -> np.ndarray]
        Combined (or single-modal) item embeddings.
    products : pd.DataFrame
        Must contain columns: item_id, in_stock, popularity_score.
    user_events : pd.DataFrame
        Must contain columns: user_id, item_id, event_type, timestamp.
    top_n : int
        Number of candidates to retrieve per user.
    recency_days : int
        Only consider interactions within this many days for the query vector.
    """

    def __init__(
        self,
        embeddings: dict,
        products: pd.DataFrame,
        user_events: pd.DataFrame,
        top_n: int = 100,
        recency_days: int = 90,
    ):
        self.embeddings = embeddings
        self.products = products.set_index("item_id") if "item_id" in products.columns else products
        self.user_events = user_events.copy()
        self.user_events["timestamp"] = pd.to_datetime(self.user_events["timestamp"])
        self.top_n = top_n
        self.recency_days = recency_days

        # Filter embeddings to items that exist in products index
        valid_ids = set(self.products.index) & set(embeddings.keys())
        self.embeddings = {k: v for k, v in embeddings.items() if k in valid_ids}
        self.item_ids = list(self.embeddings.keys())
        self.item_matrix = np.stack(
            [self.embeddings[i] for i in self.item_ids]
        ).astype(np.float32)  # shape: (N_items, dim)

        # Build FAISS index if available
        if _FAISS_AVAILABLE:
            self._faiss_index, _ = build_faiss_index(self.embeddings)

        # Popularity fallback (sorted by popularity_score desc)
        self._popular_items = (
            self.products["popularity_score"]
            .sort_values(ascending=False)
            .index.tolist()
        )

        logger.info(
            "CandidateGenerator ready: %d items, top_n=%d, recency=%d days",
            len(self.item_ids), top_n, recency_days,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_candidates(
        self,
        user_id: str,
        filter_out_of_stock: bool = True,
        exclude_purchased: bool = True,
    ) -> pd.DataFrame:
        """
        Return a DataFrame of candidate items for a user.

        Columns: item_id, similarity_score, is_cold_start
        """
        cutoff = self.user_events["timestamp"].max() - pd.Timedelta(days=self.recency_days)
        user_hist = self.user_events[
            (self.user_events["user_id"] == user_id)
            & (self.user_events["timestamp"] >= cutoff)
        ]

        purchased = set()
        if exclude_purchased:
            purchased = set(
                user_hist.loc[user_hist["event_type"] == "purchase", "item_id"]
            )

        # Cold-start: no interaction history
        if user_hist.empty:
            logger.info("Cold-start for user %s — returning popular items.", user_id)
            return self._cold_start_candidates(
                exclude=purchased,
                filter_out_of_stock=filter_out_of_stock,
            )

        query_vec = self._build_query_vector(user_hist)
        if query_vec is None:
            return self._cold_start_candidates(
                exclude=purchased,
                filter_out_of_stock=filter_out_of_stock,
            )

        candidates_df = self._retrieve(
            query_vec,
            exclude=purchased,
            filter_out_of_stock=filter_out_of_stock,
        )
        candidates_df["is_cold_start"] = False
        return candidates_df

    def get_user_history(self, user_id: str) -> pd.DataFrame:
        """Return all events for a user, including event_type."""
        return self.user_events[self.user_events["user_id"] == user_id].copy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_query_vector(self, user_hist: pd.DataFrame) -> Optional[np.ndarray]:
        """Weighted mean of interacted item embeddings."""
        vecs, weights = [], []
        for _, row in user_hist.iterrows():
            item_id = row["item_id"]
            if item_id not in self.embeddings:
                continue
            w = INTERACTION_WEIGHT.get(row["event_type"], 0.5)
            vecs.append(self.embeddings[item_id])
            weights.append(w)

        if not vecs:
            return None

        weights = np.array(weights, dtype=np.float32)
        matrix = np.stack(vecs).astype(np.float32)
        query = (weights[:, None] * matrix).sum(axis=0)
        norm = np.linalg.norm(query)
        return (query / norm).astype(np.float32) if norm > 0 else query

    def _retrieve(
        self,
        query_vec: np.ndarray,
        exclude: set,
        filter_out_of_stock: bool,
    ) -> pd.DataFrame:
        """Retrieve top candidates using FAISS or numpy fallback."""
        # Retrieve more than needed to account for post-filtering
        fetch_n = min(self.top_n * 3, len(self.item_ids))

        if _FAISS_AVAILABLE:
            scores, indices = self._faiss_index.search(
                query_vec[None, :], fetch_n
            )
            scores = scores[0]
            indices = indices[0]
            candidates = [
                (self.item_ids[idx], float(score))
                for idx, score in zip(indices, scores)
                if idx >= 0
            ]
        else:
            sims = self.item_matrix @ query_vec  # cosine similarity (vectors are L2-normed)
            top_idx = np.argpartition(sims, -fetch_n)[-fetch_n:]
            top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
            candidates = [(self.item_ids[i], float(sims[i])) for i in top_idx]

        rows = []
        for item_id, score in candidates:
            if item_id in exclude:
                continue
            if filter_out_of_stock:
                try:
                    if not self.products.loc[item_id, "in_stock"]:
                        continue
                except KeyError:
                    pass
            rows.append({"item_id": item_id, "similarity_score": score})
            if len(rows) >= self.top_n:
                break

        return pd.DataFrame(rows)

    def _cold_start_candidates(
        self,
        exclude: set,
        filter_out_of_stock: bool,
    ) -> pd.DataFrame:
        rows = []
        for item_id in self._popular_items:
            if item_id in exclude:
                continue
            if filter_out_of_stock:
                try:
                    if not self.products.loc[item_id, "in_stock"]:
                        continue
                except KeyError:
                    pass
            rows.append({"item_id": item_id, "similarity_score": 0.0})
            if len(rows) >= self.top_n:
                break
        df = pd.DataFrame(rows)
        df["is_cold_start"] = True
        return df
