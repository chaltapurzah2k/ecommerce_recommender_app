"""FAISS-based retrieval for Two-Tower embeddings with numpy fallback."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False


class ItemRetriever:
    """Retrieves top-K item IDs by inner-product similarity."""

    def __init__(self, use_faiss: bool = True):
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        self.index = None
        self.item_ids: List[str] = []
        self.item_matrix: Optional[np.ndarray] = None

    def build(self, item_embeddings: Dict[str, np.ndarray]) -> None:
        self.item_ids = list(item_embeddings.keys())
        matrix = np.stack([item_embeddings[i] for i in self.item_ids]).astype(np.float32)

        # Two-tower outputs are expected to be L2-normalized.
        self.item_matrix = matrix

        if self.use_faiss:
            dim = matrix.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(matrix)

    def search(
        self,
        user_embedding: np.ndarray,
        top_k: int = 200,
        exclude_item_ids: Optional[Iterable[str]] = None,
    ) -> List[Tuple[str, float]]:
        if self.item_matrix is None:
            raise RuntimeError("Retriever index is not built yet.")

        exclude = set(exclude_item_ids) if exclude_item_ids is not None else set()
        query = user_embedding.astype(np.float32).reshape(1, -1)
        fetch_k = min(max(top_k * 3, top_k), len(self.item_ids))

        if self.use_faiss and self.index is not None:
            scores, indices = self.index.search(query, fetch_k)
            scored = [
                (self.item_ids[idx], float(score))
                for idx, score in zip(indices[0], scores[0])
                if idx >= 0
            ]
        else:
            sims = (self.item_matrix @ query.reshape(-1)).astype(np.float32)
            idx = np.argpartition(sims, -fetch_k)[-fetch_k:]
            idx = idx[np.argsort(sims[idx])[::-1]]
            scored = [(self.item_ids[i], float(sims[i])) for i in idx]

        out: List[Tuple[str, float]] = []
        for item_id, score in scored:
            if item_id in exclude:
                continue
            out.append((item_id, score))
            if len(out) >= top_k:
                break
        return out
