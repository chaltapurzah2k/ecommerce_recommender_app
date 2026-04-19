"""
feature_engineering.py — Stage 2: Build ranking features for (user, candidate_item) pairs.

Feature groups:
- USER features     : preferred category/brand, avg spend, recency, activity count
- ITEM features     : category, brand, price, popularity_score, ctr_score
- MATCH features    : cosine similarity to cart/click items, same-category flags,
                      brand overlap, price deviation, brand interaction flag
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from utils import get_logger, EVENT_WEIGHTS

logger = get_logger("feature_engineering")


# ---------------------------------------------------------------------------
# Helper: safe cosine similarity
# ---------------------------------------------------------------------------

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _mean_vec(vecs: list) -> np.ndarray | None:
    """Return L2-normalised mean of a list of embedding vectors, or None."""
    if not vecs:
        return None
    m = np.mean(np.stack(vecs), axis=0).astype(np.float32)
    norm = np.linalg.norm(m)
    return m / norm if norm > 0 else m


# ---------------------------------------------------------------------------
# User-level aggregates
# ---------------------------------------------------------------------------

def compute_user_profiles(
    user_events: pd.DataFrame,
    products: pd.DataFrame,
    recency_days: int = 90,
) -> pd.DataFrame:
    """
    Compute per-user aggregate features.

    Returns a DataFrame indexed by user_id with columns:
    preferred_category, preferred_brand, avg_spend,
    recent_activity_count, days_since_last_interaction
    """
    user_events = user_events.copy()
    user_events["timestamp"] = pd.to_datetime(user_events["timestamp"])

    # Merge price / category / brand from products
    prod_cols = products[["item_id", "category", "brand", "price"]].copy()
    merged = user_events.merge(prod_cols, on="item_id", how="left")

    # Weight events
    merged["weight"] = merged["event_type"].map(
        {k: v + 1 for k, v in EVENT_WEIGHTS.items()}  # shift so view=1
    ).fillna(1)

    cutoff = merged["timestamp"].max() - pd.Timedelta(days=recency_days)
    recent = merged[merged["timestamp"] >= cutoff]

    def _mode_weighted(group_df, col):
        """Weighted mode for a column."""
        if group_df.empty:
            return None
        wdf = group_df.dropna(subset=[col]).groupby(col)["weight"].sum()
        return wdf.idxmax() if not wdf.empty else None

    profiles = []
    now = merged["timestamp"].max()

    for user_id, grp in merged.groupby("user_id"):
        rec_grp = recent[recent["user_id"] == user_id] if not recent.empty else pd.DataFrame()

        pref_cat = _mode_weighted(grp, "category")
        pref_brand = _mode_weighted(grp, "brand")
        avg_spend = grp.loc[grp["event_type"] == "purchase", "price"].mean()
        avg_spend = float(avg_spend) if not np.isnan(avg_spend) else float(grp["price"].mean())
        activity_count = len(rec_grp)
        last_ts = grp["timestamp"].max()
        days_since = (now - last_ts).days if pd.notna(last_ts) else 999

        profiles.append({
            "user_id": user_id,
            "preferred_category": pref_cat,
            "preferred_brand": pref_brand,
            "avg_spend": round(avg_spend, 2),
            "recent_activity_count": activity_count,
            "days_since_last_interaction": days_since,
        })

    return pd.DataFrame(profiles).set_index("user_id")


# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------

class FeatureEngineer:
    """
    Builds a feature matrix for (user_id, item_id) pairs.

    Parameters
    ----------
    products : pd.DataFrame
    user_events : pd.DataFrame
    embeddings : dict[item_id -> np.ndarray]
    recency_days : int
    """

    def __init__(
        self,
        products: pd.DataFrame,
        user_events: pd.DataFrame,
        embeddings: dict,
        recency_days: int = 90,
    ):
        self.products = products.set_index("item_id").copy()
        self.user_events = user_events.copy()
        self.user_events["timestamp"] = pd.to_datetime(self.user_events["timestamp"])
        self.embeddings = embeddings
        self.recency_days = recency_days

        # Encode categorical columns
        self._cat_enc = LabelEncoder()
        self._brand_enc = LabelEncoder()
        all_cats = self.products["category"].fillna("unknown").tolist()
        all_brands = self.products["brand"].fillna("unknown").tolist()
        self._cat_enc.fit(all_cats + ["unknown"])
        self._brand_enc.fit(all_brands + ["unknown"])

        # Precompute global item popularity rank (for trending boost)
        self.products["popularity_rank"] = (
            self.products["popularity_score"]
            .rank(pct=True, ascending=True)
            .fillna(0)
        )

        # Precompute user profiles
        logger.info("Computing user profiles…")
        self.user_profiles = compute_user_profiles(
            user_events, products.reset_index() if "item_id" not in products.columns else products,
            recency_days=recency_days,
        )

        logger.info("FeatureEngineer ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_features(
        self,
        user_id: str,
        candidates_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build a feature row for every candidate item for a given user.

        Parameters
        ----------
        candidates_df : pd.DataFrame
            Must contain at least: item_id, similarity_score.

        Returns
        -------
        pd.DataFrame with one row per candidate and all ranking features.
        """
        user_emb_context = self._get_user_embedding_context(user_id)
        user_profile = self._get_user_profile(user_id)
        user_brand_history = self._get_user_brand_history(user_id)

        rows = []
        for _, cand in candidates_df.iterrows():
            item_id = cand["item_id"]
            row = self._build_row(
                user_id=user_id,
                item_id=item_id,
                similarity_score=cand["similarity_score"],
                user_profile=user_profile,
                user_emb_context=user_emb_context,
                user_brand_history=user_brand_history,
            )
            rows.append(row)

        feat_df = pd.DataFrame(rows)
        return feat_df

    def build_training_dataset(
        self,
        user_ids: list | None = None,
        top_n_candidates: int = 100,
        candidate_generator=None,
    ) -> pd.DataFrame:
        """
        Build training dataset with labels derived from historical events.

        For each user: generate candidate pool (if candidate_generator provided,
        otherwise use items from user history + random negatives), then
        build features and assign labels.
        """
        from sklearn.utils import shuffle as sk_shuffle

        all_users = self.user_events["user_id"].unique().tolist()
        if user_ids is not None:
            all_users = [u for u in all_users if u in set(user_ids)]

        all_rows = []
        for user_id in all_users:
            if candidate_generator is not None:
                cands = candidate_generator.get_candidates(user_id)
            else:
                cands = self._simple_candidates(user_id, top_n_candidates)

            if cands.empty:
                continue

            feat_df = self.build_features(user_id, cands)
            labels = self._assign_labels(user_id, cands["item_id"].tolist())
            feat_df["label"] = labels
            feat_df["user_id"] = user_id
            all_rows.append(feat_df)

        if not all_rows:
            raise ValueError("No training rows generated — check user_events data.")

        full_df = pd.concat(all_rows, ignore_index=True)
        logger.info("Training dataset: %d rows across %d users.", len(full_df), len(all_users))
        return full_df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_user_profile(self, user_id: str) -> dict:
        if user_id in self.user_profiles.index:
            return self.user_profiles.loc[user_id].to_dict()
        return {
            "preferred_category": "unknown",
            "preferred_brand": "unknown",
            "avg_spend": self.products["price"].mean(),
            "recent_activity_count": 0,
            "days_since_last_interaction": 999,
        }

    def _get_user_brand_history(self, user_id: str) -> set:
        """Brands the user has interacted with (any event)."""
        hist = self.user_events[self.user_events["user_id"] == user_id]
        if hist.empty:
            return set()
        items = hist["item_id"].unique()
        brands = self.products.loc[
            self.products.index.isin(items), "brand"
        ].dropna().tolist()
        return set(brands)

    def _get_user_embedding_context(self, user_id: str) -> dict:
        """
        Returns dict with:
          'cart_vec'  : mean embedding of cart items
          'click_vec' : mean embedding of clicked items
        """
        cutoff = (
            self.user_events["timestamp"].max()
            - pd.Timedelta(days=self.recency_days)
        )
        hist = self.user_events[
            (self.user_events["user_id"] == user_id)
            & (self.user_events["timestamp"] >= cutoff)
        ]

        def _vecs_for_events(event_types):
            items = hist.loc[hist["event_type"].isin(event_types), "item_id"]
            return [self.embeddings[i] for i in items if i in self.embeddings]

        return {
            "cart_vec": _mean_vec(_vecs_for_events(["add_to_cart", "purchase"])),
            "click_vec": _mean_vec(_vecs_for_events(["click"])),
        }

    def _build_row(
        self,
        user_id: str,
        item_id: str,
        similarity_score: float,
        user_profile: dict,
        user_emb_context: dict,
        user_brand_history: set,
    ) -> dict:
        # Item features
        try:
            item = self.products.loc[item_id]
            item_category = str(item.get("category", "unknown"))
            item_brand = str(item.get("brand", "unknown"))
            item_price = float(item.get("price", 0.0))
            item_pop = float(item.get("popularity_score", 0.0))
            item_ctr = float(item.get("ctr_score", 0.0))
            item_pop_rank = float(item.get("popularity_rank", 0.0))
        except KeyError:
            item_category, item_brand = "unknown", "unknown"
            item_price, item_pop, item_ctr, item_pop_rank = 0.0, 0.0, 0.0, 0.0

        # User features
        pref_cat = str(user_profile.get("preferred_category", "unknown"))
        pref_brand = str(user_profile.get("preferred_brand", "unknown"))
        avg_spend = float(user_profile.get("avg_spend", item_price))

        # Match features (embedding-level)
        item_vec = self.embeddings.get(item_id)
        sim_to_cart = 0.0
        sim_to_click = 0.0
        if item_vec is not None:
            cart_vec = user_emb_context.get("cart_vec")
            click_vec = user_emb_context.get("click_vec")
            if cart_vec is not None:
                sim_to_cart = _cosine(item_vec, cart_vec)
            if click_vec is not None:
                sim_to_click = _cosine(item_vec, click_vec)

        row = {
            # Identity (not used as features directly)
            "item_id": item_id,

            # ----- USER features -----
            "user_preferred_category_enc": self._safe_enc(self._cat_enc, pref_cat),
            "user_preferred_brand_enc": self._safe_enc(self._brand_enc, pref_brand),
            "user_avg_spend": avg_spend,
            "user_recent_activity_count": int(user_profile.get("recent_activity_count", 0)),
            "user_days_since_last_interaction": int(user_profile.get("days_since_last_interaction", 999)),

            # ----- ITEM features -----
            "item_category_enc": self._safe_enc(self._cat_enc, item_category),
            "item_brand_enc": self._safe_enc(self._brand_enc, item_brand),
            "item_price": item_price,
            "item_popularity_score": item_pop,
            "item_ctr_score": item_ctr,
            "item_popularity_rank": item_pop_rank,

            # ----- MATCH features -----
            "sim_to_cart": sim_to_cart,
            "sim_to_click": sim_to_click,
            "retrieval_similarity": float(similarity_score),
            "same_category_as_pref": int(item_category == pref_cat),
            "same_brand_as_pref": int(item_brand == pref_brand),
            "price_diff_from_avg_spend": abs(item_price - avg_spend),
            "price_ratio_to_avg": item_price / avg_spend if avg_spend > 0 else 1.0,
            "user_interacted_brand": int(item_brand in user_brand_history),
        }
        return row

    def _assign_labels(self, user_id: str, item_ids: list) -> list:
        """Assign relevance labels based on highest event type per item."""
        hist = self.user_events[self.user_events["user_id"] == user_id]
        max_label = (
            hist.groupby("item_id")["event_type"]
            .apply(lambda es: max(EVENT_WEIGHTS.get(e, 0) for e in es))
        ).to_dict()
        return [max_label.get(i, 0) for i in item_ids]

    def _simple_candidates(self, user_id: str, top_n: int) -> pd.DataFrame:
        """
        Fallback candidate set when no CandidateGenerator is passed:
        union of user's interacted items + random sample of other items.
        """
        hist_items = set(
            self.user_events.loc[self.user_events["user_id"] == user_id, "item_id"]
        )
        all_ids = list(self.products.index)
        neg_pool = [i for i in all_ids if i not in hist_items]
        rng = np.random.default_rng(42)
        n_neg = max(0, top_n - len(hist_items))
        negatives = rng.choice(neg_pool, size=min(n_neg, len(neg_pool)), replace=False).tolist()
        candidates = list(hist_items) + negatives
        return pd.DataFrame({
            "item_id": candidates,
            "similarity_score": [0.5] * len(candidates),
        })

    @staticmethod
    def _safe_enc(encoder: LabelEncoder, value: str) -> int:
        try:
            return int(encoder.transform([value])[0])
        except ValueError:
            return -1


# ---------------------------------------------------------------------------
# Feature column names (used by ranker)
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "user_preferred_category_enc",
    "user_preferred_brand_enc",
    "user_avg_spend",
    "user_recent_activity_count",
    "user_days_since_last_interaction",
    "item_category_enc",
    "item_brand_enc",
    "item_price",
    "item_popularity_score",
    "item_ctr_score",
    "item_popularity_rank",
    "sim_to_cart",
    "sim_to_click",
    "retrieval_similarity",
    "same_category_as_pref",
    "same_brand_as_pref",
    "price_diff_from_avg_spend",
    "price_ratio_to_avg",
    "user_interacted_brand",
]
