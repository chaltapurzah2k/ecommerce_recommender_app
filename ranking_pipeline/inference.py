"""
inference.py — Stage 4: Online inference pipeline.

Given a user_id:
1. Generate top-N candidates via CandidateGenerator.
2. Build ranking features via FeatureEngineer.
3. Score candidates with a trained model.
4. Apply post-ranking logic:
   - Diversity boost (max K items per category).
   - Trending product boost.
5. Return top-10 recommendations as a formatted DataFrame.
"""

import numpy as np
import pandas as pd
from utils import get_logger
from feature_engineering import FEATURE_COLUMNS

logger = get_logger("inference")


# ---------------------------------------------------------------------------
# RecommendationPipeline
# ---------------------------------------------------------------------------

class RecommendationPipeline:
    """
    End-to-end inference pipeline combining candidate generation,
    feature engineering, and model scoring.

    Parameters
    ----------
    candidate_generator : CandidateGenerator
    feature_engineer    : FeatureEngineer
    model               : trained LightGBMRanker or XGBoostRanker
    products            : pd.DataFrame  (item_id, product_name, category, brand, price, …)
    top_k               : int           (number of final recommendations)
    diversity_max_per_category : int | None
        If set, limits the number of items from any single category.
    trending_boost      : float
        Multiplicative score boost applied to items in the top-20% by popularity.
    """

    def __init__(
        self,
        candidate_generator,
        feature_engineer,
        model,
        products: pd.DataFrame,
        top_k: int = 10,
        diversity_max_per_category: int | None = 3,
        trending_boost: float = 1.10,
    ):
        self.cg = candidate_generator
        self.fe = feature_engineer
        self.model = model
        self.products = products.set_index("item_id").copy()
        self.top_k = top_k
        self.diversity_max_per_category = diversity_max_per_category
        self.trending_boost = trending_boost

        # Trending threshold: top-20% by popularity_score
        pop_col = "popularity_score"
        if pop_col in self.products.columns:
            self._trending_threshold = float(
                self.products[pop_col].quantile(0.80)
            )
        else:
            self._trending_threshold = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recommend(
        self,
        user_id: str,
        filter_out_of_stock: bool = True,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Return a ranked DataFrame of top-K recommendations for a user.

        Columns:
            rank, item_id, product_name, category, brand,
            price, similarity_score, ranking_score
        """
        # --- Stage 1: Candidate generation ---
        candidates = self.cg.get_candidates(
            user_id,
            filter_out_of_stock=filter_out_of_stock,
        )
        if candidates.empty:
            logger.warning("No candidates for user %s — returning empty.", user_id)
            return pd.DataFrame()

        if verbose:
            logger.info("Candidates retrieved: %d", len(candidates))

        # --- Stage 2: Feature engineering ---
        feat_df = self.fe.build_features(user_id, candidates)

        # --- Stage 3: Model scoring ---
        X = feat_df[FEATURE_COLUMNS].astype(np.float32)
        raw_scores = self.model.predict(X)
        feat_df["raw_score"] = raw_scores
        feat_df["similarity_score"] = candidates["similarity_score"].values

        # --- Stage 4: Post-ranking boosts ---
        feat_df = self._apply_trending_boost(feat_df)

        # --- Stage 5: Diversity & select top-K ---
        ranked = feat_df.sort_values("ranking_score", ascending=False).reset_index(drop=True)
        top_items = self._apply_diversity(ranked)

        # --- Stage 6: Format output ---
        return self._format_output(top_items)

    # ------------------------------------------------------------------
    # Post-ranking helpers
    # ------------------------------------------------------------------

    def _apply_trending_boost(self, feat_df: pd.DataFrame) -> pd.DataFrame:
        """Boost score of trending items."""
        feat_df = feat_df.copy()
        feat_df["ranking_score"] = feat_df["raw_score"].copy()

        if self._trending_threshold is None:
            return feat_df

        for idx, row in feat_df.iterrows():
            item_id = row["item_id"]
            try:
                pop = float(self.products.loc[item_id, "popularity_score"])
                if pop >= self._trending_threshold:
                    feat_df.at[idx, "ranking_score"] *= self.trending_boost
            except KeyError:
                pass

        return feat_df

    def _apply_diversity(self, ranked: pd.DataFrame) -> pd.DataFrame:
        """
        Select top-K items ensuring at most `diversity_max_per_category`
        items from any single category.
        """
        if self.diversity_max_per_category is None:
            return ranked.head(self.top_k)

        selected = []
        cat_counts: dict[str, int] = {}

        for _, row in ranked.iterrows():
            item_id = row["item_id"]
            try:
                category = str(self.products.loc[item_id, "category"])
            except KeyError:
                category = "unknown"

            count = cat_counts.get(category, 0)
            if count < self.diversity_max_per_category:
                selected.append(row)
                cat_counts[category] = count + 1

            if len(selected) >= self.top_k:
                break

        return pd.DataFrame(selected).reset_index(drop=True)

    def _format_output(self, top_items: pd.DataFrame) -> pd.DataFrame:
        records = []
        for rank, (_, row) in enumerate(top_items.iterrows(), start=1):
            item_id = row["item_id"]
            try:
                item = self.products.loc[item_id]
                product_name = item.get("product_name", item_id)
                category = item.get("category", "N/A")
                brand = item.get("brand", "N/A")
                price = float(item.get("price", 0.0))
            except KeyError:
                product_name, category, brand, price = item_id, "N/A", "N/A", 0.0

            records.append({
                "rank": rank,
                "item_id": item_id,
                "product_name": product_name,
                "category": category,
                "brand": brand,
                "price": price,
                "similarity_score": round(float(row.get("similarity_score", 0.0)), 4),
                "ranking_score": round(float(row.get("ranking_score", 0.0)), 4),
            })

        return pd.DataFrame(records)

    def batch_recommend(
        self,
        user_ids: list,
        filter_out_of_stock: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """Return recommendations for multiple users."""
        return {
            uid: self.recommend(uid, filter_out_of_stock=filter_out_of_stock)
            for uid in user_ids
        }


# ---------------------------------------------------------------------------
# Convenience: load pipeline from disk
# ---------------------------------------------------------------------------

def load_pipeline(
    model_path: str,
    candidate_generator,
    feature_engineer,
    products: pd.DataFrame,
    **kwargs,
) -> RecommendationPipeline:
    """Load a serialised model and wrap it in a RecommendationPipeline."""
    import pickle
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    logger.info("Loaded model from %s", model_path)
    return RecommendationPipeline(
        candidate_generator=candidate_generator,
        feature_engineer=feature_engineer,
        model=model,
        products=products,
        **kwargs,
    )
