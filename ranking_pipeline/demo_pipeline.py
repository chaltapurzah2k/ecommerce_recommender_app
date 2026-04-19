# ========================= run_pipeline.py =========================

import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from utils import (
    get_logger,
    generate_mock_products,
    generate_mock_user_events,
    generate_mock_embeddings,
    combine_embeddings,
)

from candidate_generator import CandidateGenerator
from feature_engineering import FeatureEngineer, FEATURE_COLUMNS
from train_ranker import temporal_train_val_split, train, prepare_ranking_arrays
from inference import RecommendationPipeline

logger = get_logger("run_pipeline")


# ==========================================================
# CLI
# ==========================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--model", default="both",
                   choices=["lgbm", "xgboost", "both"])

    p.add_argument("--top_k", default=10, type=int)
    p.add_argument("--top_n_cands", default=100, type=int)
    p.add_argument("--export_csv", action="store_true", help="Export recommendations per user to CSV files")
    p.add_argument("--csv_dir", default="recommendation_exports", help="Directory for exported recommendation CSV files")

    return p.parse_args()


def _safe_user_token(user_id: str) -> str:
    """Convert a user id to a filesystem-safe token for CSV file names."""
    token = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(user_id))
    return token.strip("_") or "unknown_user"


def _attach_cart_items_before_recommendations(
    user_id: str,
    recs: pd.DataFrame,
    user_events: pd.DataFrame,
    products: pd.DataFrame,
) -> pd.DataFrame:
    """Prepend the user's latest add_to_cart items before ranked recommendations."""
    if recs is None or recs.empty:
        return recs

    events = user_events.copy()
    events["timestamp"] = pd.to_datetime(events["timestamp"], errors="coerce")
    cart = events[
        (events["user_id"] == user_id)
        & (events["event_type"] == "add_to_cart")
    ].sort_values("timestamp")

    if cart.empty:
        out = recs.copy()
        out["row_type"] = "recommendation"
        return out

    seen = set()
    cart_item_ids = []
    for item_id in cart["item_id"].astype(str).tolist():
        if item_id not in seen:
            seen.add(item_id)
            cart_item_ids.append(item_id)

    prod = products.copy().set_index("item_id")
    cart_rows = []
    for item_id in cart_item_ids:
        item = prod.loc[item_id] if item_id in prod.index else pd.Series(dtype=object)
        cart_rows.append({
            "row_type": "cart_item",
            "rank": 0,
            "item_id": item_id,
            "product_name": item.get("product_name", item_id),
            "category": item.get("category", "N/A"),
            "brand": item.get("brand", "N/A"),
            "price": float(item.get("price", 0.0)) if pd.notna(item.get("price", np.nan)) else 0.0,
            "similarity_score": np.nan,
            "ranking_score": np.nan,
        })

    rec_out = recs.copy()
    rec_out["row_type"] = "recommendation"
    cols = [
        "row_type", "rank", "item_id", "product_name", "category", "brand",
        "price", "similarity_score", "ranking_score",
    ]
    cart_df = pd.DataFrame(cart_rows, columns=cols)
    rec_out = rec_out.reindex(columns=cols)
    return pd.concat([cart_df, rec_out], ignore_index=True)


# ==========================================================
# MAIN
# ==========================================================

def main():

    args = parse_args()

    # -------------------------------------------------------
    # 1. MOCK DATA
    # -------------------------------------------------------

    logger.info("Generating mock products + user events")

    products = generate_mock_products(n=500)

    user_events = generate_mock_user_events(
        products,
        n_users=200,
        events_per_user=30
    )

    item_ids = products["item_id"].tolist()

    # -------------------------------------------------------
    # 2. EMBEDDINGS
    # -------------------------------------------------------

    logger.info("Generating embeddings")

    image_emb = generate_mock_embeddings(item_ids, dim=128, seed=10)
    text_emb = generate_mock_embeddings(item_ids, dim=128, seed=20)

    embeddings = combine_embeddings(
        image_emb,
        text_emb,
        image_weight=0.5,
        text_weight=0.5
    )

    # -------------------------------------------------------
    # 3. Candidate Generator
    # -------------------------------------------------------

    cg = CandidateGenerator(
        embeddings=embeddings,
        products=products,
        user_events=user_events,
        top_n=args.top_n_cands,
        recency_days=90,
    )

    # -------------------------------------------------------
    # 4. Feature Engineering
    # -------------------------------------------------------

    fe = FeatureEngineer(
        products=products,
        user_events=user_events,
        embeddings=embeddings,
        recency_days=90,
    )

    logger.info("Building training dataset...")

    all_users = user_events["user_id"].unique().tolist()

    full_df = fe.build_training_dataset(
        user_ids=all_users,
        top_n_candidates=args.top_n_cands,
        candidate_generator=cg,
    )

    # -------------------------------------------------------
    # 5. Train / Validation Split
    # -------------------------------------------------------

    train_df, val_df = temporal_train_val_split(
        full_df,
        user_events,
        val_fraction=0.2
    )

    # -------------------------------------------------------
    # 6. TRAIN MODELS
    # -------------------------------------------------------

    trained_models = train(
        train_df=train_df,
        val_df=val_df,
        model_type=args.model,
        output_dir="models"
    )

    if not trained_models:
        print("No models trained.")
        return

    # -------------------------------------------------------
    # 7. EXPLAINABILITY
    # -------------------------------------------------------

    print("\n" + "=" * 70)
    print("MODEL EXPLAINABILITY")
    print("=" * 70)

    X_val, _, _ = prepare_ranking_arrays(val_df)

    # ---------------- LightGBM ----------------

    if "lgbm" in trained_models:

        model = trained_models["lgbm"]

        print("\nTOP LIGHTGBM FEATURES")
        print(model.feature_importances(FEATURE_COLUMNS).head(15))

        print("\nLEARNED LIGHTGBM TREE RULES")
        model.print_rules(tree_index=0)

    # ---------------- XGBoost ----------------

    if "xgboost" in trained_models:

        model = trained_models["xgboost"]

        print("\nTOP XGBOOST FEATURES")
        print(model.feature_importances(FEATURE_COLUMNS).head(15))

        print("\nLEARNED XGBOOST TREE RULES")
        model.print_rules()

    # -------------------------------------------------------
    # 8. INFERENCE DEMO
    # -------------------------------------------------------

    model = trained_models.get("lgbm") or trained_models.get("xgboost")

    pipeline = RecommendationPipeline(
        candidate_generator=cg,
        feature_engineer=fe,
        model=model,
        products=products,
        top_k=args.top_k,
        diversity_max_per_category=3,
        trending_boost=1.10,
    )

    users = user_events["user_id"].unique()[:5]

    print("\n" + "=" * 70)
    print("INFERENCE DEMO")
    print("=" * 70)

    export_dir = None
    if args.export_csv:
        export_dir = os.path.join(os.path.dirname(__file__), args.csv_dir)
        os.makedirs(export_dir, exist_ok=True)
        print(f"CSV export enabled. Writing files to: {export_dir}")

    for user_id in users:

        recs = pipeline.recommend(user_id)
        output_df = _attach_cart_items_before_recommendations(
            user_id=user_id,
            recs=recs,
            user_events=user_events,
            products=products,
        )

        print(f"\n--- Recommendations for {user_id} ---")

        if output_df is None or output_df.empty:
            print("No recommendations")
        else:
            print(output_df.to_string(index=False))
            if export_dir:
                user_token = _safe_user_token(user_id)
                export_path = os.path.join(export_dir, f"recommendation_for_{user_token}.csv")
                output_df.to_csv(export_path, index=False)
                print(f"Exported CSV: {export_path}")


if __name__ == "__main__":
    main()