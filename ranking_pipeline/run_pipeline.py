"""
run_pipeline.py — End-to-end runner for the recommendation ranking pipeline.

Usage:
    python run_pipeline.py                          # uses mock data
    python run_pipeline.py --products products.csv \\
                           --events user_events.csv \\
                           --image_emb image_embeddings.pkl \\
                           --text_emb text_embeddings.pkl

Steps:
    1. Load (or generate) data and embeddings.
    2. Train LightGBM and XGBoost rankers.
    3. Print validation metrics.
    4. Run inference for 5 sample users and display results.
"""

import os
import sys
import argparse
import pickle
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Ensure the ranking_pipeline package is importable when run from project root.
sys.path.insert(0, os.path.dirname(__file__))

from utils import (
    get_logger,
    generate_mock_products,
    generate_mock_user_events,
    generate_mock_embeddings,
    combine_embeddings,
    load_embeddings,
)
from candidate_generator import CandidateGenerator
from feature_engineering import FeatureEngineer, FEATURE_COLUMNS
from train_ranker import temporal_train_val_split, train
from inference import RecommendationPipeline

logger = get_logger("run_pipeline")


def _derive_brand_from_name(name: object) -> str:
    text = str(name or "").strip()
    if not text:
        return "unknown"
    first_token = text.split()[0].strip("-_/.,")
    return first_token or "unknown"


_TYPE_KEYWORD_MAP = {
    "perfume": ["perfume", "eau de parfum", "eau de toilette", "fragrance", "deo", "deodorant", "body mist"],
    "earring": ["earring", "earrings", "hoop", "stud"],
    "bracelet": ["bracelet", "bangle", "cuff"],
    "ring": [" ring", "rings"],
    "necklace": ["necklace", "chain", "pendant"],
    "watch": ["watch", "smartwatch"],
    "shoe": ["shoe", "sneaker", "loafer", "boot", "sandals", "sandal", "heels", "flats"],
    "shirt": ["shirt", "t-shirt", "tee"],
    "trouser": ["trouser", "pant", "pants", "jeans", "chino", "jogger"],
    "dress": ["dress", "gown"],
}


def _infer_type_from_name(name: object) -> str:
    text = f" {str(name or '').strip().lower()} "
    if not text.strip():
        return "unknown"
    for inferred_type, keywords in _TYPE_KEYWORD_MAP.items():
        for kw in keywords:
            if kw in text:
                return inferred_type
    return "unknown"


def _derive_product_type(product_row: pd.Series) -> str:
    for col in ["product_type", "article_type", "sub_category", "category"]:
        if col in product_row.index:
            value = str(product_row.get(col) or "").strip()
            lowered = value.lower()
            if value and lowered not in {"unknown", "none", "nan", ""}:
                return lowered

    for col in ["product_name", "name"]:
        if col in product_row.index:
            inferred = _infer_type_from_name(product_row.get(col))
            if inferred != "unknown":
                return inferred
    return "unknown"


def _derive_catalog_category(product_row: pd.Series) -> str:
    raw_category = str(product_row.get("category") or "").strip().lower()
    if raw_category and raw_category not in {"unknown", "none", "nan"}:
        return raw_category

    ptype = _derive_product_type(product_row)
    if ptype in {"perfume"}:
        return "personal care"
    if ptype in {"earring", "bracelet", "ring", "necklace", "watch", "wallet", "bag"}:
        return "accessories"
    if ptype in {"shoe", "sandal", "heel", "flat"}:
        return "shoes & footwear"
    if ptype != "unknown":
        return "apparel"
    return "unknown"


def _prepare_ranker_products(products_df: pd.DataFrame, events_df: pd.DataFrame | None = None) -> pd.DataFrame:
    products = products_df.copy()
    if "item_id" not in products.columns:
        if "id" in products.columns:
            products["item_id"] = products["id"]
        else:
            raise ValueError("products_df must include item_id or id")

    products["item_id"] = products["item_id"].astype(str)

    if "product_name" not in products.columns:
        if "name" in products.columns:
            products["product_name"] = products["name"]
        else:
            products["product_name"] = products["item_id"]

    if "category" not in products.columns:
        products["category"] = "unknown"
    products["category"] = products["category"].fillna("unknown").astype(str)

    if "brand" not in products.columns:
        products["brand"] = products["product_name"].map(_derive_brand_from_name)
    products["brand"] = products["brand"].fillna("unknown").astype(str)

    if "price" not in products.columns:
        products["price"] = 0.0
    products["price"] = pd.to_numeric(products["price"], errors="coerce").fillna(0.0)

    if "image_url" not in products.columns:
        products["image_url"] = ""
    if "in_stock" not in products.columns:
        products["in_stock"] = True

    popularity = pd.Series(0.0, index=products["item_id"], dtype=np.float32)
    ctr = pd.Series(0.0, index=products["item_id"], dtype=np.float32)

    if events_df is not None and not events_df.empty:
        events = events_df.copy()
        if "item_id" in events.columns:
            events["item_id"] = events["item_id"].astype(str)
            counts = events["item_id"].value_counts()
            if not counts.empty:
                popularity = counts.reindex(products["item_id"], fill_value=0).astype(np.float32)
                max_count = float(popularity.max())
                if max_count > 0:
                    popularity = popularity / max_count

            if "event_type" in events.columns:
                click_mask = events["event_type"].astype(str).eq("click")
                view_mask = events["event_type"].astype(str).isin(["view", "click"])
                click_counts = events.loc[click_mask, "item_id"].value_counts()
                view_counts = events.loc[view_mask, "item_id"].value_counts()
                ctr_num = click_counts.reindex(products["item_id"], fill_value=0).astype(np.float32)
                ctr_den = view_counts.reindex(products["item_id"], fill_value=0).astype(np.float32)
                ctr = np.where(ctr_den > 0, ctr_num / ctr_den, 0.0)
                ctr = pd.Series(ctr, index=products["item_id"], dtype=np.float32)

    products["popularity_score"] = popularity.reindex(products["item_id"], fill_value=0.0).to_numpy()
    products["ctr_score"] = ctr.reindex(products["item_id"], fill_value=0.0).to_numpy()
    return products


def _prepare_ranker_events(
    events_df: pd.DataFrame | None,
    user_id: str,
    cart_item_ids: list[str],
) -> pd.DataFrame:
    cols = ["user_id", "item_id", "event_type", "timestamp"]
    if events_df is None or events_df.empty:
        events = pd.DataFrame(columns=cols)
    else:
        events = events_df.copy()
        if "timestamp" not in events.columns:
            if "event_time" in events.columns:
                events = events.rename(columns={"event_time": "timestamp"})
            elif "created_at" in events.columns:
                events = events.rename(columns={"created_at": "timestamp"})
        missing = [c for c in ["user_id", "item_id", "event_type", "timestamp"] if c not in events.columns]
        if missing:
            raise ValueError(f"events_df missing columns: {missing}")
        events = events[["user_id", "item_id", "event_type", "timestamp"]].copy()
        events["user_id"] = events["user_id"].astype(str)
        events["item_id"] = events["item_id"].astype(str)
        events["event_type"] = events["event_type"].astype(str)
        events["timestamp"] = pd.to_datetime(events["timestamp"], errors="coerce", utc=True).dt.tz_localize(None)
        events = events.dropna(subset=["timestamp"])
        events = events[events["user_id"] == str(user_id)].copy()

    now = pd.Timestamp.utcnow().tz_localize(None)
    synthetic_rows = []
    for idx, item_id in enumerate(cart_item_ids):
        synthetic_rows.append(
            {
                "user_id": str(user_id),
                "item_id": str(item_id),
                "event_type": "add_to_cart",
                "timestamp": now - pd.Timedelta(minutes=max(0, len(cart_item_ids) - idx)),
            }
        )

    if synthetic_rows:
        events = pd.concat([events, pd.DataFrame(synthetic_rows)], ignore_index=True)

    if events.empty:
        return pd.DataFrame(columns=cols)

    events = events.sort_values("timestamp").reset_index(drop=True)
    return events


def _resolve_ranker_model_paths(model_path: str | None = None) -> dict[str, str]:
    paths: dict[str, str] = {}
    candidate_map = {
        "lgbm": os.path.join(os.path.dirname(__file__), "models", "lgbm_ranker.pkl"),
        "xgboost": os.path.join(os.path.dirname(__file__), "models", "xgboost_ranker.pkl"),
    }

    if model_path and os.path.exists(model_path):
        lower = os.path.basename(model_path).lower()
        if "xgboost" in lower or "xgb" in lower:
            paths["xgboost"] = model_path
        else:
            paths["lgbm"] = model_path

    for key, path in candidate_map.items():
        if key not in paths and os.path.exists(path):
            paths[key] = path

    return paths


def _build_metadata_candidates(
    products: pd.DataFrame,
    cart_item_ids: list[str],
    top_n: int,
) -> pd.DataFrame:
    prod = products.copy().set_index("item_id", drop=False)
    cart_ids = [item_id for item_id in cart_item_ids if item_id in prod.index]
    if not cart_ids:
        return pd.DataFrame(columns=["item_id", "similarity_score"])

    cart_categories = {
        _derive_catalog_category(prod.loc[item_id])
        for item_id in cart_ids
    }
    cart_types = {
        _derive_product_type(prod.loc[item_id])
        for item_id in cart_ids
    }
    known_cart_types = {t for t in cart_types if t != "unknown"}
    cart_genders = {
        str(prod.loc[item_id].get("gender") or "").strip().lower()
        for item_id in cart_ids
        if str(prod.loc[item_id].get("gender") or "").strip()
    }
    price_series = pd.to_numeric(prod["price"], errors="coerce") if "price" in prod.columns else pd.Series(dtype=float)
    non_cart_prices = [
        float(p) for idx, p in zip(prod.index.tolist(), price_series.tolist())
        if idx not in set(cart_item_ids) and pd.notna(p) and float(p) > 0
    ]
    min_price = min(non_cart_prices) if non_cart_prices else 0.0
    max_price = max(non_cart_prices) if non_cart_prices else 0.0

    rows = []
    for item_id, row in prod.iterrows():
        if item_id in set(cart_item_ids):
            continue

        score = 0.0
        category = _derive_catalog_category(row)
        product_type = _derive_product_type(row)
        gender = str(row.get("gender") or "").strip().lower()
        brand = _derive_brand_from_name(row.get("product_name") or row.get("name"))

        # Strongly prioritize same item type (perfume-to-perfume, earring-to-earring, etc.).
        if known_cart_types:
            if product_type in known_cart_types:
                score += 8.0
            elif product_type == "unknown":
                score -= 3.0
            else:
                score -= 2.0
        elif product_type in cart_types and product_type != "unknown":
            score += 4.0
        if category in cart_categories and category != "unknown":
            score += 2.0
        if gender and gender in cart_genders:
            score += 0.8

        cart_brands = {
            _derive_brand_from_name(prod.loc[item_id].get("product_name") or prod.loc[item_id].get("name"))
            for item_id in cart_ids
        }
        if brand in cart_brands and brand != "unknown":
            score += 0.6

        price = float(pd.to_numeric(row.get("price"), errors="coerce") or 0.0)
        # Lower price should rank higher, independent of cart average price.
        if price > 0 and max_price > min_price:
            cheapness = (max_price - price) / (max_price - min_price)
            score += 2.0 * float(np.clip(cheapness, 0.0, 1.0))
        elif price > 0 and max_price > 0:
            score += 1.0

        rows.append({"item_id": str(item_id), "similarity_score": float(score)})

    if not rows:
        return pd.DataFrame(columns=["item_id", "similarity_score"])

    out = pd.DataFrame(rows)
    return out.sort_values("similarity_score", ascending=False).head(top_n).reset_index(drop=True)


def _apply_trending_boost_to_scores(
    scores: np.ndarray,
    candidate_ids: pd.Series,
    products: pd.DataFrame,
    trending_boost: float,
) -> np.ndarray:
    boosted = np.asarray(scores, dtype=np.float32).copy()
    if "popularity_score" not in products.columns or len(products) == 0:
        return boosted

    prod = products.set_index("item_id", drop=False)
    threshold = float(prod["popularity_score"].quantile(0.80)) if len(prod) else None
    if threshold is None or np.isnan(threshold):
        return boosted

    for idx, item_id in enumerate(candidate_ids.astype(str).tolist()):
        try:
            pop = float(prod.loc[item_id, "popularity_score"])
        except Exception:
            continue
        if pop >= threshold:
            boosted[idx] *= trending_boost
    return boosted


def recommend_from_cart_items_with_ranker(
    products_df: pd.DataFrame,
    cart_item_ids: list,
    user_id: str,
    events_df: pd.DataFrame | None = None,
    embeddings_path: str | None = None,
    model_path: str | None = None,
    top_k: int = 6,
) -> pd.DataFrame:
    """
    Generate checkout recommendations using candidate retrieval plus a saved ranker.

    Falls back to an empty DataFrame if the trained ranker cannot be loaded.
    """
    if products_df is None or products_df.empty or not cart_item_ids:
        return pd.DataFrame()

    clean_cart_ids = [str(x) for x in cart_item_ids if str(x).strip()]
    if not clean_cart_ids:
        return pd.DataFrame()

    resolved_model_paths = _resolve_ranker_model_paths(model_path)
    if not resolved_model_paths:
        logger.warning("No saved ranker model found for checkout inference.")
        return pd.DataFrame()

    products = _prepare_ranker_products(products_df, events_df)
    user_events = _prepare_ranker_events(events_df, user_id, clean_cart_ids)
    if user_events.empty:
        logger.warning("No usable events for checkout ranker inference for user %s.", user_id)
        return pd.DataFrame()

    embeddings: dict[str, np.ndarray] = {}
    if embeddings_path is None:
        embeddings_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "embeddings",
            "models",
            "product_embeddings.pkl",
        )
    if os.path.exists(embeddings_path):
        try:
            with open(embeddings_path, "rb") as f:
                payload = pickle.load(f)
            embeddings = _normalize_embedding_payload(payload, products["item_id"].tolist())
        except Exception as exc:
            logger.warning("Failed to load embeddings for checkout ranker inference: %s", exc)
            embeddings = {}

    candidate_limit = max(top_k * 5, 30)
    candidate_products = products.copy()
    candidates = pd.DataFrame()
    if embeddings:
        embedded_products = products[products["item_id"].isin(set(embeddings.keys()))].copy()
        if not embedded_products.empty:
            cg = CandidateGenerator(
                embeddings=embeddings,
                products=embedded_products,
                user_events=user_events,
                top_n=candidate_limit,
                recency_days=365,
            )
            candidates = cg.get_candidates(
                str(user_id),
                filter_out_of_stock=False,
                exclude_purchased=False,
            )
            candidates["item_id"] = candidates["item_id"].astype(str)
            candidates = candidates[~candidates["item_id"].isin(set(clean_cart_ids))].reset_index(drop=True)
            if not candidates.empty:
                candidate_products = embedded_products

    if candidates.empty:
        candidates = _build_metadata_candidates(products, clean_cart_ids, candidate_limit)
        embeddings = {}

    if candidates.empty:
        return pd.DataFrame()

    fe = FeatureEngineer(
        products=candidate_products,
        user_events=user_events,
        embeddings=embeddings,
        recency_days=365,
    )

    models: dict[str, object] = {}
    for model_name, path in resolved_model_paths.items():
        try:
            with open(path, "rb") as f:
                models[model_name] = pickle.load(f)
        except Exception as exc:
            logger.warning("Failed to load saved ranker model %s: %s", path, exc)

    if not models:
        return pd.DataFrame()

    feat_df = fe.build_features(str(user_id), candidates)
    X = feat_df[FEATURE_COLUMNS].astype(np.float32)
    scored = feat_df.copy()
    scored["item_id"] = candidates["item_id"].astype(str).values
    scored["similarity_score"] = pd.to_numeric(candidates["similarity_score"], errors="coerce").fillna(0.0).values

    ranking_cols: list[str] = []
    for model_name, model in models.items():
        raw_scores = model.predict(X)
        boosted = _apply_trending_boost_to_scores(
            scores=raw_scores,
            candidate_ids=scored["item_id"],
            products=candidate_products,
            trending_boost=1.10,
        )
        col_name = "lgbm_ranking_score" if model_name == "lgbm" else "xgboost_ranking_score"
        scored[col_name] = boosted
        ranking_cols.append(col_name)

    scored["model_ensemble_score"] = scored[ranking_cols].mean(axis=1)

    sim_min = float(scored["similarity_score"].min())
    sim_max = float(scored["similarity_score"].max())
    if sim_max > sim_min:
        scored["similarity_norm"] = (scored["similarity_score"] - sim_min) / (sim_max - sim_min)
    else:
        scored["similarity_norm"] = 1.0

    model_min = float(scored["model_ensemble_score"].min())
    model_max = float(scored["model_ensemble_score"].max())
    if model_max > model_min:
        scored["model_norm"] = (scored["model_ensemble_score"] - model_min) / (model_max - model_min)
    else:
        scored["model_norm"] = 1.0

    # Business objective first (type match + cheaper price), model score second.
    scored["ranking_score"] = 0.70 * scored["similarity_norm"] + 0.30 * scored["model_norm"]
    ranked = scored.sort_values("ranking_score", ascending=False).reset_index(drop=True)

    selector_model = next(iter(models.values()))
    selector = RecommendationPipeline(
        candidate_generator=None,
        feature_engineer=None,
        model=selector_model,
        products=candidate_products,
        top_k=top_k,
        diversity_max_per_category=3,
        trending_boost=1.10,
    )
    ranked = selector._apply_diversity(ranked)
    if ranked.empty:
        return pd.DataFrame()

    meta_cols = [
        c for c in ["item_id", "product_name", "name", "category", "brand", "price", "image_url"]
        if c in candidate_products.columns
    ]
    if meta_cols:
        meta = candidate_products[meta_cols].drop_duplicates(subset=["item_id"])
        ranked = ranked.drop(columns=[c for c in ["product_name", "category", "brand", "price"] if c in ranked.columns], errors="ignore")
        ranked = ranked.merge(meta, on="item_id", how="left")

    if "name" not in ranked.columns and "product_name" in ranked.columns:
        ranked["name"] = ranked["product_name"]
    return ranked.head(top_k).reset_index(drop=True)


def _normalize_embedding_payload(payload, item_ids: list) -> dict:
    """Normalize common embedding payload shapes to item_id -> np.ndarray."""
    if isinstance(payload, dict):
        # Already item_id -> vector
        if payload and all(not isinstance(v, (dict, list, tuple)) or np.asarray(v).ndim == 1 for v in payload.values()):
            out = {}
            for k, v in payload.items():
                try:
                    out[str(k)] = np.asarray(v, dtype=np.float32)
                except Exception:
                    continue
            if out:
                return out

        # Common shape: {"product_ids": [...], "combined_embeddings": [[...], ...]}
        ids = payload.get("product_ids") or payload.get("item_ids")
        emb = payload.get("combined_embeddings") or payload.get("embeddings")
        if ids is not None and emb is not None:
            ids = [str(x) for x in list(ids)]
            arr = np.asarray(emb, dtype=np.float32)
            n = min(len(ids), len(arr))
            return {ids[i]: arr[i] for i in range(n)}

        # Fallback shape: modality arrays without ids, align to provided item_ids.
        for key in ("text_embeddings", "image_embeddings"):
            if key in payload:
                arr = np.asarray(payload[key], dtype=np.float32)
                n = min(len(item_ids), len(arr))
                return {str(item_ids[i]): arr[i] for i in range(n)}

    arr = np.asarray(payload, dtype=np.float32)
    n = min(len(item_ids), len(arr))
    return {str(item_ids[i]): arr[i] for i in range(n)}


def recommend_from_cart_items(
    products_df: pd.DataFrame,
    cart_item_ids: list,
    embeddings_path: str | None = None,
    top_k: int = 6,
) -> pd.DataFrame:
    """
    Generate recommendations from cart items using ranking pipeline candidate generation.

    This is a lightweight inference helper for app integrations. It does not train models.
    """
    if products_df is None or products_df.empty or not cart_item_ids:
        return pd.DataFrame()

    products = products_df.copy()
    if "item_id" not in products.columns:
        if "id" in products.columns:
            products["item_id"] = products["id"]
        else:
            return pd.DataFrame()

    products["item_id"] = products["item_id"].astype(str)
    if "in_stock" not in products.columns:
        products["in_stock"] = True
    if "popularity_score" not in products.columns:
        products["popularity_score"] = np.random.default_rng(7).uniform(0, 1, len(products))

    clean_cart_ids = [str(x) for x in cart_item_ids if str(x).strip()]
    if not clean_cart_ids:
        return pd.DataFrame()

    if embeddings_path is None:
        embeddings_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "embeddings",
            "models",
            "product_embeddings.pkl",
        )

    if not os.path.exists(embeddings_path):
        return pd.DataFrame()

    try:
        with open(embeddings_path, "rb") as f:
            payload = pickle.load(f)
        embeddings = _normalize_embedding_payload(payload, products["item_id"].tolist())
    except Exception:
        return pd.DataFrame()

    if not embeddings:
        return pd.DataFrame()

    # Build synthetic short interaction history from current cart.
    now = datetime.utcnow()
    events = []
    for i, item_id in enumerate(clean_cart_ids):
        events.append(
            {
                "user_id": "checkout_user",
                "item_id": item_id,
                "event_type": "add_to_cart",
                "timestamp": now - timedelta(minutes=(len(clean_cart_ids) - i)),
            }
        )
    user_events = pd.DataFrame(events)

    cg = CandidateGenerator(
        embeddings=embeddings,
        products=products,
        user_events=user_events,
        top_n=max(top_k * 4, 24),
        recency_days=365,
    )

    cands = cg.get_candidates(
        "checkout_user",
        filter_out_of_stock=False,
        exclude_purchased=False,
    )
    if cands.empty:
        return pd.DataFrame()

    cands["item_id"] = cands["item_id"].astype(str)
    cands = cands[~cands["item_id"].isin(set(clean_cart_ids))]
    if cands.empty:
        return pd.DataFrame()

    meta_cols = [c for c in ["item_id", "product_name", "name", "category", "price", "image_url"] if c in products.columns]
    out = cands.merge(products[meta_cols], on="item_id", how="left")
    out = out.sort_values("similarity_score", ascending=False).head(top_k).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="E-commerce recommendation ranking pipeline")
    p.add_argument("--products",   default=None, help="Path to products.csv")
    p.add_argument("--events",     default=None, help="Path to user_events.csv")
    p.add_argument("--image_emb",  default=None, help="Path to image_embeddings.pkl")
    p.add_argument("--text_emb",   default=None, help="Path to text_embeddings.pkl")
    p.add_argument("--model",      default="both", choices=["lgbm", "xgboost", "both"])
    p.add_argument("--top_k",      default=10, type=int)
    p.add_argument("--top_n_cands",default=100, type=int)
    p.add_argument("--output_dir", default="models")
    p.add_argument("--user_ids",   default=None, nargs="+", help="Specific user_ids for inference demo (can be login emails)")
    p.add_argument("--export_csv", action="store_true", dest="export_csv", help="Export recommendations per user to CSV files (enabled by default)")
    p.add_argument("--no_export_csv", action="store_false", dest="export_csv", help="Disable CSV export")
    p.add_argument("--csv_dir",    default="recommendation_exports", help="Directory for exported recommendation CSV files")
    p.set_defaults(export_csv=True)
    return p.parse_args()


def _safe_user_token(user_id: str) -> str:
    """Convert a user id/email to a filesystem-safe token for CSV file names."""
    # Keep common email characters so exports remain human-readable.
    token = "".join(ch if ch.isalnum() or ch in ("-", "_", "@", ".", "+") else "_" for ch in str(user_id))
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

    prod = products.copy()
    if "item_id" in prod.columns:
        prod = prod.set_index("item_id")

    cart_rows = []
    for item_id in cart_item_ids:
        item = prod.loc[item_id] if item_id in prod.index else pd.Series(dtype=object)
        cart_rows.append({
            "rank": 0,
            "item_id": item_id,
            "product_name": item.get("product_name", item_id),
            "category": item.get("category", "N/A"),
            "brand": item.get("brand", "N/A"),
            "price": float(item.get("price", 0.0)) if pd.notna(item.get("price", np.nan)) else 0.0,
            "similarity_score": np.nan,
            "ranking_score": np.nan,
            "row_type": "cart_item",
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


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_or_generate_data(args):
    """Load real data files if provided, otherwise generate mock data."""
    use_mock = args.products is None or args.events is None

    if use_mock:
        logger.info("Real data not provided — generating mock data.")
        products = generate_mock_products(n=500)
        user_events = generate_mock_user_events(products, n_users=200, events_per_user=30)
    else:
        logger.info("Loading products from %s", args.products)
        products = pd.read_csv(args.products)
        required_item_cols = {"item_id", "product_name", "category", "brand", "price"}
        missing = required_item_cols - set(products.columns)
        if missing:
            raise ValueError(f"products.csv is missing columns: {missing}")

        logger.info("Loading user events from %s", args.events)
        user_events = pd.read_csv(args.events, parse_dates=["timestamp"])

    # Ensure in_stock and score columns exist
    if "in_stock" not in products.columns:
        products["in_stock"] = True
    if "popularity_score" not in products.columns:
        products["popularity_score"] = np.random.default_rng(0).uniform(0, 1, len(products))
    if "ctr_score" not in products.columns:
        products["ctr_score"] = np.random.default_rng(1).uniform(0, 1, len(products))

    return products, user_events


def load_or_generate_embeddings(args, item_ids: list):
    """Load real embeddings or generate mock ones."""
    if args.image_emb and args.text_emb:
        logger.info("Loading embeddings from disk.")
        image_emb = load_embeddings(args.image_emb)
        text_emb = load_embeddings(args.text_emb)
    else:
        logger.info("Generating mock embeddings (dim=128).")
        image_emb = generate_mock_embeddings(item_ids, dim=128, seed=10)
        text_emb = generate_mock_embeddings(item_ids, dim=128, seed=20)

    combined = combine_embeddings(image_emb, text_emb, image_weight=0.5, text_weight=0.5)
    return combined


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ------------------------------------------------------------------ #
    # 1. Data                                                              #
    # ------------------------------------------------------------------ #
    products, user_events = load_or_generate_data(args)
    item_ids = products["item_id"].tolist()
    logger.info("Products: %d | Events: %d", len(products), len(user_events))

    # ------------------------------------------------------------------ #
    # 2. Embeddings                                                        #
    # ------------------------------------------------------------------ #
    embeddings = load_or_generate_embeddings(args, item_ids)

    # ------------------------------------------------------------------ #
    # 3. Candidate generator                                               #
    # ------------------------------------------------------------------ #
    cg = CandidateGenerator(
        embeddings=embeddings,
        products=products,
        user_events=user_events,
        top_n=args.top_n_cands,
        recency_days=90,
    )

    # ------------------------------------------------------------------ #
    # 4. Feature engineering + training dataset                            #
    # ------------------------------------------------------------------ #
    fe = FeatureEngineer(
        products=products,
        user_events=user_events,
        embeddings=embeddings,
        recency_days=90,
    )

    logger.info("Building training dataset (this may take a moment)…")
    all_users = user_events["user_id"].unique().tolist()
    train_full_df = fe.build_training_dataset(
        user_ids=all_users,
        top_n_candidates=args.top_n_cands,
        candidate_generator=cg,
    )

    # ------------------------------------------------------------------ #
    # 5. Temporal train/val split                                          #
    # ------------------------------------------------------------------ #
    train_df, val_df = temporal_train_val_split(
        train_full_df, user_events, val_fraction=0.2
    )

    if train_df.empty or val_df.empty:
        logger.error("Train or validation set is empty — aborting training.")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # 6. Train models                                                      #
    # ------------------------------------------------------------------ #
    trained_models = train(
        train_df=train_df,
        val_df=val_df,
        model_type=args.model,
        output_dir=os.path.join(os.path.dirname(__file__), args.output_dir),
    )

    if not trained_models:
        logger.error("No models were successfully trained.")
        sys.exit(1)

    # Pick the best available model for inference demo
    model = trained_models.get("lgbm") or trained_models.get("xgboost")

    # ------------------------------------------------------------------ #
    # 7. Inference demo                                                     #
    # ------------------------------------------------------------------ #
    pipeline = RecommendationPipeline(
        candidate_generator=cg,
        feature_engineer=fe,
        model=model,
        products=products,
        top_k=args.top_k,
        diversity_max_per_category=3,
        trending_boost=1.10,
    )

    demo_users = args.user_ids or user_events["user_id"].unique()[:5].tolist()
    print("\n" + "=" * 70)
    print("  INFERENCE DEMO — Top-10 Recommendations")
    print("=" * 70)

    export_dir = None
    if args.export_csv:
        export_dir = os.path.join(os.path.dirname(__file__), args.csv_dir)
        os.makedirs(export_dir, exist_ok=True)
        print(f"CSV export enabled. Writing files to: {export_dir}")

    for user_id in demo_users:
        recs = pipeline.recommend(user_id, verbose=True)
        output_df = _attach_cart_items_before_recommendations(
            user_id=user_id,
            recs=recs,
            user_events=user_events,
            products=products,
        )
        print(f"\n--- Recommendations for {user_id} ---")
        if output_df is None or output_df.empty:
            print("  No recommendations available.")
        else:
            pd.set_option("display.max_colwidth", 35)
            pd.set_option("display.width", 120)
            print(output_df.to_string(index=False))

            if export_dir:
                user_token = _safe_user_token(user_id)
                export_path = os.path.join(export_dir, f"recommendation_for_{user_token}.csv")
                output_df.to_csv(export_path, index=False)
                print(f"  Exported CSV: {export_path}")

    print("\nPipeline run complete.")


if __name__ == "__main__":
    main()
