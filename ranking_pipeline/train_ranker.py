"""
train_ranker.py — Stage 3: Train LightGBM and XGBoost Learning-to-Rank models.

Design decisions:
- Train/validation split is done by timestamp to prevent label leakage.
- Group structure (required by learning-to-rank) is derived from user_id.
- Both LambdaRank (LightGBM) and rank:pairwise / rank:ndcg (XGBoost) are supported.
- Models and encoders are serialised with pickle for re-use in inference.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from utils import get_logger, evaluate_rankings, LABEL_MAP
from feature_engineering import FEATURE_COLUMNS

logger = get_logger("train_ranker")


# ---------------------------------------------------------------------------
# Model wrappers
# ---------------------------------------------------------------------------

class LightGBMRanker:
    """Thin wrapper around lightgbm.LGBMRanker."""

    def __getstate__(self): return self.__dict__
    def __setstate__(self, d): self.__dict__.update(d)

    def __init__(self, **kwargs):
        import lightgbm as lgb
        default_params = dict(
            objective="lambdarank",
            metric="ndcg",
            eval_at=[5, 10],
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            max_depth=-1,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        default_params.update(kwargs)
        self.model = lgb.LGBMRanker(**default_params)

    def fit(self, X_train, y_train, groups_train, X_val, y_val, groups_val):
        import lightgbm as lgb
        self.model.fit(
            X_train, y_train,
            group=groups_train,
            eval_set=[(X_val, y_val)],
            eval_group=[groups_val],
            callbacks=[
                lgb.early_stopping(stopping_rounds=30, verbose=False),
                lgb.log_evaluation(period=50),
            ],
        )
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def feature_importances(self, feature_names: list) -> pd.Series:
        return pd.Series(
            self.model.feature_importances_,
            index=feature_names,
        ).sort_values(ascending=False)


class XGBoostRanker:
    """Thin wrapper around xgboost.XGBRanker."""

    def __getstate__(self): return self.__dict__
    def __setstate__(self, d): self.__dict__.update(d)

    def __init__(self, **kwargs):
        import xgboost as xgb
        default_params = dict(
            objective="rank:ndcg",
            eval_metric="ndcg@10",
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
            device="cpu",
        )
        default_params.update(kwargs)
        self.model = xgb.XGBRanker(**default_params)

    def fit(self, X_train, y_train, groups_train, X_val, y_val, groups_val):
        self.model.fit(
            X_train, y_train,
            qid=_groups_to_qid(groups_train),
            eval_set=[(X_val, y_val)],
            eval_qid=[_groups_to_qid(groups_val)],
            verbose=50,
        )
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def feature_importances(self, feature_names: list) -> pd.Series:
        return pd.Series(
            self.model.feature_importances_,
            index=feature_names,
        ).sort_values(ascending=False)


def _groups_to_qid(groups: np.ndarray) -> np.ndarray:
    """Convert group sizes array to per-sample qid array (required by XGBoost)."""
    qid = np.repeat(np.arange(len(groups)), groups)
    return qid


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def temporal_train_val_split(
    df: pd.DataFrame,
    user_events: pd.DataFrame,
    val_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by timestamp: early interactions → train, recent → validation.
    Groups (user_id) that appear in validation are removed from training
    to prevent leakage (a user can only be in one split).
    """
    user_events = user_events.copy()
    user_events["timestamp"] = pd.to_datetime(user_events["timestamp"])

    # Compute latest event timestamp per user
    last_event = user_events.groupby("user_id")["timestamp"].max()
    cutoff = last_event.quantile(1 - val_fraction)

    val_users = set(last_event[last_event >= cutoff].index)
    train_users = set(last_event[last_event < cutoff].index)

    train_df = df[df["user_id"].isin(train_users)].reset_index(drop=True)
    val_df = df[df["user_id"].isin(val_users)].reset_index(drop=True)

    logger.info(
        "Split: train=%d rows (%d users), val=%d rows (%d users)",
        len(train_df), len(train_users), len(val_df), len(val_users),
    )
    return train_df, val_df


def prepare_ranking_arrays(df: pd.DataFrame):
    """
    Convert feature DataFrame into numpy arrays + group sizes for LTR.
    DataFrame must contain 'user_id', 'label', and all FEATURE_COLUMNS.
    Returns (X, y, groups).
    """
    df = df.sort_values("user_id").reset_index(drop=True)
    X = df[FEATURE_COLUMNS].astype(np.float32)
    y = df["label"].values.astype(np.int32)
    groups = df.groupby("user_id", sort=False).size().values
    return X, y, groups


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def compute_val_metrics(
    model,
    val_df: pd.DataFrame,
    k: int = 10,
) -> dict:
    """
    Score validation data, produce per-user ranking, then compute metrics.
    """
    X_val, _, _ = prepare_ranking_arrays(val_df)
    scores = model.predict(X_val)
    val_df = val_df.copy()
    val_df["score"] = scores

    results = []
    for user_id, grp in val_df.groupby("user_id"):
        grp_sorted = grp.sort_values("score", ascending=False)
        recommended = grp_sorted["item_id"].tolist()
        relevant_scores = dict(zip(grp["item_id"], grp["label"]))
        relevant_set = set(grp.loc[grp["label"] > 0, "item_id"])
        results.append({
            "recommended": recommended,
            "relevant_scores": relevant_scores,
            "relevant_set": relevant_set,
        })

    return evaluate_rankings(results, k=k)


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_type: str = "both",  # "lgbm", "xgboost", or "both"
    output_dir: str = "models",
) -> dict:
    """
    Train one or both rankers and save them to disk.

    Returns dict with keys 'lgbm' and/or 'xgboost' pointing to model instances.
    """
    os.makedirs(output_dir, exist_ok=True)

    X_train, y_train, g_train = prepare_ranking_arrays(train_df)
    X_val, y_val, g_val = prepare_ranking_arrays(val_df)

    trained = {}

    if model_type in ("lgbm", "both"):
        logger.info("Training LightGBM Ranker…")
        try:
            lgbm_model = LightGBMRanker()
            lgbm_model.fit(X_train, y_train, g_train, X_val, y_val, g_val)
            metrics = compute_val_metrics(lgbm_model, val_df)
            logger.info("LightGBM validation metrics: %s", metrics)
            _print_metrics("LightGBM", metrics)

            path = os.path.join(output_dir, "lgbm_ranker.pkl")
            with open(path, "wb") as f:
                pickle.dump(lgbm_model, f)
            logger.info("LightGBM model saved to %s", path)
            trained["lgbm"] = lgbm_model

            # Feature importance
            fi = lgbm_model.feature_importances(FEATURE_COLUMNS)
            logger.info("Top-5 LightGBM features:\n%s", fi.head(5).to_string())
        except Exception as e:
            logger.error("LightGBM training failed: %s", e)

    if model_type in ("xgboost", "both"):
        logger.info("Training XGBoost Ranker…")
        try:
            xgb_model = XGBoostRanker()
            xgb_model.fit(X_train, y_train, g_train, X_val, y_val, g_val)
            metrics = compute_val_metrics(xgb_model, val_df)
            logger.info("XGBoost validation metrics: %s", metrics)
            _print_metrics("XGBoost", metrics)

            path = os.path.join(output_dir, "xgboost_ranker.pkl")
            with open(path, "wb") as f:
                pickle.dump(xgb_model, f)
            logger.info("XGBoost model saved to %s", path)
            trained["xgboost"] = xgb_model

            fi = xgb_model.feature_importances(FEATURE_COLUMNS)
            logger.info("Top-5 XGBoost features:\n%s", fi.head(5).to_string())
        except Exception as e:
            logger.error("XGBoost training failed: %s", e)

    return trained


def load_model(path: str):
    """Load a serialised model from disk."""
    with open(path, "rb") as f:
        model = pickle.load(f)
    logger.info("Loaded model from %s", path)
    return model


def _print_metrics(model_name: str, metrics: dict) -> None:
    print(f"\n{'='*50}")
    print(f"  {model_name} — Validation Metrics")
    print(f"{'='*50}")
    for k, v in metrics.items():
        print(f"  {k:<20} {v:.4f}")
    print(f"{'='*50}\n")
