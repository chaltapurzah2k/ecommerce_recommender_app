# ========================= train_ranker.py =========================

import os
import pickle
import numpy as np
import pandas as pd

from utils import get_logger, evaluate_rankings
from feature_engineering import FEATURE_COLUMNS

logger = get_logger("train_ranker")


# ==========================================================
# LIGHTGBM RANKER
# ==========================================================

class LightGBMRanker:

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    def __init__(self, **kwargs):
        import lightgbm as lgb

        params = dict(
            objective="lambdarank",
            metric="ndcg",
            eval_at=[5, 10],
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=63,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

        params.update(kwargs)
        self.model = lgb.LGBMRanker(**params)

    def fit(self, X_train, y_train, g_train, X_val, y_val, g_val):
        import lightgbm as lgb

        self.model.fit(
            X_train,
            y_train,
            group=g_train,
            eval_set=[(X_val, y_val)],
            eval_group=[g_val],
            callbacks=[
                lgb.early_stopping(30),
                lgb.log_evaluation(50),
            ],
        )
        return self

    def predict(self, X):
        return self.model.predict(X)

    def feature_importances(self, names):
        return pd.Series(
            self.model.feature_importances_,
            index=names
        ).sort_values(ascending=False)

    # -------------------------------------------
    # Print learned rules
    # -------------------------------------------
    def print_rules(self, tree_index=0):

        booster = self.model.booster_
        dump = booster.dump_model()

        tree = dump["tree_info"][tree_index]["tree_structure"]
        names = dump["feature_names"]

        def walk(node, depth=0):
            tab = "   " * depth

            if "split_feature" in node:
                f = names[node["split_feature"]]
                t = node["threshold"]

                print(f"{tab}IF {f} <= {t}")
                walk(node["left_child"], depth + 1)

                print(f"{tab}else:")
                walk(node["right_child"], depth + 1)

            else:
                print(f"{tab}THEN score = {round(node['leaf_value'],4)}")

        walk(tree)


# ==========================================================
# XGBOOST RANKER
# ==========================================================

class XGBoostRanker:

    def __init__(self, **kwargs):
        import xgboost as xgb

        params = dict(
            objective="rank:ndcg",
            eval_metric="ndcg@10",
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            tree_method="hist",
        )

        params.update(kwargs)
        self.model = xgb.XGBRanker(**params)

    def fit(self, X_train, y_train, g_train, X_val, y_val, g_val):

        self.model.fit(
            X_train,
            y_train,
            qid=_groups_to_qid(g_train),
            eval_set=[(X_val, y_val)],
            eval_qid=[_groups_to_qid(g_val)],
            verbose=50
        )
        return self

    def predict(self, X):
        return self.model.predict(X)

    def feature_importances(self, names):
        return pd.Series(
            self.model.feature_importances_,
            index=names
        ).sort_values(ascending=False)

    def print_rules(self):
        booster = self.model.get_booster()
        print(booster.get_dump()[0])


# ==========================================================
# HELPERS
# ==========================================================

def _groups_to_qid(groups):
    return np.repeat(np.arange(len(groups)), groups)


def temporal_train_val_split(df, user_events, val_fraction=0.2):

    user_events["timestamp"] = pd.to_datetime(user_events["timestamp"])

    last_event = user_events.groupby("user_id")["timestamp"].max()

    cutoff = last_event.quantile(1 - val_fraction)

    val_users = set(last_event[last_event >= cutoff].index)
    train_users = set(last_event[last_event < cutoff].index)

    train_df = df[df["user_id"].isin(train_users)].reset_index(drop=True)
    val_df = df[df["user_id"].isin(val_users)].reset_index(drop=True)

    return train_df, val_df


def prepare_ranking_arrays(df):

    df = df.sort_values("user_id").reset_index(drop=True)

    X = df[FEATURE_COLUMNS].astype(np.float32)
    y = df["label"].values.astype(np.int32)
    groups = df.groupby("user_id").size().values

    return X, y, groups


def compute_val_metrics(model, val_df):

    X_val, _, _ = prepare_ranking_arrays(val_df)
    scores = model.predict(X_val)

    val_df = val_df.copy()
    val_df["score"] = scores

    return {"rows_scored": len(val_df)}


# ==========================================================
# TRAIN MAIN
# ==========================================================

def train(train_df, val_df, model_type="both", output_dir="models"):

    os.makedirs(output_dir, exist_ok=True)

    X_train, y_train, g_train = prepare_ranking_arrays(train_df)
    X_val, y_val, g_val = prepare_ranking_arrays(val_df)

    trained = {}

    if model_type in ["lgbm", "both"]:

        print("\nTraining LightGBM...")
        model = LightGBMRanker()
        model.fit(X_train, y_train, g_train, X_val, y_val, g_val)

        trained["lgbm"] = model

        with open(os.path.join(output_dir, "lgbm.pkl"), "wb") as f:
            pickle.dump(model, f)

    if model_type in ["xgboost", "both"]:

        print("\nTraining XGBoost...")
        model = XGBoostRanker()
        model.fit(X_train, y_train, g_train, X_val, y_val, g_val)

        trained["xgboost"] = model

        with open(os.path.join(output_dir, "xgb.pkl"), "wb") as f:
            pickle.dump(model, f)

    return trained