from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15
FP_COST = 10.0  # illustrative assumption; replace with PayGuard's measured cost


def clean(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Integrity checks, drop inf/NaN/duplicates, sort oldest -> newest."""
    expected = {"Time", "Amount", "Class"} | {f"V{i}" for i in range(1, 29)}
    missing = expected - set(df_raw.columns)
    assert not missing, f"missing columns: {missing}"
    assert set(df_raw["Class"].unique()) <= {0, 1}, "Class must be 0/1"
    assert (df_raw["Amount"] >= 0).all(), "negative amounts found"
    return (df_raw.replace([np.inf, -np.inf], np.nan).dropna().drop_duplicates()
            .astype({"Class": int}).sort_values("Time").reset_index(drop=True))


def make_features(df: pd.DataFrame):
    """Return X (log1p Amount, no Time/Class), y, raw amounts."""
    amount_raw = df["Amount"].to_numpy()
    y = df["Class"].to_numpy()
    X = df.drop(columns=["Class", "Time"])
    X["Amount"] = np.log1p(X["Amount"])
    const = [c for c in X.columns if X[c].nunique() == 1]
    return X.drop(columns=const), y, amount_raw


def chronological_split(n: int, train=TRAIN_FRAC, val=VAL_FRAC):
    """Index boundaries (a, b): train=[:a], val=[a:b], test=[b:]."""
    return int(n * train), int(n * (train + val))


def build_models() -> dict:
    return {
        "Logistic Regression (baseline)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)),
        ]),
        "LightGBM": LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.8, class_weight="balanced",
            random_state=SEED, n_jobs=-1, verbosity=-1),
    }


def cost(y_true, pred, amounts, fp_cost: float = FP_COST) -> float:
    """Missed fraud costs its amount; each false alarm costs fp_cost."""
    y_true, pred, amounts = map(np.asarray, (y_true, pred, amounts))
    missed = (y_true == 1) & (pred == 0)
    false_alarm = (y_true == 0) & (pred == 1)
    return float(amounts[missed].sum() + fp_cost * false_alarm.sum())


def best_threshold(y_true, proba, amounts, fp_cost: float = FP_COST) -> float:
    """Lowest-cost threshold. Call ONLY on validation data."""
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    cands = np.unique(np.concatenate([np.linspace(0.01, 0.99, 99), proba[y_true == 1]]))
    costs = np.array([cost(y_true, (proba >= t).astype(int), amounts, fp_cost) for t in cands])
    return float(cands[np.flatnonzero(costs == costs.min())[-1]])  # ties -> higher threshold


def predict_transaction(transaction: dict, model, threshold: float, feature_columns) -> dict:
    """Score ONE transaction. Input has RAW V1..V28 and RAW Amount; log1p applied once here."""
    row = pd.DataFrame([transaction])
    missing = set(feature_columns) - set(row.columns)
    if missing:
        raise ValueError(f"missing features: {sorted(missing)}")
    row["Amount"] = np.log1p(row["Amount"])
    row = row[list(feature_columns)]
    score = float(model.predict_proba(row)[0, 1])
    pred = int(score >= threshold)
    return {"fraud_score": round(score, 6), "prediction": pred,
            "result": "FRAUD" if pred else "LEGITIMATE"}
