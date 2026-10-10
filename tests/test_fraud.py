import numpy as np
import pandas as pd
import pytest

from part_a_fraud import models as m


def synth(n=2000, fraud_rate=0.05, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, 28)), columns=[f"V{i}" for i in range(1, 29)])
    df["Time"] = np.arange(n)
    df["Amount"] = rng.gamma(2, 50, size=n)
    df["Class"] = (rng.random(n) < fraud_rate).astype(int)
    df.loc[df.Class == 1, "V1"] += 4            # learnable signal
    return df


def test_cost_counts_missed_amount_and_false_alarms():
    y, pred, amt = [1, 1, 0, 0], [1, 0, 1, 0], [100, 250, 5, 5]
    assert m.cost(y, pred, amt, fp_cost=10) == 250 + 10


def test_best_threshold_prefers_catching_big_fraud():
    y = np.array([1, 0, 0, 0])
    proba = np.array([0.05, 0.04, 0.03, 0.02])
    amt = np.array([10_000, 1, 1, 1])
    t = m.best_threshold(y, proba, amt, fp_cost=10)
    assert proba[0] >= t                         # the $10k fraud must be flagged


def test_clean_rejects_bad_data():
    df = synth()
    with pytest.raises(AssertionError):
        m.clean(df.assign(Class=2))
    with pytest.raises(AssertionError):
        m.clean(df.assign(Amount=-1.0))
    with pytest.raises(AssertionError):
        m.clean(df.drop(columns=["V5"]))


def test_clean_drops_duplicates_and_sorts():
    df = synth(200)
    dirty = pd.concat([df, df.iloc[:10]]).sample(frac=1, random_state=1)
    out = m.clean(dirty)
    assert len(out) == 200 and out.Time.is_monotonic_increasing


def test_features_drop_time_and_log_amount():
    df = m.clean(synth(300))
    X, y, amt = m.make_features(df)
    assert "Time" not in X and "Class" not in X
    assert np.allclose(X["Amount"], np.log1p(amt))


def test_chronological_split_boundaries():
    a, b = m.chronological_split(1000)
    assert (a, b) == (700, 850)


def test_end_to_end_and_single_prediction():
    df = m.clean(synth())
    X, y, amt = m.make_features(df)
    a, b = m.chronological_split(len(df))
    model = m.build_models()["LightGBM"].fit(X.iloc[:a], y[:a])
    p_va = model.predict_proba(X.iloc[a:b])[:, 1]
    t = m.best_threshold(y[a:b], p_va, amt[a:b])
    raw = df.drop(columns=["Class", "Time"])
    fraud_i = int(np.where(y == 1)[0][-1])
    res = m.predict_transaction(raw.iloc[fraud_i].to_dict(), model, t, X.columns)
    assert set(res) == {"fraud_score", "prediction", "result"}
    assert 0 <= res["fraud_score"] <= 1


def test_predict_transaction_missing_feature_raises():
    df = m.clean(synth(400))
    X, y, _ = m.make_features(df)
    model = m.build_models()["Logistic Regression (baseline)"].fit(X, y)
    with pytest.raises(ValueError):
        m.predict_transaction({"Amount": 10.0}, model, 0.5, X.columns)
