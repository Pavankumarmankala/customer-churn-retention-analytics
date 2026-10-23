import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from generate_customers import generate, FEATURES
from churn_model import (LogisticRegressionGD, stratified_split, auc_score,
                         classification_metrics, risk_tier)


def test_dataset_sanity():
    df = generate()
    assert 8_000 <= len(df) <= 12_000
    assert 0.15 <= df["churned"].mean() <= 0.30
    assert df[FEATURES + ["churned"]].isna().sum().sum() == 0
    assert df["customer_id"].is_unique


def test_model_beats_thresholds():
    # Thresholds set from the verified full run: test AUC 0.8901,
    # accuracy 84.05% (see outputs/metrics.json). Margins left for safety.
    df = generate()
    train, test = stratified_split(df)
    model = LogisticRegressionGD().fit(train[FEATURES].to_numpy(float),
                                       train["churned"].to_numpy(float))
    proba = model.predict_proba(test[FEATURES].to_numpy(float))
    auc = auc_score(test["churned"].to_numpy(), proba)
    m = classification_metrics(test["churned"].to_numpy(), proba)
    assert auc > 0.80
    assert m["accuracy_pct"] > 75.0
    assert m["precision_churn_pct"] > 50.0 and m["recall_churn_pct"] > 40.0


def test_at_risk_output_shape():
    df = generate(n_customers=2_000)
    model = LogisticRegressionGD().fit(df[FEATURES].to_numpy(float),
                                       df["churned"].to_numpy(float))
    df["p"] = model.predict_proba(df[FEATURES].to_numpy(float))
    assert df["p"].between(0, 1).all()
    df["tier"] = df["p"].map(risk_tier)
    at_risk = df[df.tier.isin(["high", "medium"])].sort_values("p", ascending=False)
    assert len(at_risk) > 0
    assert at_risk["p"].is_monotonic_decreasing
    assert set(at_risk.tier) <= {"high", "medium"}
    # high tier really does churn more than low tier in-sample
    assert df[df.tier == "high"].churned.mean() > df[df.tier == "low"].churned.mean()
