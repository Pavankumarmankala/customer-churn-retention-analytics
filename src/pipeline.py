"""Medallion pipeline: bronze (raw) -> silver (validated + model-scored)
-> gold (EDA driver tables, at-risk list, tier summary).

Also writes outputs/metrics.json and outputs/findings.md (the retention
playbook, generated from the computed numbers so it can never drift
from the data). Revenue at risk is defined on MONTHLY spend, with the
annualised figure (x12) shown alongside.
"""
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
from generate_customers import generate, FEATURES, TRUE_COEFFICIENTS
from churn_model import (LogisticRegressionGD, stratified_split, auc_score,
                         classification_metrics, risk_tier)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"; OUT = ROOT / "outputs"

TENURE_BANDS = [0, 6, 12, 24, 36, 999]
TENURE_LABELS = ["0-6", "7-12", "13-24", "25-36", "37+"]
SPEND_BANDS = [0, 25, 50, 75, 100, 1e9]
SPEND_LABELS = ["<$25", "$25-50", "$50-75", "$75-100", "$100+"]


def _churn_table(df, col, labels=None):
    g = df.groupby(col).agg(customers=("customer_id", "count"),
                            churned=("churned", "sum"),
                            monthly_spend=("monthly_spend", "sum")).reset_index()
    g["churn_rate_pct"] = np.round(100 * g.churned / g.customers, 2)
    if labels is not None:
        g[col] = pd.Categorical(g[col], categories=labels, ordered=True)
        g = g.sort_values(col)
    return g


def run(n_customers: int = 10_000) -> dict:
    t_start = time.time()
    for layer in ["bronze", "silver", "gold"]:
        (DATA / layer).mkdir(parents=True, exist_ok=True)
    OUT.mkdir(exist_ok=True)

    # BRONZE - raw generated customer base
    raw = generate(n_customers=n_customers)
    raw.to_parquet(DATA / "bronze" / "customers_raw.parquet", index=False)
    raw.to_csv(DATA / "bronze" / "customers_raw.csv", index=False)

    # SILVER - validate, dedupe, train model, score every customer
    df = raw.drop_duplicates(subset=["customer_id"]).copy()
    for c in FEATURES:
        df = df[df[c].notna()]
    train, test = stratified_split(df)
    model = LogisticRegressionGD().fit(train[FEATURES].to_numpy(float),
                                       train["churned"].to_numpy(float))
    test_proba = model.predict_proba(test[FEATURES].to_numpy(float))
    test_metrics = classification_metrics(test["churned"].to_numpy(), test_proba)
    test_auc = round(auc_score(test["churned"].to_numpy(), test_proba), 4)

    df["churn_probability_model"] = np.round(
        model.predict_proba(df[FEATURES].to_numpy(float)), 4)
    df["risk_tier"] = df["churn_probability_model"].map(risk_tier)
    silver_cols = ["customer_id", "signup_month", "tenure_months", "monthly_spend",
                   "orders_last_90d", "days_since_last_order", "support_tickets",
                   "discount_usage_pct", "failed_payments", "engagement_score",
                   "churn_probability_truth", "churn_probability_model",
                   "risk_tier", "churned"]
    df[silver_cols].to_parquet(DATA / "silver" / "customers_scored.parquet", index=False)
    df[silver_cols].to_csv(DATA / "silver" / "customers_scored.csv", index=False)

    # GOLD - EDA driver tables
    df["tenure_band"] = pd.cut(df.tenure_months, TENURE_BANDS, labels=TENURE_LABELS)
    df["spend_band"] = pd.cut(df.monthly_spend, SPEND_BANDS, labels=SPEND_LABELS)
    df["tickets_band"] = np.where(df.support_tickets >= 4, "4+", df.support_tickets.astype(str))
    df["failed_band"] = np.where(df.failed_payments >= 3, "3+", df.failed_payments.astype(str))
    gold = {
        "churn_by_tenure_band": _churn_table(df, "tenure_band", TENURE_LABELS),
        "churn_by_spend_band": _churn_table(df, "spend_band", SPEND_LABELS),
        "churn_by_support_tickets": _churn_table(df, "tickets_band"),
        "churn_by_failed_payments": _churn_table(df, "failed_band"),
        "churn_by_cohort": _churn_table(df, "signup_month"),
    }
    coef = pd.DataFrame({
        "feature": FEATURES,
        "model_coefficient_standardized": np.round(model.coef_, 4),
        "model_odds_ratio_per_std": np.round(np.exp(model.coef_), 3),
        "true_coefficient_raw": [TRUE_COEFFICIENTS[f] for f in FEATURES],
    }).sort_values("model_coefficient_standardized", key=abs, ascending=False)
    gold["feature_coefficients"] = coef.reset_index(drop=True)

    tier_order = pd.CategoricalDtype(["high", "medium", "low"], ordered=True)
    tier_sum = df.assign(risk_tier=df.risk_tier.astype(tier_order)).groupby(
        "risk_tier", observed=True).agg(
        customers=("customer_id", "count"),
        avg_predicted_prob=("churn_probability_model", "mean"),
        actual_churn_rate_pct=("churned", lambda s: round(100 * s.mean(), 2)),
        monthly_revenue=("monthly_spend", "sum")).reset_index()
    tier_sum["annual_revenue"] = np.round(tier_sum.monthly_revenue * 12, 2)
    tier_sum["avg_predicted_prob"] = np.round(tier_sum.avg_predicted_prob, 4)
    tier_sum["monthly_revenue"] = np.round(tier_sum.monthly_revenue, 2)
    gold["tier_summary"] = tier_sum

    at_risk = (df[df.risk_tier.isin(["high", "medium"])]
               .sort_values("churn_probability_model", ascending=False)
               [["customer_id", "signup_month", "tenure_months", "monthly_spend",
                 "orders_last_90d", "days_since_last_order", "support_tickets",
                 "discount_usage_pct", "failed_payments", "engagement_score",
                 "churn_probability_model", "risk_tier", "churned"]].copy())
    at_risk["annual_value"] = np.round(at_risk.monthly_spend * 12, 2)
    gold["at_risk_customers"] = at_risk
    for name, frame in gold.items():
        frame.to_csv(DATA / "gold" / f"{name}.csv", index=False)

    # Segments for the retention playbook
    def seg(mask):
        s = df[mask]
        return {"customers": int(len(s)),
                "churn_rate_pct": round(100 * s.churned.mean(), 2) if len(s) else 0,
                "monthly_revenue": round(float(s.monthly_spend.sum()), 2)}
    segments = {
        "high_risk_tier": seg(df.risk_tier == "high"),
        "tickets_3plus": seg(df.support_tickets >= 3),
        "tickets_0": seg(df.support_tickets == 0),
        "failed_2plus": seg(df.failed_payments >= 2),
        "failed_0": seg(df.failed_payments == 0),
        "dormant_60plus": seg(df.days_since_last_order >= 60),
        "tenure_0_6": seg(df.tenure_band == "0-6"),
        "tenure_37plus": seg(df.tenure_band == "37+"),
    }

    high = tier_sum[tier_sum.risk_tier == "high"].iloc[0]
    elapsed = round(time.time() - t_start, 2)
    metrics = {
        "customers": int(len(df)),
        "churned_customers": int(df.churned.sum()),
        "overall_churn_rate_pct": round(100 * df.churned.mean(), 2),
        "monthly_revenue_total": round(float(df.monthly_spend.sum()), 2),
        "test_set_customers": int(len(test)),
        "test_auc": test_auc,
        **test_metrics,
        "high_risk_customers": int(high.customers),
        "medium_risk_customers": int(tier_sum[tier_sum.risk_tier == "medium"].iloc[0].customers),
        "at_risk_customers_total": int(len(at_risk)),
        "monthly_revenue_at_risk_high_tier": float(high.monthly_revenue),
        "annual_revenue_at_risk_high_tier": float(high.annual_revenue),
        "actual_churn_rate_high_tier_pct": float(high.actual_churn_rate_pct),
        "segments": segments,
        "pipeline_runtime_seconds": elapsed,
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    _write_findings(metrics, coef, tier_sum)
    print(json.dumps(metrics, indent=2))
    return metrics


def _write_findings(m, coef, tier_sum):
    s = m["segments"]
    top = coef.iloc[0]
    lines = f"""# Findings & Retention Playbook — Customer Churn & Retention Analytics

## What the data says (all figures computed from the 10,000-customer dataset)
- Overall churn rate: **{m['overall_churn_rate_pct']}%** ({m['churned_customers']} of {m['customers']} customers).
- Strongest model driver: **{top.feature}** (standardized coefficient {top.model_coefficient_standardized}, odds ratio {top.model_odds_ratio_per_std} per standard deviation).
- New customers (tenure 0-6 months) churn at **{s['tenure_0_6']['churn_rate_pct']}%** vs **{s['tenure_37plus']['churn_rate_pct']}%** for 37+ months.
- Customers with 3+ support tickets churn at **{s['tickets_3plus']['churn_rate_pct']}%** vs **{s['tickets_0']['churn_rate_pct']}%** with zero tickets.
- Customers with 2+ failed payments churn at **{s['failed_2plus']['churn_rate_pct']}%** vs **{s['failed_0']['churn_rate_pct']}%** with none.
- Customers dormant 60+ days (no order) churn at **{s['dormant_60plus']['churn_rate_pct']}%**.

## Revenue at risk
Revenue at risk = sum of monthly spend of the high-risk tier (model probability >= 0.50);
annualised = monthly x 12.
- High-risk tier: **{m['high_risk_customers']} customers**, **${m['monthly_revenue_at_risk_high_tier']:,.2f}/month** (**${m['annual_revenue_at_risk_high_tier']:,.2f}/year**) at risk; this tier's actual churn rate was {m['actual_churn_rate_high_tier_pct']}%.

## Retention strategies
1. **Proactive save-desk outreach — high-risk tier.** Target: {m['high_risk_customers']} customers scored >= 0.50 churn probability. Why: the model separates churners with test AUC {m['test_auc']}, and this tier actually churned at {m['actual_churn_rate_high_tier_pct']}%. Revenue at stake: ${m['monthly_revenue_at_risk_high_tier']:,.2f}/month. Action: personal outreach + tailored offer before renewal.
2. **Service recovery for repeat support contacts.** Target: {s['tickets_3plus']['customers']} customers with 3+ support tickets, churning at {s['tickets_3plus']['churn_rate_pct']}% (vs {s['tickets_0']['churn_rate_pct']}% with zero tickets). Revenue in segment: ${s['tickets_3plus']['monthly_revenue']:,.2f}/month. Action: root-cause fix, named support contact, service credit, follow-up within 7 days of the third ticket.
3. **Failed-payment rescue (dunning fix).** Target: {s['failed_2plus']['customers']} customers with 2+ failed payments, churning at {s['failed_2plus']['churn_rate_pct']}% (vs {s['failed_0']['churn_rate_pct']}% with none). Revenue in segment: ${s['failed_2plus']['monthly_revenue']:,.2f}/month. Action: card-updater service, smart retry timing, payment-method backup prompt — fix billing friction before it becomes a cancellation.
4. **Dormancy win-back campaign.** Target: {s['dormant_60plus']['customers']} customers with no order in 60+ days, churning at {s['dormant_60plus']['churn_rate_pct']}%. Revenue in segment: ${s['dormant_60plus']['monthly_revenue']:,.2f}/month. Action: re-engagement sequence at day 45 (before the 60-day cliff), personalised recommendations, limited-time incentive for discount-sensitive customers.
"""
    (OUT / "findings.md").write_text(lines)


if __name__ == "__main__":
    run()
