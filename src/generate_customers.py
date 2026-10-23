"""Synthetic subscription / e-commerce customer generator.

Generates ~10,000 customers with behavioural features and a churn label
drawn from a DOCUMENTED logistic probability function of those features,
so the signal is real, learnable, and the EDA drivers match the true
generating process (which the model then has to rediscover).

Churn probability (logit):
    -0.90  intercept
    -0.040 * tenure_months          (loyalty: longer tenure -> less churn)
    -0.008 * monthly_spend          (higher spend -> slightly less churn)
    +0.040 * days_since_last_order  (going dormant -> strong churn signal)
    -0.120 * orders_last_90d        (recent activity protects)
    +0.300 * support_tickets        (bad experiences drive churn)
    +0.010 * discount_usage_pct     (discount-dependent customers churn)
    +0.450 * failed_payments        (billing friction drives churn)
    -0.022 * engagement_score       (engaged customers stay)

Snapshot date: 2026-10-01. signup_month = snapshot - tenure (cohort).
"""
import numpy as np
import pandas as pd

SNAPSHOT = pd.Timestamp("2026-10-01")

TRUE_COEFFICIENTS = {  # on raw (unstandardized) features — the ground truth
    "tenure_months": -0.040, "monthly_spend": -0.008,
    "days_since_last_order": 0.040, "orders_last_90d": -0.120,
    "support_tickets": 0.300, "discount_usage_pct": 0.010,
    "failed_payments": 0.450, "engagement_score": -0.022,
}
INTERCEPT = -0.90
FEATURES = list(TRUE_COEFFICIENTS)


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def generate(n_customers: int = 10_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    tenure = rng.integers(1, 49, n_customers)                       # 1-48 months
    engagement = np.clip(rng.normal(55, 22, n_customers), 0, 100)   # 0-100 score
    orders = np.clip(rng.poisson(np.maximum(engagement / 11, 0.2)), 0, 25)
    days_since = np.round(np.clip(
        rng.normal(105 - engagement * 0.95 - orders * 2.0, 18), 0, 120)).astype(int)
    spend = np.round(rng.lognormal(np.log(55), 0.55, n_customers)
                     * (0.7 + orders / 10), 2)
    tickets = np.clip(rng.poisson(1.0, n_customers), 0, 10)
    discount = np.round(np.clip(rng.normal(30, 20, n_customers), 0, 90), 1)
    failed = np.clip(rng.poisson(0.35, n_customers), 0, 5)

    df = pd.DataFrame({
        "customer_id": [f"CUST-{i:05d}" for i in range(n_customers)],
        "signup_month": [(SNAPSHOT - pd.DateOffset(months=int(t))).strftime("%Y-%m")
                         for t in tenure],
        "tenure_months": tenure,
        "monthly_spend": spend,
        "orders_last_90d": orders,
        "days_since_last_order": days_since,
        "support_tickets": tickets,
        "discount_usage_pct": discount,
        "failed_payments": failed,
        "engagement_score": np.round(engagement, 1),
    })

    logit = np.full(n_customers, INTERCEPT)
    for feat, coef in TRUE_COEFFICIENTS.items():
        logit = logit + coef * df[feat].to_numpy(dtype=float)
    df["churn_probability_truth"] = np.round(_sigmoid(logit), 4)
    df["churned"] = (rng.random(n_customers) < df["churn_probability_truth"]).astype(int)
    return df


if __name__ == "__main__":
    out = generate()
    print(out.shape)
    print("churn rate:", round(out.churned.mean(), 4))
