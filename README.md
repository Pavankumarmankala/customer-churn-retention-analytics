# Customer Churn & Retention Analytics

End-to-end data analyst project: identify which customers are at risk of
leaving, quantify the revenue at risk, and turn the drivers into a
retention playbook. A logistic regression model is implemented from
scratch in NumPy (no scikit-learn) so every number can be traced.

## Verified results (actual run, seed = 42)

| Metric | Value |
|---|---|
| Customers analysed | 10,000 |
| Overall churn rate | 26.10% (2,610 churned) |
| Total monthly revenue | $760,494.74 |
| Test set (stratified 20%) | 2,000 customers |
| Test AUC | 0.8901 |
| Test accuracy | 84.05% |
| Precision (churn class) | 74.34% |
| Recall (churn class) | 59.39% |
| High-risk customers (model p >= 0.50) | 2,075 — actual churn 73.16% |
| Monthly revenue at risk (high tier) | $104,630.33 |
| Annualised revenue at risk (high tier) | $1,255,563.96 |
| Pipeline runtime | 0.73s |

Full numbers: `outputs/metrics.json`. Playbook: `outputs/findings.md`.
Charts + HTML dashboard: `outputs/`.

## Churn drivers (computed, gold tables in data/gold/)
The label was generated from a documented logistic function of the
features (`src/generate_customers.py`), and the model + EDA had to
rediscover them — they did:
- **Days since last order is the strongest driver** — standardized
  coefficient 1.19, odds ratio 3.29 per SD. Customers dormant 60+ days
  churn at 58.30% (2,957 customers).
- **Tenure protects** — 0-6 month customers churn at 36.50% vs 18.63%
  for 37+ months.
- **Spend protects** — <$25/month customers churn at 48.09% vs 10.45%
  for $100+/month.
- **Support friction hurts** — 3+ tickets: 34.18% churn vs 22.29% with
  zero tickets.
- **Billing friction hurts** — 2+ failed payments: 35.06% vs 24.17%
  with none; 3+ failed payments: 44.23%.

## Retention playbook (summary — full version in outputs/findings.md)
1. **Proactive save-desk outreach** to the 2,075 high-risk customers —
   $104,630.33/month at stake, tier actually churned at 73.16%.
2. **Service recovery** for the 831 customers with 3+ support tickets
   (34.18% churn) — $64,550.54/month in segment revenue.
3. **Failed-payment rescue** for the 462 customers with 2+ failed
   payments (35.06% churn) — $36,620.78/month; card updater + smart retries.
4. **Dormancy win-back** for the 2,957 customers with no order in 60+
   days (58.30% churn) — $183,275.45/month; engage at day 45, before
   the 60-day cliff.

Revenue at risk is defined as the sum of monthly spend of the
high-risk tier; annualised = monthly x 12.

## Run it
```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python run_pipeline.py     # generates, analyses, models, dashboards
.venv/bin/python -m pytest tests/    # 3 tests (dataset, model AUC, at-risk output)
```

## Layout
- `src/generate_customers.py` — synthetic 10k-customer dataset; churn drawn from a documented logistic function of 8 behavioural features
- `src/churn_model.py` — logistic regression from scratch in NumPy (gradient descent, stratified split, rank-based AUC)
- `src/pipeline.py` — bronze (raw Parquet) -> silver (validated, model-scored) -> gold (driver tables, at-risk list, tier summary); also generates metrics.json + findings.md
- `src/dashboard.py` — charts (churn by tenure, churn by tickets, ROC, tier distribution) + HTML dashboard
- `sql/analytics_queries.sql` — 10 Athena/Redshift queries (churn by segment, cohort retention by signup month, revenue at risk, model calibration…)

## Resume bullets (verified — safe to use)
**Customer Churn & Retention Analytics**
- Analysed churn across 10,000 customers (26.1% churn rate, $760K monthly revenue) in Python (pandas, NumPy) and SQL, identifying dormancy as the strongest churn driver — customers with no order in 60+ days churned at 58.3%, and new customers (0-6 months) at 36.5% vs 18.6% for 37+ month tenure.
- Built a logistic regression model from scratch in NumPy (gradient descent on standardized features) that scored a test AUC of 0.89 and 84.1% accuracy, segmenting 2,075 customers into a high-risk tier that actually churned at 73.2% with $104.6K in monthly revenue at risk ($1.26M annualised).
- Delivered a ranked at-risk customer list and a 4-strategy retention playbook targeting computed drivers (support-ticket recovery for 831 customers at 34.2% churn, failed-payment rescue for 462 at 35.1%, dormancy win-back for 2,957 at 58.3%), with 10 SQL queries for cohort retention and revenue-at-risk reporting.
