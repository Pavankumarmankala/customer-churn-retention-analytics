-- Customer Churn & Retention Analytics
-- Dialect: Amazon Athena / Redshift (standard SQL). Source: silver layer
-- Table: customers_scored (one row per customer, model probability + tier)

-- 1. Churn rate by tenure band
SELECT CASE WHEN tenure_months <= 6 THEN '0-6' WHEN tenure_months <= 12 THEN '7-12'
            WHEN tenure_months <= 24 THEN '13-24' WHEN tenure_months <= 36 THEN '25-36'
            ELSE '37+' END AS tenure_band,
       COUNT(*) AS customers, SUM(churned) AS churned,
       ROUND(100.0 * SUM(churned) / COUNT(*), 2) AS churn_rate_pct
FROM customers_scored GROUP BY 1 ORDER BY MIN(tenure_months);

-- 2. Churn rate by monthly spend band
SELECT CASE WHEN monthly_spend < 25 THEN '<$25' WHEN monthly_spend < 50 THEN '$25-50'
            WHEN monthly_spend < 75 THEN '$50-75' WHEN monthly_spend < 100 THEN '$75-100'
            ELSE '$100+' END AS spend_band,
       COUNT(*) AS customers, SUM(churned) AS churned,
       ROUND(100.0 * SUM(churned) / COUNT(*), 2) AS churn_rate_pct,
       SUM(monthly_spend) AS monthly_revenue
FROM customers_scored GROUP BY 1 ORDER BY MIN(monthly_spend);

-- 3. Churn rate by support tickets
SELECT support_tickets, COUNT(*) AS customers, SUM(churned) AS churned,
       ROUND(100.0 * SUM(churned) / COUNT(*), 2) AS churn_rate_pct
FROM customers_scored GROUP BY 1 ORDER BY 1;

-- 4. Churn rate by failed payments
SELECT failed_payments, COUNT(*) AS customers, SUM(churned) AS churned,
       ROUND(100.0 * SUM(churned) / COUNT(*), 2) AS churn_rate_pct
FROM customers_scored GROUP BY 1 ORDER BY 1;

-- 5. Cohort retention by signup month (retained = not churned at snapshot)
SELECT signup_month, COUNT(*) AS cohort_customers,
       SUM(1 - churned) AS retained_customers,
       ROUND(100.0 * SUM(1 - churned) / COUNT(*), 2) AS retention_rate_pct,
       ROUND(AVG(tenure_months), 1) AS avg_tenure_months,
       SUM(monthly_spend) AS monthly_revenue
FROM customers_scored GROUP BY 1 ORDER BY 1;

-- 6. Revenue at risk by predicted risk tier (monthly, and annualised x12)
SELECT risk_tier, COUNT(*) AS customers,
       SUM(monthly_spend) AS monthly_revenue_at_risk,
       SUM(monthly_spend) * 12 AS annual_revenue_at_risk,
       ROUND(100.0 * SUM(churned) / COUNT(*), 2) AS actual_churn_rate_pct
FROM customers_scored GROUP BY 1
ORDER BY CASE risk_tier WHEN 'high' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END;

-- 7. Ranked at-risk customer list for the retention team (high + medium)
SELECT customer_id, signup_month, tenure_months, monthly_spend,
       monthly_spend * 12 AS annual_value, days_since_last_order,
       support_tickets, failed_payments, engagement_score,
       churn_probability_model, risk_tier
FROM customers_scored
WHERE risk_tier IN ('high', 'medium')
ORDER BY churn_probability_model DESC;

-- 8. Churn by engagement band
SELECT CASE WHEN engagement_score < 25 THEN '0-24' WHEN engagement_score < 50 THEN '25-49'
            WHEN engagement_score < 75 THEN '50-74' ELSE '75-100' END AS engagement_band,
       COUNT(*) AS customers, SUM(churned) AS churned,
       ROUND(100.0 * SUM(churned) / COUNT(*), 2) AS churn_rate_pct
FROM customers_scored GROUP BY 1 ORDER BY MIN(engagement_score);

-- 9. Churn by dormancy (days since last order)
SELECT CASE WHEN days_since_last_order < 30 THEN '<30d' WHEN days_since_last_order < 60 THEN '30-59d'
            WHEN days_since_last_order < 90 THEN '60-89d' ELSE '90d+' END AS dormancy_band,
       COUNT(*) AS customers, SUM(churned) AS churned,
       ROUND(100.0 * SUM(churned) / COUNT(*), 2) AS churn_rate_pct
FROM customers_scored GROUP BY 1 ORDER BY MIN(days_since_last_order);

-- 10. Model calibration: predicted probability vs actual churn by tier
SELECT risk_tier, COUNT(*) AS customers,
       ROUND(AVG(churn_probability_model), 4) AS avg_predicted_prob,
       ROUND(100.0 * SUM(churned) / COUNT(*), 2) AS actual_churn_rate_pct
FROM customers_scored GROUP BY 1
ORDER BY CASE risk_tier WHEN 'high' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END;
