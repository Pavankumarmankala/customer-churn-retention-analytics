# Findings & Retention Playbook — Customer Churn & Retention Analytics

## What the data says (all figures computed from the 10,000-customer dataset)
- Overall churn rate: **26.1%** (2610 of 10000 customers).
- Strongest model driver: **days_since_last_order** (standardized coefficient 1.1915, odds ratio 3.292 per standard deviation).
- New customers (tenure 0-6 months) churn at **36.5%** vs **18.63%** for 37+ months.
- Customers with 3+ support tickets churn at **34.18%** vs **22.29%** with zero tickets.
- Customers with 2+ failed payments churn at **35.06%** vs **24.17%** with none.
- Customers dormant 60+ days (no order) churn at **58.3%**.

## Revenue at risk
Revenue at risk = sum of monthly spend of the high-risk tier (model probability >= 0.50);
annualised = monthly x 12.
- High-risk tier: **2075 customers**, **$104,630.33/month** (**$1,255,563.96/year**) at risk; this tier's actual churn rate was 73.16%.

## Retention strategies
1. **Proactive save-desk outreach — high-risk tier.** Target: 2075 customers scored >= 0.50 churn probability. Why: the model separates churners with test AUC 0.8901, and this tier actually churned at 73.16%. Revenue at stake: $104,630.33/month. Action: personal outreach + tailored offer before renewal.
2. **Service recovery for repeat support contacts.** Target: 831 customers with 3+ support tickets, churning at 34.18% (vs 22.29% with zero tickets). Revenue in segment: $64,550.54/month. Action: root-cause fix, named support contact, service credit, follow-up within 7 days of the third ticket.
3. **Failed-payment rescue (dunning fix).** Target: 462 customers with 2+ failed payments, churning at 35.06% (vs 24.17% with none). Revenue in segment: $36,620.78/month. Action: card-updater service, smart retry timing, payment-method backup prompt — fix billing friction before it becomes a cancellation.
4. **Dormancy win-back campaign.** Target: 2957 customers with no order in 60+ days, churning at 58.3%. Revenue in segment: $183,275.45/month. Action: re-engagement sequence at day 45 (before the 60-day cliff), personalised recommendations, limited-time incentive for discount-sensitive customers.
