"""Builds chart PNGs and a self-contained HTML dashboard from gold data."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "data" / "gold"; OUT = ROOT / "outputs"; CHARTS = OUT / "charts"
SILVER = ROOT / "data" / "silver"


def _roc_points(y, scores, n=101):
    thr = np.linspace(0, 1, n)
    tpr = [((scores >= t) & (y == 1)).sum() / max((y == 1).sum(), 1) for t in thr]
    fpr = [((scores >= t) & (y == 0)).sum() / max((y == 0).sum(), 1) for t in thr]
    return fpr, tpr


def build():
    CHARTS.mkdir(parents=True, exist_ok=True)
    tenure = pd.read_csv(GOLD / "churn_by_tenure_band.csv")
    tickets = pd.read_csv(GOLD / "churn_by_support_tickets.csv")
    tiers = pd.read_csv(GOLD / "tier_summary.csv")
    scored = pd.read_csv(SILVER / "customers_scored.csv")
    metrics = json.loads((OUT / "metrics.json").read_text())

    plt.figure(figsize=(7, 3.5))
    plt.bar(tenure.tenure_band.astype(str), tenure.churn_rate_pct)
    plt.ylabel("Churn rate %"); plt.title("Churn rate by tenure band"); plt.tight_layout()
    plt.savefig(CHARTS / "churn_by_tenure.png", dpi=150); plt.close()

    plt.figure(figsize=(7, 3.5))
    plt.bar(tickets.tickets_band.astype(str), tickets.churn_rate_pct)
    plt.xlabel("Support tickets"); plt.ylabel("Churn rate %")
    plt.title("Churn rate by support tickets"); plt.tight_layout()
    plt.savefig(CHARTS / "churn_by_support_tickets.png", dpi=150); plt.close()

    fpr, tpr = _roc_points(scored.churned.to_numpy(),
                           scored.churn_probability_model.to_numpy())
    plt.figure(figsize=(5.5, 4))
    plt.plot(fpr, tpr, label=f"Model (AUC {metrics['test_auc']} on test set)")
    plt.plot([0, 1], [0, 1], "--", color="grey", label="Chance")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC curve — churn model (full base, for shape)"); plt.legend(); plt.tight_layout()
    plt.savefig(CHARTS / "roc_curve.png", dpi=150); plt.close()

    plt.figure(figsize=(7, 3.5))
    plt.bar(tiers.risk_tier.astype(str), tiers.customers)
    plt.ylabel("Customers"); plt.title("Customers by predicted risk tier"); plt.tight_layout()
    plt.savefig(CHARTS / "risk_tier_distribution.png", dpi=150); plt.close()

    cards = "".join(f"<div class='card'><h3>{k.replace('_',' ').title()}</h3><p>{v}</p></div>"
                    for k, v in metrics.items() if not isinstance(v, dict))
    html = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>Customer Churn & Retention Analytics</title>
<style>body{{font-family:Arial,sans-serif;margin:24px;background:#f7f7f9}}
.card{{display:inline-block;background:#fff;border:1px solid #ddd;border-radius:8px;
padding:12px 16px;margin:6px;min-width:170px}} img{{max-width:100%;background:#fff;
border:1px solid #ddd;border-radius:8px;margin:8px 0}}</style></head><body>
<h1>Customer Churn &amp; Retention Analytics</h1>{cards}
<h2>Churn by tenure band</h2><img src='charts/churn_by_tenure.png'>
<h2>Churn by support tickets</h2><img src='charts/churn_by_support_tickets.png'>
<h2>ROC curve</h2><img src='charts/roc_curve.png'>
<h2>Risk tier distribution</h2><img src='charts/risk_tier_distribution.png'>
</body></html>"""
    (OUT / "dashboard.html").write_text(html)
    print("dashboard + 4 charts written to", OUT)


if __name__ == "__main__":
    build()
