"""Logistic regression implemented from scratch in NumPy.

No scikit-learn: full-batch gradient descent on standardized features
with L2 regularization, a stratified 80/20 train/test split, and a
rank-based (Mann-Whitney) AUC. The point of hand-rolling it is that
every number reported can be traced to this file.
"""
import numpy as np
import pandas as pd
from generate_customers import FEATURES


def _sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -35, 35)))


def stratified_split(df: pd.DataFrame, seed: int = 42, test_frac: float = 0.2):
    rng = np.random.default_rng(seed)
    test_idx = []
    for cls in (0, 1):
        idx = df.index[df["churned"] == cls].to_numpy().copy()
        rng.shuffle(idx)
        test_idx.extend(idx[: int(len(idx) * test_frac)])
    test_mask = df.index.isin(test_idx)
    return df[~test_mask].reset_index(drop=True), df[test_mask].reset_index(drop=True)


class LogisticRegressionGD:
    def __init__(self, lr: float = 0.3, epochs: int = 1500, l2: float = 1e-4):
        self.lr, self.epochs, self.l2 = lr, epochs, l2

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.mean_, self.std_ = X.mean(axis=0), X.std(axis=0) + 1e-9
        Xs = (X - self.mean_) / self.std_
        n, d = Xs.shape
        self.coef_ = np.zeros(d)
        self.intercept_ = 0.0
        for _ in range(self.epochs):
            p = _sigmoid(Xs @ self.coef_ + self.intercept_)
            err = p - y
            grad_w = Xs.T @ err / n + self.l2 * self.coef_
            self.coef_ -= self.lr * grad_w
            self.intercept_ -= self.lr * err.mean()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xs = (X - self.mean_) / self.std_
        return _sigmoid(Xs @ self.coef_ + self.intercept_)


def auc_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney U)."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    pos, neg = y_true == 1, y_true == 0
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2)
                 / (pos.sum() * neg.sum()))


def classification_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    return {
        "accuracy_pct": round(100 * (tp + tn) / len(y_true), 2),
        "precision_churn_pct": round(100 * tp / (tp + fp), 2) if tp + fp else 0,
        "recall_churn_pct": round(100 * tp / (tp + fn), 2) if tp + fn else 0,
        "true_positives": tp, "false_positives": fp,
        "false_negatives": fn, "true_negatives": tn,
    }


def risk_tier(p: float) -> str:
    return "high" if p >= 0.5 else "medium" if p >= 0.30 else "low"
