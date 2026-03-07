import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from typing import Dict


def evaluate(y_true, y_pred) -> Dict[str, float]:
    out = {}
    try:
        out["roc_auc"] = roc_auc_score(y_true, y_pred)
    except Exception:
        out["roc_auc"] = float("nan")
    try:
        out["pr_auc"] = average_precision_score(y_true, y_pred)
    except Exception:
        out["pr_auc"] = float("nan")
    try:
        out["log_loss"] = log_loss(y_true, np.clip(y_pred, 1e-15, 1 - 1e-15))
    except Exception:
        out["log_loss"] = float("nan")
    return out


def precision_at_k_by_user(meta: pd.DataFrame, y_true: pd.Series, y_pred: pd.Series, k: int = 5, user_col: str = "user_id") -> float:
    if user_col not in meta.columns:
        # global top-k
        order = np.argsort(-y_pred)
        topk = order[:k]
        return float(y_true.iloc[topk].mean())

    df = pd.DataFrame({"user_id": meta[user_col], "y_true": y_true, "y_pred": y_pred})
    precisions = []
    for uid, g in df.groupby("user_id"):
        g = g.sort_values("y_pred", ascending=False)
        top = g.head(k)
        if len(top) == 0:
            continue
        precisions.append(top["y_true"].mean())
    return float(np.nanmean(precisions)) if precisions else 0.0
