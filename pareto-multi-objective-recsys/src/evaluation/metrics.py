from typing import Dict, Any
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss


def classification_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    """Compute ROC-AUC, PR-AUC (average precision), and log loss for binary labels.

    y_true: binary 0/1
    y_score: score or probability
    """
    metrics = {"roc_auc": None, "pr_auc": None, "log_loss": None}
    try:
        if len(np.unique(y_true)) > 1:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        else:
            metrics["roc_auc"] = float('nan')
    except Exception:
        metrics["roc_auc"] = float('nan')

    try:
        metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
    except Exception:
        metrics["pr_auc"] = float('nan')

    try:
        # clip probabilities to valid range
        probs = np.clip(y_score, 1e-12, 1 - 1e-12)
        metrics["log_loss"] = float(log_loss(y_true, probs))
    except Exception:
        metrics["log_loss"] = float('nan')

    return metrics
