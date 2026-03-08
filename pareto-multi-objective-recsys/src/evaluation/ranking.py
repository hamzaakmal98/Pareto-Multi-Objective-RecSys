from typing import List, Dict, Optional
import numpy as np
import pandas as pd


def dcg_at_k(rels: List[float], k: int) -> float:
    rels = np.asarray(rels)[:k]
    if rels.size == 0:
        return 0.0
    gains = (2 ** rels - 1)
    discounts = np.log2(np.arange(2, rels.size + 2))
    return float(np.sum(gains / discounts))


def ndcg_at_k(true_rels: List[float], scores: List[float], k: int) -> float:
    # sort by scores descending
    order = np.argsort(scores)[::-1]
    rels = np.asarray(true_rels)[order]
    dcg = dcg_at_k(rels, k)
    # ideal DCG
    ideal = np.sort(true_rels)[::-1]
    idcg = dcg_at_k(ideal, k)
    return float(dcg / idcg) if idcg > 0 else 0.0


def precision_at_k(true_rels: List[int], scores: List[float], k: int) -> float:
    order = np.argsort(scores)[::-1]
    topk = np.asarray(true_rels)[order][:k]
    if topk.size == 0:
        return 0.0
    return float(np.sum(topk) / len(topk))


def recall_at_k(true_rels: List[int], scores: List[float], k: int) -> float:
    positives = np.sum(true_rels)
    if positives == 0:
        return 0.0
    order = np.argsort(scores)[::-1]
    topk = np.asarray(true_rels)[order][:k]
    return float(np.sum(topk) / positives)


def evaluate_ranking(df: pd.DataFrame, user_col: str, score_col: str, target_col: str, ks: List[int]) -> Dict[str, float]:
    """Compute average NDCG@k, Precision@k, Recall@k across users with at least one positive.

    df must contain user_col, score_col, target_col (binary).
    """
    users = df[user_col].unique()
    results = {f'ndcg@{k}': [] for k in ks}
    results.update({f'prec@{k}': [] for k in ks})
    results.update({f'recall@{k}': [] for k in ks})

    for u in users:
        sub = df[df[user_col] == u]
        if sub.shape[0] == 0:
            continue
        trues = sub[target_col].fillna(0).astype(int).values
        scores = sub[score_col].fillna(0).values
        # only evaluate users with at least one positive to avoid degenerate averages
        if trues.sum() == 0:
            continue
        for k in ks:
            results[f'ndcg@{k}'].append(ndcg_at_k(trues, scores, k))
            results[f'prec@{k}'].append(precision_at_k(trues, scores, k))
            results[f'recall@{k}'].append(recall_at_k(trues, scores, k))

    # average
    avg = {k: (float(np.mean(v)) if v else 0.0) for k, v in results.items()}
    return avg


def scalarize_scores(df: pd.DataFrame, score_cols: List[str], weights: List[float], out_col: str = 'scalar_score') -> pd.DataFrame:
    arr = df[score_cols].fillna(0.0).values.astype(float)
    # min-max normalize each column
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    denom = (maxs - mins)
    denom[denom == 0] = 1.0
    norm = (arr - mins) / denom
    scalar = norm.dot(np.array(weights))
    df[out_col] = scalar
    return df
