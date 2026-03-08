from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

from src.rerank.scalarization import scalarize_df
from src.evaluation.ranking import evaluate_ranking


def timestamp_baseline(preds: pd.DataFrame, user_col: str = 'user_id', item_col: str = 'video_id', time_col: str = 'timestamp') -> pd.DataFrame:
    df = preds.copy()
    if time_col not in df.columns:
        # fallback to original row order
        df['_order'] = np.arange(len(df))
        df = df.sort_values('_order', ascending=False)
    else:
        df = df.sort_values(time_col, ascending=False)
    return df[[user_col, item_col] + [c for c in df.columns if c not in (user_col, item_col)]]


def score_baseline(preds: pd.DataFrame, score_col: str) -> pd.DataFrame:
    df = preds.copy()
    if score_col not in df.columns:
        raise ValueError(f'{score_col} not found in predictions')
    df = df.sort_values(score_col, ascending=False)
    return df


def weighted_scalar_baseline(preds: pd.DataFrame, score_cols: List[str], weights: List[float]) -> pd.DataFrame:
    df = preds.copy()
    df = scalarize_df(df, score_cols, weights, out_col='scalar_score')
    df = df.sort_values('scalar_score', ascending=False)
    return df


def evaluate_baseline(preds: pd.DataFrame, baseline_df: pd.DataFrame, ks: List[int], user_col: str = 'user_id') -> Dict[str, Dict]:
    # Evaluate baseline_df by merging targets from preds
    merged = baseline_df.merge(preds, on=[user_col, 'video_id'], how='left', suffixes=('', '_orig'))
    score_col = None
    # choose first numeric score column available
    for c in ['scalar_score', 'like_score', 'longview_score', 'creator_score']:
        if c in merged.columns:
            score_col = c
            break
    results = {}
    if 'is_like' in merged.columns and score_col:
        results['is_like'] = evaluate_ranking(merged, user_col, score_col, 'is_like', ks)
    if 'long_view' in merged.columns and score_col:
        results['long_view'] = evaluate_ranking(merged, user_col, score_col, 'long_view', ks)
    if 'creator_interest_proxy' in merged.columns and score_col:
        results['creator'] = evaluate_ranking(merged, user_col, score_col, 'creator_interest_proxy', ks)
    return results


def sweep_weights(preds: pd.DataFrame, score_cols: List[str], weight_grid: List[List[float]], ks: List[int], user_col: str = 'user_id') -> Tuple[Dict[Tuple[float,...], Dict], Tuple[float,...]]:
    best_metrics = None
    best_weights = None
    results = {}
    # For each weight vector, normalize and evaluate average NDCG@K across tasks present
    for w in weight_grid:
        # normalize
        arr = np.array(w, dtype=float)
        if arr.sum() == 0:
            continue
        norm = list(arr / arr.sum())
        df_ranked = weighted_scalar_baseline(preds, score_cols, norm)
        metrics = evaluate_baseline(preds, df_ranked, ks, user_col=user_col)
        # compute aggregate score: mean ndcg@K across available tasks for K=first ks
        k0 = ks[0]
        ndcgs = []
        for task, mets in metrics.items():
            key = f'ndcg@{k0}'
            if key in mets:
                ndcgs.append(mets[key])
        agg = float(np.mean(ndcgs)) if ndcgs else 0.0
        results[tuple(norm)] = {'metrics': metrics, 'agg_ndcg@k': agg}
        if best_metrics is None or agg > best_metrics:
            best_metrics = agg
            best_weights = tuple(norm)
    return results, best_weights
