from typing import List, Tuple
import numpy as np
import pandas as pd


def is_dominated(point: np.ndarray, others: np.ndarray) -> bool:
    # point dominated if any other is >= in all dims and > in at least one
    ge = np.all(others >= point, axis=1)
    gt = np.any(others > point, axis=1)
    return np.any(ge & gt)


def pareto_frontier(points: np.ndarray) -> List[int]:
    # returns indices of non-dominated points
    n = points.shape[0]
    nondominated = []
    for i in range(n):
        others = np.delete(points, i, axis=0)
        if not is_dominated(points[i], others):
            nondominated.append(i)
    return nondominated


def extract_frontier(df: pd.DataFrame, score_cols: List[str], tie_break: str = None, secondary_score: str = None) -> pd.DataFrame:
    # compute frontier per user and return dataframe of frontier items
    rows = []
    for user, group in df.groupby('user_id'):
        pts = group[score_cols].fillna(0.0).values.astype(float)
        if pts.shape[0] == 0:
            continue
        idxs = pareto_frontier(pts)
        front = group.iloc[idxs].copy()
        # optional ordering: by secondary_score or tie_break
        if secondary_score and secondary_score in front.columns:
            front = front.sort_values(secondary_score, ascending=False)
        elif tie_break and tie_break in front.columns:
            front = front.sort_values(tie_break, ascending=False)
        rows.append(front)
    if rows:
        return pd.concat(rows, ignore_index=True)
    else:
        return df.iloc[0:0]
