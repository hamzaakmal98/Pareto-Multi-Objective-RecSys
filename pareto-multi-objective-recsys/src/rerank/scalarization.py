from typing import List, Dict
import numpy as np
import pandas as pd


def min_max_normalize(arr: np.ndarray) -> np.ndarray:
    mins = np.nanmin(arr, axis=0)
    maxs = np.nanmax(arr, axis=0)
    denom = (maxs - mins)
    denom[denom == 0] = 1.0
    return (arr - mins) / denom


def scalarize_df(df: pd.DataFrame, score_cols: List[str], weights: List[float], out_col: str = 'scalar_score') -> pd.DataFrame:
    arr = df[score_cols].fillna(0.0).values.astype(float)
    norm = min_max_normalize(arr)
    scalar = norm.dot(np.array(weights[:norm.shape[1]]))
    df[out_col] = scalar
    return df
