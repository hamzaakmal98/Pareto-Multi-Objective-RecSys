from pathlib import Path
import pandas as pd


def top_n_per_objective(pred_df: pd.DataFrame, user_col: str, item_col: str, score_col: str, n: int) -> pd.DataFrame:
    # returns DataFrame of top-n per user for given score_col
    pred_df = pred_df[[user_col, item_col, score_col]].copy()
    pred_df['_rank'] = pred_df.groupby(user_col)[score_col].rank(method='first', ascending=False)
    out = pred_df[pred_df['_rank'] <= n].drop(columns=['_rank'])
    return out


def union_candidates(pred_df: pd.DataFrame, user_col: str, item_col: str, score_cols: list, top_n: int) -> pd.DataFrame:
    # collect top-N per objective and union
    pieces = []
    for sc in score_cols:
        pieces.append(top_n_per_objective(pred_df, user_col, item_col, sc, top_n))
    union_df = pd.concat(pieces, ignore_index=True)
    # deduplicate per user-item keeping max across scores where possible
    agg_cols = {c: 'max' for c in score_cols if c in union_df.columns}
    grouped = union_df.groupby([user_col, item_col]).agg(agg_cols).reset_index()
    return grouped
