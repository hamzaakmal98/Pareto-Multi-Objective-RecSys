import pandas as pd
from typing import Tuple, List, Optional


def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that likely leak the label (heuristic).

    We conservatively drop columns containing keywords like 'future', 'engagement',
    or exact matches to known target-like names (except the target itself).
    """
    drop_keywords = ["future", "engagement", "engaged", "next_", "label_"]
    to_drop = [c for c in df.columns if any(k in c.lower() for k in drop_keywords)]
    return df.drop(columns=[c for c in to_drop if c in df.columns], errors="ignore")


def build_dataset(df: pd.DataFrame, target: str = "is_like", timestamp_col: Optional[str] = "timestamp") -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return features X, labels y, and meta DataFrame (user_id/timestamp if present).

    - Drops leakage columns heuristically.
    - Selects numeric features excluding id, target and timestamp.
    """
    df = df.copy()
    if target not in df.columns:
        raise ValueError(f"target column '{target}' not found in dataframe")

    # Keep meta
    meta_cols = [c for c in ["user_id", timestamp_col] if c in df.columns]
    meta = df[meta_cols].copy() if meta_cols else pd.DataFrame(index=df.index)

    # Drop leakage
    df = drop_leakage_columns(df)

    # Prepare X and y
    y = df[target].astype(int)

    # Exclude ID-like, timestamp, and the target
    exclude = set([target, timestamp_col, "user_id"]) & set(df.columns)
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    features = [c for c in numeric if c not in exclude]

    X = df[features].fillna(0.0)
    return X, y, meta


def temporal_train_test_split(meta: pd.DataFrame, test_size: float = 0.2, timestamp_col: str = "timestamp"):
    """Return boolean mask for train/test split by time if timestamp exists."""
    if timestamp_col not in meta.columns:
        return None
    times = meta[timestamp_col]
    cutoff = times.quantile(1.0 - test_size)
    train_mask = times <= cutoff
    return train_mask, ~train_mask
