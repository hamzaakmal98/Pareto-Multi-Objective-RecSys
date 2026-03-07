from pathlib import Path
from typing import Dict, Tuple, List, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from .feature_registry import FeatureRegistry
from .utils import save_df, ensure_dir


def join_tables(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join available tables into a single interactions-level DataFrame.

    Expected minimal table: `interactions` which contains `user_id` and `video_id`.
    Optional tables (users, videos, video_stats) will be left-joined onto interactions.
    """
    if "interactions" not in tables:
        raise ValueError("interactions table is required in tables")
    df = tables["interactions"].copy()
    join_keys = {"users": "user_id", "videos": "video_id", "video_stats": "video_id"}
    for key, kcol in join_keys.items():
        if key in tables:
            df = df.merge(tables[key], how="left", on=kcol, suffixes=("", f"_{key}"))
    return df


def separate_column_types(df: pd.DataFrame, id_cols: List[str] = None, target_col: str = "is_like") -> Dict[str, List[str]]:
    """Return dict of column lists: ids, numeric, categorical, target.

    Numeric columns are those with pandas numeric dtype (excluding id and target).
    Categorical columns are object or category dtype excluding id and target.
    """
    id_cols = id_cols or ["user_id", "video_id"]
    cols = set(df.columns)
    ids = [c for c in id_cols if c in cols]
    target = [target_col] if target_col in cols else []
    numeric = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ids + target]
    categorical = [c for c in df.select_dtypes(include=[object, "category"]).columns if c not in ids + target]
    return {"ids": ids, "numeric": numeric, "categorical": categorical, "target": target}


def apply_leakage_policy(df: pd.DataFrame, registry: FeatureRegistry) -> Tuple[pd.DataFrame, List[str]]:
    """Remove banned columns using the provided registry and return removed list."""
    cols = df.columns.tolist()
    parts = registry.filter_columns(cols)
    banned = parts.get("banned", [])
    df = df.drop(columns=banned, errors="ignore")
    return df, banned


def impute_missing(df: pd.DataFrame, numeric_strategy: str = "zero") -> pd.DataFrame:
    """Impute missing values safely.

    numeric_strategy: 'zero' or 'median'
    """
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    if numeric_strategy == "median":
        for c in num_cols:
            df[c] = df[c].fillna(df[c].median())
    else:
        df[num_cols] = df[num_cols].fillna(0.0)
    cat_cols = df.select_dtypes(include=[object, "category"]).columns
    df[cat_cols] = df[cat_cols].fillna("__MISSING__")
    return df


def temporal_splits(df: pd.DataFrame, timestamp_col: Optional[str] = "timestamp", test_frac: float = 0.15, val_frac: float = 0.15, user_col: Optional[str] = "user_id") -> Tuple[pd.Index, pd.Index, pd.Index]:
    """Create time-aware train/val/test splits if timestamp exists.

    Returns indexes for train, val, test.
    If timestamp_col is missing, falls back to grouped/random split and returns indexes.
    """
    if timestamp_col and timestamp_col in df.columns:
        times = pd.to_datetime(df[timestamp_col])
        cutoff_val = times.quantile(1.0 - (test_frac + val_frac))
        cutoff_test = times.quantile(1.0 - test_frac)
        train_idx = df.index[times <= cutoff_val]
        val_idx = df.index[(times > cutoff_val) & (times <= cutoff_test)]
        test_idx = df.index[times > cutoff_test]
        return train_idx, val_idx, test_idx
    # fallback
    import warnings

    warnings.warn("timestamp column not found — falling back to grouped/random split")
    if user_col and user_col in df.columns:
        # perform grouped split by user to avoid leakage between same user
        users = df[user_col].drop_duplicates()
        u_train, u_rest = train_test_split(users, test_size=(test_frac + val_frac), random_state=42)
        u_val, u_test = train_test_split(u_rest, test_size=test_frac / (test_frac + val_frac), random_state=42)
        train_idx = df[df[user_col].isin(u_train)].index
        val_idx = df[df[user_col].isin(u_val)].index
        test_idx = df[df[user_col].isin(u_test)].index
        return train_idx, val_idx, test_idx
    # pure random
    idx = df.index
    train_idx, rest = train_test_split(idx, test_size=(test_frac + val_frac), random_state=42)
    val_idx, test_idx = train_test_split(rest, test_size=test_frac / (test_frac + val_frac), random_state=42)
    return pd.Index(train_idx), pd.Index(val_idx), pd.Index(test_idx)


def build_and_save_processed(tables: Dict[str, pd.DataFrame], config: Dict, processed_dir: str = "data/processed") -> Dict:
    """Main entry: join tables, apply leakage rules, impute, split and save artifacts.

    Returns a summary dict for printing.
    """
    ensure_dir(processed_dir)
    df = join_tables(tables)

    # Setup registry from config
    allowed = config.get("feature_groups", {}).get("allowed", [])
    banned = config.get("feature_groups", {}).get("banned", [])
    registry = FeatureRegistry(allowed_groups=allowed, banned_groups=banned)

    # Apply leakage policy
    df_clean, banned_cols = apply_leakage_policy(df, registry)

    # Impute
    df_clean = impute_missing(df_clean, numeric_strategy=config.get("impute", {}).get("numeric", "zero"))

    # Separate column types
    col_types = separate_column_types(df_clean, id_cols=config.get("id_cols", ["user_id", "video_id"]), target_col=config.get("target_col", "is_like"))

    # Save artifacts
    save_df(df_clean, Path(processed_dir) / "dataset_joined.csv")
    # Save typed column lists
    for k, cols in col_types.items():
        Path(processed_dir).joinpath(f"cols_{k}.txt").write_text("\n".join(cols))

    # Splits
    train_idx, val_idx, test_idx = temporal_splits(df_clean, timestamp_col=config.get("timestamp_col", "timestamp"), test_frac=config.get("test_frac", 0.15), val_frac=config.get("val_frac", 0.15), user_col=config.get("id_cols", ["user_id"])[:1][0])

    # Save indices
    pd.Series(list(train_idx)).to_csv(Path(processed_dir) / "train_idx.csv", index=False)
    pd.Series(list(val_idx)).to_csv(Path(processed_dir) / "val_idx.csv", index=False)
    pd.Series(list(test_idx)).to_csv(Path(processed_dir) / "test_idx.csv", index=False)

    # Prepare final X/y/meta and save
    target = config.get("target_col", "is_like")
    ids = col_types.get("ids", [])
    meta = df_clean[ids + [config.get("timestamp_col", "timestamp")]] if ids else pd.DataFrame()
    X = df_clean[[c for c in df_clean.columns if c not in ids + [target]]]
    y = df_clean[target] if target in df_clean.columns else pd.Series(dtype=int)
    save_df(X, Path(processed_dir) / "X.csv")
    save_df(y.to_frame(name=target), Path(processed_dir) / "y.csv")
    save_df(meta, Path(processed_dir) / "meta.csv")

    summary = {
        "loaded_tables": list(tables.keys()),
        "final_feature_count": len(X.columns),
        "banned_columns_removed": banned_cols,
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "test_size": len(test_idx),
    }
    return summary
