import pandas as pd
from typing import Dict, Optional
from pathlib import Path


def load_csv(path: Path, parse_dates: Optional[list] = None) -> pd.DataFrame:
    """Load a CSV from `path` into a DataFrame.

    Args:
        path: Path to CSV file.
        parse_dates: list of columns to parse as datetime (optional).

    Returns:
        pd.DataFrame

    Raises:
        FileNotFoundError if path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    if parse_dates is None:
        parse_dates = ["timestamp"]
    try:
        return pd.read_csv(path, parse_dates=[c for c in parse_dates if c in pd.read_csv(path, nrows=0).columns], low_memory=False)
    except Exception:
        # Last resort: read without date parsing
        return pd.read_csv(path, low_memory=False)


def load_tables(config: Dict) -> Dict[str, pd.DataFrame]:
    """Load the set of expected tables from the `data` section of a config dict.

    Expected keys (optional):
      - interactions: path
      - users: path
      - videos: path (basic)
      - video_stats: path (aggregated stats)

    The returned dict maps table names to DataFrames (only for files that exist).
    """
    data_cfg = config.get("data", {})
    data_dir = Path(data_cfg.get("dir", "data"))
    tables = {}
    mapping = {
        "interactions": data_cfg.get("interactions", "interactions.csv"),
        "users": data_cfg.get("users", "users.csv"),
        "videos": data_cfg.get("videos", "videos.csv"),
        "video_stats": data_cfg.get("video_stats", "video_stats.csv"),
    }
    for name, rel in mapping.items():
        if rel is None:
            continue
        path = Path(rel)
        if not path.is_absolute():
            path = data_dir / rel
        if path.exists():
            try:
                tables[name] = load_csv(path)
            except Exception as e:
                # skip problematic files but surface the issue
                raise
    return tables

