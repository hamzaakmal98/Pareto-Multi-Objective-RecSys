from pathlib import Path
from typing import Dict, List, Optional
import re


class FeatureRegistry:
    """Infer and hold feature groupings from raw tables.

    Usage:
        reg = FeatureRegistry()
        reg.infer_from_dfs(dfs)
        reg.TARGET_COLUMNS, reg.CATEGORICAL_COLUMNS, ...
    """

    def __init__(self):
        self.TARGET_COLUMNS: List[str] = []
        self.BANNED_LEAKAGE_COLUMNS: List[str] = []
        self.ID_COLUMNS: List[str] = []
        self.CATEGORICAL_COLUMNS: List[str] = []
        self.NUMERIC_COLUMNS: List[str] = []
        self.TEMPORAL_COLUMNS: List[str] = []
        self.OPTIONAL_COLUMNS: List[str] = []
        self.detected: Dict[str, List[str]] = {}
        self.creator_task_enabled: bool = False

    @staticmethod
    def _find_cols_by_pattern(cols, patterns):
        found = []
        for p in patterns:
            rx = re.compile(p, flags=re.IGNORECASE)
            for c in cols:
                if rx.search(c):
                    found.append(c)
        return sorted(set(found))

    def infer_from_dfs(self, dfs: Dict[str, "pandas.DataFrame"]):
        # Collect column names from known tables
        all_cols = set()
        for name, df in dfs.items():
            all_cols.update(df.columns.tolist())

        cols = sorted(all_cols)
        self.detected["all"] = cols

        # ID columns heuristics
        id_patterns = [r"^user_id$", r"^video_id$", r"^impression_id$", r"^creator_id$", r"session_id"]
        self.ID_COLUMNS = self._find_cols_by_pattern(cols, id_patterns)

        # Temporal columns heuristics
        time_patterns = [r"timestamp", r"ts$", r"time", r"date"]
        self.TEMPORAL_COLUMNS = self._find_cols_by_pattern(cols, time_patterns)

        # Targets: like, long view, creator interest proxies
        like_cols = self._find_cols_by_pattern(cols, [r"\blike(s)?\b", r"liked", r"is_like"])
        long_view_cols = self._find_cols_by_pattern(cols, [r"long_view", r"longview", r"long_watch", r"view_time_ratio"])
        follow_cols = self._find_cols_by_pattern(cols, [r"follow", r"is_follow", r"followed", r"creator_follow"])

        # Assign primary target column names (prefer canonical)
        if like_cols:
            self.TARGET_COLUMNS.append(like_cols[0])
        if long_view_cols:
            self.TARGET_COLUMNS.append(long_view_cols[0])

        # Creator interest: prefer explicit follow; otherwise will try to synthesize
        if follow_cols:
            self.TARGET_COLUMNS.append(follow_cols[0])
            self.creator_task_enabled = True
        else:
            # Creator task optional — may be synthesized later
            self.creator_task_enabled = False

        # Banned leakage: current-row engagement outcomes — include detected target-like columns
        self.BANNED_LEAKAGE_COLUMNS = list(set(self.TARGET_COLUMNS))

        # Numeric vs categorical heuristics
        # Simple heuristic: names containing rate/count/duration/avg are numeric
        numeric_patterns = [r"count", r"rate", r"duration", r"avg", r"score", r"time", r"impressions", r"views"]
        numeric = self._find_cols_by_pattern(cols, numeric_patterns)

        # Categorical: category, device, country, placement, version, bucket
        cat_patterns = [r"category", r"device", r"country", r"placement", r"app_version", r"bucket", r"genre"]
        categorical = self._find_cols_by_pattern(cols, cat_patterns)

        # Anything left that looks like an id but not in ID_COLUMNS consider categorical
        for c in cols:
            if c.endswith("_id") and c not in self.ID_COLUMNS:
                categorical.append(c)

        self.NUMERIC_COLUMNS = sorted(set(numeric) - set(self.BANNED_LEAKAGE_COLUMNS))
        self.CATEGORICAL_COLUMNS = sorted(set(categorical) - set(self.BANNED_LEAKAGE_COLUMNS))

        # Optional columns are those not assigned yet
        assigned = set(self.ID_COLUMNS + self.TEMPORAL_COLUMNS + self.NUMERIC_COLUMNS + self.CATEGORICAL_COLUMNS + self.BANNED_LEAKAGE_COLUMNS)
        self.OPTIONAL_COLUMNS = sorted(set(cols) - assigned)

        return self

    def to_dict(self):
        return {
            "TARGET_COLUMNS": self.TARGET_COLUMNS,
            "BANNED_LEAKAGE_COLUMNS": self.BANNED_LEAKAGE_COLUMNS,
            "ID_COLUMNS": self.ID_COLUMNS,
            "CATEGORICAL_COLUMNS": self.CATEGORICAL_COLUMNS,
            "NUMERIC_COLUMNS": self.NUMERIC_COLUMNS,
            "TEMPORAL_COLUMNS": self.TEMPORAL_COLUMNS,
            "OPTIONAL_COLUMNS": self.OPTIONAL_COLUMNS,
            "creator_task_enabled": self.creator_task_enabled,
        }
