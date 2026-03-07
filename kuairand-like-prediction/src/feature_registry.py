from typing import Dict, List
import re


class FeatureRegistry:
    """Simple registry to manage allowed and banned feature groups for leakage control.

    This class uses substring matching on column names to map them to groups. It's
    intentionally conservative: banned groups remove columns if any banned keyword
    matches the column name.
    """

    def __init__(self, allowed_groups: List[str] = None, banned_groups: List[str] = None):
        self.allowed_groups = allowed_groups or []
        self.banned_groups = banned_groups or []

    def is_banned(self, col: str) -> bool:
        col_l = col.lower()
        for k in self.banned_groups:
            if k.lower() in col_l:
                return True
        return False

    def is_allowed(self, col: str) -> bool:
        if not self.allowed_groups:
            return True
        col_l = col.lower()
        for k in self.allowed_groups:
            if k.lower() in col_l:
                return True
        return False

    def filter_columns(self, cols: List[str]) -> Dict[str, List[str]]:
        """Partition columns into allowed and banned lists.

        Returns a dict with keys: 'allowed', 'banned', 'unknown'.
        """
        allowed = []
        banned = []
        unknown = []
        for c in cols:
            if self.is_banned(c):
                banned.append(c)
            elif self.is_allowed(c):
                allowed.append(c)
            else:
                unknown.append(c)
        return {"allowed": allowed, "banned": banned, "unknown": unknown}


def infer_feature_groups_from_patterns(patterns: Dict[str, str], cols: List[str]) -> Dict[str, List[str]]:
    """Assign columns to groups by regex patterns.

    patterns: mapping group -> regex
    Returns mapping group -> matched cols
    """
    out = {g: [] for g in patterns}
    for c in cols:
        for g, p in patterns.items():
            if re.search(p, c):
                out[g].append(c)
    return out
