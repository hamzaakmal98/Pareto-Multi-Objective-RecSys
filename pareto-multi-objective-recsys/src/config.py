from pathlib import Path
from typing import Any, Dict, Optional
import yaml


def load_config(path: Optional[str]) -> Dict[str, Any]:
    """Load YAML config file. If path is None, return empty dict."""
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with p.open('r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    return cfg

