from pathlib import Path
import json
import csv
from typing import Union


def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: Union[str, Path], obj, indent: int = 2):
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)
    return p


def write_csv(path: Union[str, Path], rows, header=None):
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(header)
        for r in rows:
            writer.writerow(r)
    return p


def write_text(path: Union[str, Path], text: str):
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(text, encoding="utf-8")
    return p


def find_under_tree(root: Path, name: str):
    """Recursively find first directory named `name` under `root`.

    Returns Path or None.
    """
    for p in root.rglob(name):
        if p.is_dir():
            return p
    return None
