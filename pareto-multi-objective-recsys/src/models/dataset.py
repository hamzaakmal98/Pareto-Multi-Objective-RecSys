from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import torch
from torch.utils.data import Dataset


class InteractionDataset(Dataset):
    def __init__(self, path: Path, id_cols: List[str], cat_cols: List[str], num_cols: List[str], target_cols: List[str]):
        df = pd.read_parquet(path)
        self.df = df
        self.id_cols = id_cols
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.target_cols = target_cols

        # prepare categorical id encoding (simple factorize)
        self.cat_maps = {}
        for c in self.cat_cols:
            if c in df.columns:
                vals, cats = pd.factorize(df[c].fillna('__MISSING__'))
                self.cat_maps[c] = (vals.astype('int64'), list(cats))
        # numeric values
        self.numerics = None
        if self.num_cols:
            existing_num_cols = [c for c in self.num_cols if c in df.columns]
            self.numerics = df[existing_num_cols].fillna(0).astype(float).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cat_inputs = {}
        for c in self.cat_cols:
            if c in self.df.columns:
                val = row[c]
                # use factorize map
                map_vals, cats = self.cat_maps.get(c, (None, None))
                if map_vals is not None:
                    # pandas factorize returns arrays in same order; we stored whole column mapping
                    cat_idx = int(map_vals[idx])
                else:
                    cat_idx = 0
                cat_inputs[c] = torch.tensor(cat_idx, dtype=torch.long)

        numeric = torch.tensor(self.numerics[idx]) if self.numerics is not None else torch.empty(0)

        targets = {}
        for t in self.target_cols:
            if t in self.df.columns:
                targets[t] = torch.tensor(row[t] if pd.notna(row[t]) else 0, dtype=torch.float)
            else:
                targets[t] = None

        return cat_inputs, numeric, targets
