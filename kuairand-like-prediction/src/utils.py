import os
import joblib
import torch


def save_sklearn(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def save_torch(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_df(df, path: str):
    """Save DataFrame to CSV (creates parent dir)."""
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False)


def load_yaml(path: str):
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)

