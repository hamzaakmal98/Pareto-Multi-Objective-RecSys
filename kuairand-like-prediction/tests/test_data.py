import pandas as pd
from src.dataset import build_dataset


def test_build_dataset():
    df = pd.DataFrame({
        "user_id": [1, 1, 2],
        "feat1": [0.1, 0.2, 0.3],
        "is_like": [0, 1, 0],
        "timestamp": [1, 2, 3],
    })
    X, y, meta = build_dataset(df, target="is_like", timestamp_col="timestamp")
    assert "feat1" in X.columns
    assert len(y) == 3
