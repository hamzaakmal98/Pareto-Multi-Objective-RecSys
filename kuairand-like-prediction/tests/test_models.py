import pandas as pd
from src.models.logreg import train_logreg


def test_logreg_train():
    df = pd.DataFrame({
        "f1": [0.1, 0.2, 0.9, 0.8],
        "f2": [1, 2, 3, 4],
        "is_like": [0, 0, 1, 1],
    })
    X = df[["f1", "f2"]]
    y = df["is_like"]
    clf = train_logreg(X, y)
    preds = clf.predict_proba(X)[:, 1]
    assert preds.max() <= 1.0
