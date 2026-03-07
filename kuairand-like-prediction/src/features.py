import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def simple_preprocess(X: pd.DataFrame, scaler: StandardScaler = None):
    X = X.copy()
    if scaler is None:
        scaler = StandardScaler()
        X.loc[:, :] = scaler.fit_transform(X.values)
        return X, scaler
    else:
        X.loc[:, :] = scaler.transform(X.values)
        return X, scaler
