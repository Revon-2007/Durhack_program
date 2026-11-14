import numpy as np
from sklearn.model_selection import GroupKFold
from src.models import ColourToValue

def grouped_cv(X, y, groups, n_splits=5, n_estimators=300, seed=42) -> dict:
    """Cross-validate by batch so near-identical photos never sit in both train and test."""
    n_splits = min(n_splits, len(np.unique(groups)))
    errs = []
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        m = ColourToValue(n_estimators, seed).fit(X[tr], y[tr])
        pred, _ = m.predict(X[te])
        errs.append(np.abs(pred - y[te]))
    err = np.concatenate(errs)
    return {"mae": float(err.mean()), "p90_abs_error": float(np.quantile(err, 0.9)),
            "n": int(len(err)), "folds": n_splits}
