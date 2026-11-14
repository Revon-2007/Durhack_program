"""
Colour <-> chemistry models.

ColourToValue : colour features -> measured value (pH, volume_ml). Used on real photos.
ValueToColour : measured value -> expected RGB. Used for the digital twin / QA
                (compare predicted colour with the photographed colour).
"""
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor

class ColourToValue:
    def __init__(self, n_estimators=300, seed=42):
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
        self.y_range = (0.0, 1.0)

    def fit(self, X, y):
        self.model.fit(X, y)
        self.y_range = (float(np.min(y)), float(np.max(y)))
        return self

    def predict(self, X):
        """Returns (prediction, confidence 0-1). Confidence falls as trees disagree."""
        X = np.atleast_2d(X)
        per_tree = np.stack([t.predict(X) for t in self.model.estimators_])
        mean, std = per_tree.mean(axis=0), per_tree.std(axis=0)
        span = max(self.y_range[1] - self.y_range[0], 1e-9)
        conf = np.clip(1 - 2 * std / span, 0, 1)
        return mean, conf

    def save(self, path): joblib.dump(self, path)

    @staticmethod
    def load(path): return joblib.load(path)

class ValueToColour:
    def __init__(self, n_estimators=200, seed=42):
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=-1)

    def fit(self, values, rgb):
        self.model.fit(np.asarray(values).reshape(len(values), -1), rgb)
        return self

    def predict(self, values):
        return self.model.predict(np.asarray(values).reshape(len(values), -1))

    def save(self, path): joblib.dump(self, path)

    @staticmethod
    def load(path): return joblib.load(path)
