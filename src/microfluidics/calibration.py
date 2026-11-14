"""
Link what the channels show to the TOTAL liquid in the nappy.
Channels fill/saturate non-linearly, so we fit a monotone curve
(total filled channel length, mm) -> (true volume, mL) from lab measurements.
"""
import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression

class VolumeCalibration:
    def fit(self, total_length_mm, volume_ml):
        x, y = np.asarray(total_length_mm, float), np.asarray(volume_ml, float)
        self.iso = IsotonicRegression(out_of_bounds="clip", increasing=True).fit(x, y)
        self.x_range = (float(x.min()), float(x.max()))
        return self

    def predict(self, total_length_mm):
        return self.iso.predict(np.atleast_1d(np.asarray(total_length_mm, float)))

    def save(self, path): joblib.dump(self, path)

    @staticmethod
    def load(path): return joblib.load(path)
