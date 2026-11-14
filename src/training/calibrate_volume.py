"""
Fit the total-length -> total-volume calibration curve and report detection accuracy.

measurements.csv needs: image, batch, volume_ml  (+ optional ch1_len_mm, ch2_len_mm, ... true lengths)

    python -m src.training.calibrate_volume
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from src.config import load_config, resolve
from src.preprocessing import load_image, prepare_image
from src.microfluidics import detect_all, VolumeCalibration

def main():
    cfg = load_config(); t = cfg["training"]
    meas = pd.read_csv(resolve(t["measurements_csv"]))
    rows, len_err = [], []
    for _, m in meas.iterrows():
        img, _, _ = prepare_image(load_image(Path(resolve(t["image_dir"])) / m["image"]), cfg)
        det = detect_all(img, cfg)
        rows.append(det["total_length_mm"])
        for name, c in det["channels"].items():
            col = f"{name}_len_mm"
            if col in meas.columns:
                len_err.append(abs(c["length_mm"] - m[col]))
    x, y, g = np.array(rows), meas["volume_ml"].values, meas["batch"].values
    errs = []
    for tr, te in GroupKFold(n_splits=min(t["n_splits"], len(np.unique(g)))).split(x, y, g):
        errs.append(np.abs(VolumeCalibration().fit(x[tr], y[tr]).predict(x[te]) - y[te]))
    errs = np.concatenate(errs)
    cal = VolumeCalibration().fit(x, y)
    out = resolve(cfg["paths"]["trained_dir"]); out.mkdir(parents=True, exist_ok=True)
    cal.save(out / "volume_calibration.joblib")
    metrics = {"volume_mae_ml_grouped_cv": float(errs.mean()), "n": int(len(y)),
               "calibrated_length_range_mm": cal.x_range}
    if len_err:
        metrics["channel_length_mae_mm"] = float(np.mean(len_err))
    (out / "calibration_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
