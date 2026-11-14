"""
Full pipeline on one photo: colour chemistry + filled-channel volume + cross-check.

    python -m src.inference.predict_image path/to/photo.jpg
"""
import json, sys
from datetime import datetime, timezone
from pathlib import Path
import cv2
from src.config import load_config, resolve
from src.preprocessing import load_image, extract_all, feature_vector
from src.models import ColourToValue
from src.microfluidics import detect_all, draw_overlay, VolumeCalibration

def predict(image_path, cfg=None, return_overlay=False):
    cfg = cfg or load_config()
    res = extract_all(load_image(image_path), cfg)
    corrected = res["image"]
    trained = resolve(cfg["paths"]["trained_dir"])
    out = {
        "image": str(image_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "colour_readings": {n: [round(c[k], 1) for k in ("r", "g", "b")]
                            for n, c in res["chambers"].items()},
        "wb_gains": [round(g, 3) for g in res["wb_gains"]],
        "estimates": {}, "warnings": list(res["warnings"]),
    }

    # --- colour -> value models (pH, colour-based volume) ---
    for name, spec in cfg["models"].items():
        mpath = trained / f"{name}.joblib"
        if not mpath.exists():
            out["warnings"].append(f"No trained model '{name}' - run src.training.train_colour")
            continue
        model = ColourToValue.load(mpath)
        val, conf = model.predict(feature_vector(res["chambers"][spec["chamber"]]))
        out["estimates"][spec["target"]] = {"value": round(float(val[0]), 2),
                                            "confidence": round(float(conf[0]), 2)}
        if not (model.y_range[0] <= val[0] <= model.y_range[1]):
            out["warnings"].append(f"{spec['target']} outside calibrated range.")

    # --- filled-channel length -> volume ---
    det = detect_all(corrected, cfg)
    out["channels"] = {n: {k: (round(v, 5) if k == "volume_ml" else v)
                           for k, v in c.items() if k in ("fraction", "length_mm", "volume_ml", "confidence")}
                       for n, c in det["channels"].items()}
    out["total_filled_length_mm"] = det["total_length_mm"]
    out["total_channel_volume_ml"] = round(det["total_channel_volume_ml"], 5)
    out["warnings"] += det["warnings"]

    cpath = trained / "volume_calibration.joblib"
    if cpath.exists():
        cal = VolumeCalibration.load(cpath)
        v = float(cal.predict(det["total_length_mm"])[0])
        out["volume_from_channels_ml"] = round(v, 2)
        lo, hi = cal.x_range
        if not (lo <= det["total_length_mm"] <= hi):
            out["warnings"].append("Channel fill outside calibrated range.")
        if "volume_ml" in out["estimates"]:   # cross-check the two independent estimates
            diff = abs(v - out["estimates"]["volume_ml"]["value"])
            out["volume_crosscheck_diff_ml"] = round(diff, 2)
            if diff > cfg["channels"]["crosscheck_tol_ml"]:
                out["warnings"].append(f"Colour and channel volume estimates disagree by {diff:.1f} mL.")
    else:
        out["warnings"].append("No volume calibration - run src.training.calibrate_volume")

    if return_overlay:
        return out, draw_overlay(corrected, cfg, det)
    return out

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python -m src.inference.predict_image <photo>")
    result, overlay = predict(sys.argv[1], return_overlay=True)
    print(json.dumps(result, indent=2))
    d = resolve(load_config()["paths"]["prediction_dir"]); d.mkdir(parents=True, exist_ok=True)
    stem = Path(sys.argv[1]).stem
    (d / f"{stem}.json").write_text(json.dumps(result, indent=2))
    cv2.imwrite(str(d / f"{stem}_overlay.png"), overlay)
    print(f"Saved {d / (stem + '_overlay.png')}")
