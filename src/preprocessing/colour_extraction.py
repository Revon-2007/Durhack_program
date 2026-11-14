"""Turn a photo of the indicator patch into numeric colour features."""
import cv2
import numpy as np
from .image_cleaning import roi_slice, white_balance, _trimmed_mean, denoise

FEATURE_NAMES = ["r", "g", "b", "h_sin", "h_cos", "s", "v", "L", "a", "b_lab"]

def roi_colour(img_bgr: np.ndarray, roi: dict, trim: float = 0.1) -> dict:
    """Robust colour of one ROI: RGB, HSV and CIELAB (lighting-robust features)."""
    ys, xs = roi_slice(img_bgr, roi)
    patch = img_bgr[ys, xs]
    mean_bgr = _trimmed_mean(patch.reshape(-1, 3).astype(np.float64), trim)
    px = np.uint8(np.clip(mean_bgr, 0, 255)).reshape(1, 1, 3)
    h, s, v = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0, 0].astype(float)
    L, a, b = cv2.cvtColor(px, cv2.COLOR_BGR2LAB)[0, 0].astype(float)
    rgb = mean_bgr[::-1]
    hue_rad = np.deg2rad(h * 2)  # OpenCV hue is 0-179
    return {
        "r": rgb[0], "g": rgb[1], "b": rgb[2],
        "h_sin": np.sin(hue_rad), "h_cos": np.cos(hue_rad),  # hue is circular
        "s": s, "v": v, "L": L, "a": a, "b_lab": b,
        "std": float(patch.reshape(-1, 3).std(axis=0).mean()),  # high std => glare/uneven fill
    }

def prepare_image(img_bgr: np.ndarray, cfg: dict):
    """Denoise + white balance. Returns (corrected_bgr, gains_rgb, warnings)."""
    ccfg, layout = cfg["colour"], cfg["layout"]
    img = denoise(img_bgr)
    img, gains = white_balance(img, layout["reference_white"],
                               target=ccfg["reference_white_target"], trim=ccfg["roi_trim_fraction"])
    warnings = []
    if gains.max() > ccfg["max_wb_gain"] or gains.min() < 1 / ccfg["max_wb_gain"]:
        warnings.append("Strong lighting correction applied - retake photo in better light.")
    return img, gains, warnings

def extract_all(img_bgr: np.ndarray, cfg: dict) -> dict:
    """
    Full colour pipeline for one photo: prepare_image -> per-chamber colour.
    Returns {"chambers", "wb_gains", "warnings", "image" (corrected BGR)}.
    """
    img, gains, warnings = prepare_image(img_bgr, cfg)
    trim = cfg["colour"]["roi_trim_fraction"]
    chambers = {name: roi_colour(img, roi, trim) for name, roi in cfg["layout"]["chambers"].items()}
    for name, c in chambers.items():
        if c["std"] > 25:
            warnings.append(f"Chamber '{name}' colour is uneven (glare or partial fill).")
    return {"chambers": chambers, "wb_gains": gains.tolist(), "warnings": warnings, "image": img}

def feature_vector(colour: dict) -> np.ndarray:
    return np.array([colour[k] for k in FEATURE_NAMES], dtype=np.float64)
