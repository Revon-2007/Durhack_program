"""
Detect how far liquid has advanced along each microfluidic channel.

Method (per channel ROI, after white balance):
 1. Convert pixels to CIELAB; project each onto the dry->wet colour axis: t=0 dry, t=1 wet.
 2. Median over the channel's width -> a 1-D profile along the channel (inlet on the left).
 3. Smooth, threshold at t=0.5, bridge small gaps (bubbles), find the liquid front,
    refine to sub-pixel by interpolating the 0.5 crossing.
 4. length_mm = front fraction x channel length;  V = width x depth x length.
"""
import cv2
import numpy as np
from src.preprocessing.image_cleaning import roi_slice
from .volume_calculation import channel_volume_ml

def _lab(rgb) -> np.ndarray:
    px = np.uint8([[list(rgb)[::-1]]])
    return cv2.cvtColor(px, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float64)

def _smooth(x: np.ndarray, k: int) -> np.ndarray:
    k = max(1, k | 1)  # force odd
    if k == 1 or len(x) < k:
        return x
    xp = np.pad(x, (k // 2, k // 2), mode="edge")
    return np.convolve(xp, np.ones(k) / k, mode="valid")

def fill_profile(img_bgr, roi, dry_rgb, wet_rgb, smooth_px=9):
    """Return (profile along channel in [~0,~1], dry-wet contrast in Lab units)."""
    ys, xs = roi_slice(img_bgr, roi)
    lab = cv2.cvtColor(img_bgr[ys, xs], cv2.COLOR_BGR2LAB).astype(np.float64)
    dry, wet = _lab(dry_rgb), _lab(wet_rgb)
    axis = wet - dry
    denom = float(axis @ axis)
    t = ((lab - dry) @ axis) / denom
    return _smooth(np.median(t, axis=0), smooth_px), float(np.sqrt(denom))

def front_position(profile: np.ndarray, gap_tol_px: int) -> float:
    """Liquid front in pixels from the inlet (0 = empty, len(profile) = full)."""
    filled = profile > 0.5
    last, gap = -1, 0
    for i, f in enumerate(filled):
        if f:
            last, gap = i, 0
        else:
            gap += 1
            if gap > gap_tol_px:
                break
    n = len(profile)
    if last < 0:
        return 0.0
    if last == n - 1:
        return float(n)
    p0, p1 = profile[last], profile[last + 1]
    frac = (p0 - 0.5) / (p0 - p1) if p0 > p1 else 0.0
    return float(np.clip(last + 0.5 + frac, 0, n))

def detect_channel(img_bgr, ch: dict, fcfg: dict, defaults: dict) -> dict:
    profile, contrast = fill_profile(img_bgr, ch, fcfg["dry_rgb"], fcfg["wet_rgb"], fcfg["smooth_px"])
    n = len(profile)
    front = front_position(profile, max(1, int(fcfg["gap_tol_frac"] * n)))
    fraction = front / n
    length_mm = fraction * ch["length_mm"]
    w = ch.get("width_mm", defaults["channel_width_mm"])
    d = ch.get("depth_mm", defaults["channel_depth_mm"])
    decisive = float(np.mean(np.abs(profile - 0.5) > 0.25))  # fraction of columns clearly wet/dry
    return {"fraction": round(fraction, 4), "length_mm": round(length_mm, 2),
            "volume_ml": channel_volume_ml(length_mm, w, d), "confidence": round(decisive, 2),
            "front_px": front, "n_px": n, "contrast": contrast}

def detect_all(img_bgr, cfg: dict) -> dict:
    """Run detection on every channel of a (white-balanced) image."""
    fcfg, defaults = cfg["channels"], cfg["microfluidics"]
    channels, warnings = {}, []
    for name, ch in fcfg["items"].items():
        r = detect_channel(img_bgr, ch, fcfg, defaults)
        channels[name] = r
        if r["contrast"] < fcfg["min_contrast"]:
            warnings.append(f"{name}: dry and wet colours too similar to detect reliably.")
        if r["confidence"] < 0.7:
            warnings.append(f"{name}: unclear liquid front (confidence {r['confidence']:.2f}).")
    return {"channels": channels,
            "total_length_mm": round(sum(c["length_mm"] for c in channels.values()), 2),
            "total_channel_volume_ml": sum(c["volume_ml"] for c in channels.values()),
            "warnings": warnings}

def draw_overlay(img_bgr, cfg, detection) -> np.ndarray:
    """Annotated copy: green box = channel ROI, red line = detected front."""
    out = img_bgr.copy()
    H, W = out.shape[:2]
    for name, ch in cfg["channels"]["items"].items():
        x0, y0 = int(ch["x"] * W), int(ch["y"] * H)
        x1, y1 = int((ch["x"] + ch["w"]) * W), int((ch["y"] + ch["h"]) * H)
        r = detection["channels"][name]
        fx = int(x0 + (x1 - x0) * r["fraction"])
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 200, 0), 1)
        cv2.line(out, (fx, y0 - 3), (fx, y1 + 3), (0, 0, 255), 2)
        cv2.putText(out, f"{name}: {r['length_mm']:.1f} mm", (x0, y0 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA)
    return out
