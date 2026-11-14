"""Lighting correction and ROI helpers."""
import cv2
import numpy as np

def roi_slice(img: np.ndarray, roi: dict):
    """Convert a fractional ROI {x,y,w,h} into pixel slices."""
    H, W = img.shape[:2]
    x0, y0 = int(roi["x"] * W), int(roi["y"] * H)
    x1, y1 = int((roi["x"] + roi["w"]) * W), int((roi["y"] + roi["h"]) * H)
    if x1 <= x0 or y1 <= y0 or x0 < 0 or y0 < 0 or x1 > W or y1 > H:
        raise ValueError(f"ROI {roi} falls outside the image ({W}x{H})")
    return slice(y0, y1), slice(x0, x1)

def white_balance(img_bgr: np.ndarray, ref_roi: dict, target=(235, 235, 235), trim=0.1):
    """
    Per-channel gain so the white reference patch reads `target` (RGB).
    Returns (corrected_bgr, gains_rgb). Gains far from 1.0 mean poor lighting.
    """
    ys, xs = roi_slice(img_bgr, ref_roi)
    patch = img_bgr[ys, xs].reshape(-1, 3).astype(np.float64)
    ref_bgr = _trimmed_mean(patch, trim)
    target_bgr = np.array(target[::-1], dtype=np.float64)
    gains_bgr = target_bgr / np.maximum(ref_bgr, 1.0)
    corrected = np.clip(img_bgr.astype(np.float64) * gains_bgr, 0, 255).astype(np.uint8)
    return corrected, gains_bgr[::-1]

def _trimmed_mean(pixels: np.ndarray, trim: float) -> np.ndarray:
    """Mean after dropping the darkest/brightest `trim` fraction (by luminance)."""
    lum = pixels.mean(axis=1)
    lo, hi = np.quantile(lum, [trim, 1 - trim])
    keep = (lum >= lo) & (lum <= hi)
    return pixels[keep].mean(axis=0) if keep.any() else pixels.mean(axis=0)

def denoise(img_bgr: np.ndarray, ksize: int = 5) -> np.ndarray:
    return cv2.GaussianBlur(img_bgr, (ksize, ksize), 0)
