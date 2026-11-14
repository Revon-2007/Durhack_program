"""
Generate SYNTHETIC patch photos + measurements.csv so the pipeline can be tested
without a physical prototype. Colour curves and fill behaviour are ILLUSTRATIVE.
Replace with real photos + lab measurements before drawing any conclusion.

Channels of different length saturate at different nappy volumes (like a volumetric bar chart).

    python scripts/make_synthetic_data.py --n 200
"""
import argparse, sys
from pathlib import Path
import cv2, numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config, resolve

PH_PTS = [5.5, 6.5, 7.0, 8.0, 8.5]
PH_RGB = np.array([[235,190,65],[190,205,85],[110,180,105],[65,145,160],[55,130,170]])
VOL_PTS = [0, 100]
VOL_RGB = np.array([[235,235,225],[70,120,190]])
CAPACITY_ML = {"ch1": 33.3, "ch2": 66.7, "ch3": 100.0}   # nappy volume at which each channel is full

def interp_rgb(x, xp, rgbs):
    return np.array([np.interp(x, xp, rgbs[:, c]) for c in range(3)])

def px_box(img, roi):
    H, W = img.shape[:2]
    return (int(roi["x"]*W), int(roi["y"]*H), int((roi["x"]+roi["w"])*W), int((roi["y"]+roi["h"])*H))

def fill(img, roi, rgb):
    x0, y0, x1, y1 = px_box(img, roi)
    img[y0:y1, x0:x1] = np.array(rgb)[::-1]

def draw_channel(img, roi, frac, dry, wet, rng):
    x0, y0, x1, y1 = px_box(img, roi)
    n = x1 - x0
    front = frac * n
    s = 1 / (1 + np.exp(-(front - (np.arange(n) + 0.5)) / 1.5))   # soft meniscus
    dry, wet = np.array(dry, float), np.array(wet, float)
    cols = dry[None, :] + s[:, None] * (wet - dry)[None, :]
    img[y0:y1, x0:x1] = cols[None, :, ::-1].repeat(y1 - y0, 0)   # RGB -> BGR

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
    cfg = load_config(); L = cfg["layout"]; C = cfg["channels"]; rng = np.random.default_rng(a.seed)
    img_dir = resolve(cfg["training"]["image_dir"]); img_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(a.n):
        ph, vol = rng.uniform(5.5, 8.5), rng.uniform(0, 100)
        img = np.full((400, 600, 3), 200, np.float64)
        fill(img, L["reference_white"], (235, 235, 235))
        fill(img, L["chambers"]["ph"], interp_rgb(ph, PH_PTS, PH_RGB))
        fill(img, L["chambers"]["ammonia"], (200, 200, 120))
        fill(img, L["chambers"]["moisture"], interp_rgb(vol, VOL_PTS, VOL_RGB))
        truth = {}
        for name, ch in C["items"].items():
            frac = float(np.clip(vol / CAPACITY_ML[name] + rng.normal(0, 0.01), 0, 1))
            draw_channel(img, ch, frac, C["dry_rgb"], C["wet_rgb"], rng)
            truth[f"{name}_len_mm"] = round(frac * ch["length_mm"], 2)
        gains = 1 + rng.uniform(-0.2, 0.2, 3)            # simulated lighting tint
        img = np.clip(img * gains * rng.uniform(0.85, 1.05), 0, 255)
        img = np.clip(img + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
        name = f"synthetic_{i:04d}.png"
        cv2.imwrite(str(img_dir / name), img)
        rows.append({"image": name, "batch": i // 10, "ph": round(ph, 2),
                     "volume_ml": round(vol, 1), **truth, "synthetic": True})
    pd.DataFrame(rows).to_csv(resolve(cfg["training"]["measurements_csv"]), index=False)
    print(f"Wrote {a.n} synthetic images to {img_dir}")

if __name__ == "__main__":
    main()
