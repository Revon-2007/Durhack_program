"""Build a tabular colour dataset from photos + measurements.csv."""
from pathlib import Path
import numpy as np
import pandas as pd
from src.preprocessing import load_image, extract_all, feature_vector

def build_table(measurements_csv, image_dir, cfg) -> pd.DataFrame:
    """
    measurements.csv needs an `image` column plus measured values
    (e.g. ph, volume_ml) and optionally `batch` (used for grouped CV).
    Returns one row per (image, chamber) with colour features + measurements.
    """
    meas = pd.read_csv(measurements_csv)
    rows = []
    for _, m in meas.iterrows():
        res = extract_all(load_image(Path(image_dir) / m["image"]), cfg)
        for chamber, colour in res["chambers"].items():
            rows.append({**m.to_dict(), "chamber": chamber,
                         **{k: colour[k] for k in colour}})
    return pd.DataFrame(rows)

def xy(table: pd.DataFrame, chamber: str, target: str):
    from src.preprocessing import FEATURE_NAMES
    t = table[(table.chamber == chamber)].dropna(subset=[target])
    groups = t["batch"].values if "batch" in t else np.arange(len(t))
    return t[FEATURE_NAMES].values, t[target].values, groups
