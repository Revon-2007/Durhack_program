import numpy as np
from src.config import load_config
from src.preprocessing import prepare_image
from src.microfluidics import detect_all
from src.microfluidics.channel_fill import front_position

CFG = load_config()
DRY, WET = CFG["channels"]["dry_rgb"], CFG["channels"]["wet_rgb"]

def box(img, roi):
    H, W = img.shape[:2]
    return int(roi["x"]*W), int(roi["y"]*H), int((roi["x"]+roi["w"])*W), int((roi["y"]+roi["h"])*H)

def make_img(fracs, tint=(1, 1, 1), bubble=None):
    """fracs: {channel: filled fraction}. bubble: (channel, start_frac, end_frac) dry gap inside liquid."""
    img = np.full((400, 600, 3), 200, np.float64)
    x0, y0, x1, y1 = box(img, CFG["layout"]["reference_white"])
    img[y0:y1, x0:x1] = 235
    for name, ch in CFG["channels"]["items"].items():
        x0, y0, x1, y1 = box(img, ch)
        n = x1 - x0
        wet = (np.arange(n) + 0.5) < fracs[name] * n
        if bubble and bubble[0] == name:
            wet &= ~(((np.arange(n)) >= bubble[1] * n) & ((np.arange(n)) < bubble[2] * n))
        cols = np.where(wet[:, None], np.array(WET)[None, :], np.array(DRY)[None, :]).astype(float)
        img[y0:y1, x0:x1] = cols[None, :, ::-1]
    return np.clip(img * np.array(tint)[::-1], 0, 255).astype(np.uint8)

def run(img):
    corrected, _, _ = prepare_image(img, CFG)
    return detect_all(corrected, CFG)

def test_known_lengths():
    det = run(make_img({"ch1": 0.5, "ch2": 0.25, "ch3": 0.8}))
    assert abs(det["channels"]["ch1"]["length_mm"] - 15) < 1.0
    assert abs(det["channels"]["ch2"]["length_mm"] - 15) < 1.0
    assert abs(det["channels"]["ch3"]["length_mm"] - 72) < 1.5

def test_empty_and_full():
    det = run(make_img({"ch1": 0, "ch2": 0, "ch3": 0}))
    assert det["total_length_mm"] < 1.0
    det = run(make_img({"ch1": 1, "ch2": 1, "ch3": 1}))
    assert abs(det["total_length_mm"] - 180) < 2.0

def test_lighting_tint_is_corrected():
    det = run(make_img({"ch1": 0.5, "ch2": 0.5, "ch3": 0.5}, tint=(1.05, 0.9, 0.85)))
    assert abs(det["channels"]["ch3"]["length_mm"] - 45) < 2.0

def test_small_bubble_is_bridged():
    det = run(make_img({"ch1": 0, "ch2": 0, "ch3": 0.8}, bubble=("ch3", 0.40, 0.41)))
    assert abs(det["channels"]["ch3"]["length_mm"] - 72) < 2.0   # bubble does NOT stop the front

def test_front_position_subpixel():
    profile = np.array([1, 1, 1, 0.8, 0.2, 0, 0, 0], float)
    assert 3.5 < front_position(profile, 2) < 4.5
