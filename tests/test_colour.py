import numpy as np
from src.config import load_config
from src.preprocessing import extract_all
from src.preprocessing.image_cleaning import white_balance
from src.microfluidics import calculate_volume

CFG = load_config()

def make_img(tint=(1, 1, 1)):
    img = np.full((400, 600, 3), 200, np.uint8)
    L = CFG["layout"]
    def fill(roi, rgb):
        H, W = img.shape[:2]
        img[int(roi["y"]*H):int((roi["y"]+roi["h"])*H),
            int(roi["x"]*W):int((roi["x"]+roi["w"])*W)] = np.array(rgb)[::-1]
    fill(L["reference_white"], (235, 235, 235))
    for name, roi in L["chambers"].items():
        fill(roi, (110, 180, 105))
    return np.clip(img * np.array(tint)[::-1], 0, 255).astype(np.uint8)

def test_white_balance_removes_tint():
    res = extract_all(make_img(tint=(1.05, 0.9, 0.85)), CFG)
    r, g, b = [res["chambers"]["ph"][k] for k in "rgb"]
    assert abs(r - 110) < 6 and abs(g - 180) < 6 and abs(b - 105) < 6

def test_volume_geometry():
    # 2 mm x 0.5 mm x 50 mm = 50 mm^3 = 0.05 mL
    assert abs(calculate_volume(50, 2.0, 0.5) - 0.05) < 1e-9
