"""Augmentations applied at the PHOTO level (simulate bad lighting) before extraction."""
import numpy as np

def random_lighting(img_bgr: np.ndarray, rng: np.random.Generator, strength: float = 0.15):
    gains = 1 + rng.uniform(-strength, strength, size=3)
    return np.clip(img_bgr * gains, 0, 255).astype(np.uint8)
