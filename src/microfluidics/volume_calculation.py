"""Geometry-based volume: V = A x L (mm^3 == uL; 1000 uL == 1 mL)."""

def channel_volume_ml(filled_length_mm: float, width_mm: float, depth_mm: float) -> float:
    return width_mm * depth_mm * filled_length_mm / 1000.0

def calculate_volume(filled_length_mm, width_mm=2.0, depth_mm=0.5, retention_factor=1.0) -> float:
    """Channel volume x a simple retention factor (1.0 = channel only). For a non-linear
    link to total nappy volume use VolumeCalibration instead."""
    return channel_volume_ml(filled_length_mm, width_mm, depth_mm) * retention_factor
