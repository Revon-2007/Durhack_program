"""Load config.yaml from the project root."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]

def load_config(path=None) -> dict:
    with open(path or ROOT / "config.yaml") as f:
        return yaml.safe_load(f)

def resolve(rel: str) -> Path:
    return ROOT / rel
