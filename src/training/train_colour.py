"""
Train colour -> value models listed under `models:` in config.yaml.

    python -m src.training.train_colour
"""
import json
from src.config import load_config, resolve
from src.dataset import build_table, xy
from src.models import ColourToValue
from src.training.evaluate import grouped_cv

def main():
    cfg = load_config()
    tcfg = cfg["training"]
    table = build_table(resolve(tcfg["measurements_csv"]), resolve(tcfg["image_dir"]), cfg)
    out_dir = resolve(cfg["paths"]["trained_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    all_metrics = {}
    for name, spec in cfg["models"].items():
        X, y, g = xy(table, spec["chamber"], spec["target"])
        metrics = grouped_cv(X, y, g, tcfg["n_splits"], tcfg["n_estimators"], tcfg["seed"])
        model = ColourToValue(tcfg["n_estimators"], tcfg["seed"]).fit(X, y)  # final fit on all data
        model.save(out_dir / f"{name}.joblib")
        all_metrics[name] = {**metrics, "target": spec["target"], "chamber": spec["chamber"]}
        print(f"[{name}] {spec['target']}: grouped-CV MAE={metrics['mae']:.3f} "
              f"(p90 {metrics['p90_abs_error']:.3f}, n={metrics['n']})")
    (out_dir / "metrics.json").write_text(json.dumps(all_metrics, indent=2))

if __name__ == "__main__":
    main()
