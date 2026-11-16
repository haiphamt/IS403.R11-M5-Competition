"""T02 step 1: out-of-fold backtests inside Train for the stacking base learners.

    python scripts/backtest_oof.py --models linear_regression lightgbm xgboost --origins 1829 1857 --out results/oof

For each origin the base model is fitted on days <= origin and forecasts origin+1..origin+28 (labels <= d_1885 only, so neither
Validation nor Test labels are used). Predictions are clipped at 0 like the final forecasts.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from m5 import global_models, optional  # noqa: E402

TRAIN_END = 1885


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True, choices=global_models.MODELS)
    ap.add_argument("--origins", nargs="+", type=int, default=[1829, 1857])
    ap.add_argument("--lgbm-params", default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "oof")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--max-series", type=int, default=None)
    a = ap.parse_args()
    if any(o + 28 > TRAIN_END for o in a.origins):
        raise SystemExit("every origin+28 must be <= 1885 (Train labels only)")
    a.out.mkdir(parents=True, exist_ok=True)
    params = json.loads(Path(a.lgbm_params).read_text()) if a.lgbm_params else None
    panel = optional.load_sample_panel(a.data_dir, max_series=a.max_series)
    ids = np.array(panel.series_ids)
    rows, timings = [], {}
    for origin in a.origins:
        X, y, Xf, keys = global_models.build_table(panel, origin)
        for model in a.models:
            pred, info, _ = global_models.fit_predict(model, X, y, Xf, n_jobs=a.n_jobs, lgbm_params=params)
            timings[f"{model}@{origin}"] = info
            sidx = pd.Series(range(len(ids)), index=ids).loc[keys["series_id"]].to_numpy()
            rows.append(pd.DataFrame({"origin": origin, "model": model, "series_id": keys["series_id"].to_numpy(),
                                      "d_num": keys["d_num"].to_numpy(), "y_pred": np.maximum(pred, 0),
                                      "y_true": panel.y[sidx, keys["d_num"].to_numpy() - 1]}))
            print(f"origin {origin} {model}: fit {info['fit_s']:.1f}s", flush=True)
    pd.concat(rows, ignore_index=True).to_parquet(a.out / f"oof_{'_'.join(a.models)}.parquet", index=False)
    (a.out / f"timings_{'_'.join(a.models)}.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
