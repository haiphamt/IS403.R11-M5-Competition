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

sys.path.insert(0, str(ROOT / "scripts"))
from run_experiment import LOCAL_MODELS, run_local  # noqa: E402  (same chunked, checkpointed local runner as the main runs)

TRAIN_END = 1885


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True, choices=global_models.MODELS + LOCAL_MODELS)
    ap.add_argument("--origins", nargs="+", type=int, default=[1829, 1857])
    ap.add_argument("--lgbm-params", default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "oof")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--max-series", type=int, default=None)
    ap.add_argument("--manifest", type=Path, default=None, help="series list (default: the 1,200-series sample)")
    ap.add_argument("--workers", type=int, default=4, help="processes for local models")
    ap.add_argument("--chunk", type=int, default=50, help="series per checkpoint for local models")
    a = ap.parse_args()
    if any(o + 28 > TRAIN_END for o in a.origins):
        raise SystemExit("every origin+28 must be <= 1885 (Train labels only)")
    a.out.mkdir(parents=True, exist_ok=True)
    params = json.loads(Path(a.lgbm_params).read_text()) if a.lgbm_params else None
    panel = optional.load_sample_panel(a.data_dir, manifest=a.manifest, max_series=a.max_series)
    ids = np.array(panel.series_ids)
    rows, timings = [], {}
    for origin in a.origins:
        days = np.arange(origin + 1, origin + 29)
        for model in [m for m in a.models if m in LOCAL_MODELS]:
            ckpt = a.out / "ckpt" / f"{model}_{origin}"
            ckpt.mkdir(parents=True, exist_ok=True)
            res, wall = run_local(model, panel, origin, a.workers, a.chunk, ckpt)
            assert [r["series_id"] for r in res] == list(ids)
            fc = np.vstack([r["forecast"] for r in res])
            timings[f"{model}@{origin}"] = {"fit_predict_sum_s": float(sum(r["fit_predict_s"] for r in res)), "wall_s": wall,
                                            "workers": a.workers, "n_fallback": int(sum(r["fallback"] for r in res))}
            rows.append(pd.DataFrame({"origin": origin, "model": model, "series_id": np.repeat(ids, 28), "d_num": np.tile(days, len(ids)),
                                      "y_pred": np.maximum(fc, 0).ravel(), "y_true": panel.y[:, days - 1].ravel()}))
            print(f"origin {origin} {model}: wall {wall:.1f}s", flush=True)
        glob_models = [m for m in a.models if m not in LOCAL_MODELS]
        if not glob_models:
            continue
        X, y, Xf, keys = global_models.build_table(panel, origin)
        for model in glob_models:
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
