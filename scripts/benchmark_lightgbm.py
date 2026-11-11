"""C03: benchmark LightGBM on 10 original M5 series using Train history only, never Test."""

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter_ns

import lightgbm
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from m5.data import DEFAULT_CONFIG, load_raw_data
from m5.lightgbm_forecast import lightgbm_forecast


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--n-series", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "tables")
    args = parser.parse_args()
    if args.n_series < 1 or args.repeats < 1:
        parser.error("n-series and repeats must be positive")
    with DEFAULT_CONFIG.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    with (DEFAULT_CONFIG.parent / "sample.yaml").open(encoding="utf-8") as handle:
        sample = yaml.safe_load(handle)

    # Limit rows, but retain the full daily history of each selected series.
    raw = load_raw_data(args.data_dir, nrows=args.n_series)
    sales = raw["sales_train_validation"]
    train_start, train_end = config["splits"]["train"]
    history = sales[[f"d_{d}" for d in range(train_start, train_end + 1)]].to_numpy(dtype=float)
    horizon = config["forecast_horizon"]

    lightgbm_forecast(history[0], horizon)  # warm-up outside timing
    rows = []
    for series_id, values in zip(sales["id"], history):
        start = perf_counter_ns()
        for _ in range(args.repeats):
            prediction = lightgbm_forecast(values, horizon)
        seconds = (perf_counter_ns() - start) / 1e9 / args.repeats
        if len(prediction) != horizon or not np.isfinite(prediction).all():
            raise RuntimeError(f"Invalid LightGBM forecast for {series_id}")
        rows.append({"series_id": str(series_id), "model": "LightGBM", "history_days": len(values),
                     "horizon": horizon, "repeats": args.repeats, "seconds_per_series": seconds})

    runtime = ("kaggle" if "KAGGLE_KERNEL_RUN_TYPE" in os.environ else
               "colab" if "COLAB_RELEASE_TAG" in os.environ else "local")
    table = pd.DataFrame(rows)
    mean_seconds = float(table["seconds_per_series"].mean())
    summary = {
        "task": "C03", "runtime": runtime,
        "recorded_at": datetime.now(timezone(timedelta(hours=7))).isoformat(),
        "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
        "lightgbm": lightgbm.__version__,
        "operating_system": platform.system(), "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(), "n_series": len(table), "repeats": args.repeats,
        "series_selection": "first rows of sales_train_validation.csv; runtime pilot, not D02 sample",
        "train_range": [train_start, train_end], "horizon": horizon,
        "model": "LGBMRegressor default params, random_state=42, one model per series",
        "features": "lags 28/35/42; rolling mean and std over 7 and 28 days shifted by 28",
        "per_model": {"LightGBM": {
            "mean_seconds_per_series": mean_seconds,
            "max_seconds_per_series": float(table["seconds_per_series"].max()),
            "estimated_sample_seconds_one_run": mean_seconds * sample["n_series"],
            # The plan fits each series twice: Validation (M01/M02), then Test (F03).
            "estimated_sample_seconds_validation_plus_test": 2 * mean_seconds * sample["n_series"],
        }},
        "configured_sample_n_series": sample["n_series"],
        "timing_scope": "one fit plus forecast call per series; excludes CSV loading and file writes",
        "sample_decision": "unchanged; C05 decides the sample size from all models' timings",
        "test_set_used": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / f"c03_lightgbm_runtime_{runtime}.csv", index=False)
    (args.output_dir / f"c03_lightgbm_summary_{runtime}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(table.to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
