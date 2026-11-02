"""C04: benchmark 10 original M5 series using Train history only, never Test."""

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter_ns

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from m5.data import DEFAULT_CONFIG, load_raw_data
from m5.seasonal_naive import seasonal_naive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--n-series", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=1000)
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
    horizon, season_length = config["forecast_horizon"], 7
    rows = []
    for series_id, values in zip(sales["id"], history):
        seasonal_naive(values, horizon, season_length)  # warm-up outside timing
        start = perf_counter_ns()
        for _ in range(args.repeats):
            prediction = seasonal_naive(values, horizon, season_length)
        seconds = (perf_counter_ns() - start) / 1e9 / args.repeats
        if len(prediction) != horizon or not np.isfinite(prediction).all():
            raise RuntimeError(f"Invalid forecast for {series_id}")
        rows.append({"series_id": str(series_id), "history_days": len(values),
                     "horizon": horizon, "season_length": season_length,
                     "repeats": args.repeats, "seconds_per_forecast": seconds})

    runtime = ("kaggle" if "KAGGLE_KERNEL_RUN_TYPE" in os.environ else
               "colab" if "COLAB_RELEASE_TAG" in os.environ else "local")
    table = pd.DataFrame(rows)
    mean_seconds = float(table["seconds_per_forecast"].mean())
    summary = {
        "task": "C04", "runtime": runtime,
        "recorded_at": datetime.now(timezone(timedelta(hours=7))).isoformat(),
        "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
        "operating_system": platform.system(), "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(), "n_series": len(table), "repeats": args.repeats,
        "series_selection": "first rows of sales_train_validation.csv; runtime pilot, not D02 sample",
        "train_range": [train_start, train_end], "horizon": horizon, "season_length": season_length,
        "mean_seconds_per_forecast": mean_seconds,
        "configured_sample_n_series": sample["n_series"],
        "estimated_sample_forecast_seconds": mean_seconds * sample["n_series"],
        "timing_scope": "forecast call including input validation; excludes CSV loading and file writes",
        "sample_decision": "unchanged; C05 still requires timing evidence from other models",
        "test_set_used": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / f"c04_seasonal_naive_runtime_{runtime}.csv", index=False)
    (args.output_dir / f"c04_seasonal_naive_summary_{runtime}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(table.to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
