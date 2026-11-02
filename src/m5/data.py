"""Read the five original M5 CSV files on local, Colab, or Kaggle runtimes."""

import argparse
from collections import defaultdict
import os
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "data.yaml"
SALES_IDS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
REQUIRED_COLUMNS = {
    "calendar": [
        "date", "wm_yr_wk", "d", "wday", "month", "year",
        "event_name_1", "event_type_1", "event_name_2", "event_type_2",
        "snap_CA", "snap_TX", "snap_WI",
    ],
    "sell_prices": ["store_id", "item_id", "wm_yr_wk", "sell_price"],
    "sales_train_validation": SALES_IDS + [f"d_{d}" for d in range(1, 1914)],
    "sales_train_evaluation": SALES_IDS + [f"d_{d}" for d in range(1, 1942)],
    "sample_submission": ["id"] + [f"F{d}" for d in range(1, 29)],
}


def _find_raw_dir(config, data_dir):
    filenames = list(config["files"].values())

    def missing(directory):
        return [name for name in filenames if not (directory / name).is_file()]

    explicit = data_dir if data_dir is not None else os.environ.get("M5_DATA_DIR")
    if explicit is not None:
        directory = Path(explicit).expanduser().resolve()
        absent = missing(directory)
        if absent:
            raise FileNotFoundError(f"{directory}: thieu file {', '.join(absent)}")
        return directory

    directory = PROJECT_ROOT / config["paths"]["raw"]
    if not missing(directory):
        return directory

    kaggle_input = Path(config["paths"]["kaggle_input"])
    candidates = sorted({
        path.parent for path in kaggle_input.rglob(config["files"]["calendar"])
        if not missing(path.parent)
    }) if kaggle_input.is_dir() else []
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(
            "Co nhieu bo du lieu M5. Hay truyen data_dir hoac dat M5_DATA_DIR: "
            + ", ".join(map(str, candidates))
        )
    raise FileNotFoundError(
        f"Khong tim du 5 file M5. Tai {directory} con thieu: "
        f"{', '.join(missing(directory))}. "
        "Hay truyen data_dir hoac dat bien moi truong M5_DATA_DIR."
    )


def load_raw_data(data_dir=None, *, nrows=None, config_path=DEFAULT_CONFIG):
    """Return five DataFrames keyed by the names in configs/data.yaml.

    data_dir overrides M5_DATA_DIR, then local data/raw and Kaggle are checked.
    nrows limits EVERY file for a quick I/O smoke check, not a model benchmark.
    The function does not split data, select a sample, or fill missing prices.
    """
    if nrows is not None and (not isinstance(nrows, int) or nrows < 0):
        raise ValueError("nrows phai la so nguyen khong am hoac None")
    with Path(config_path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or set(config.get("files", {})) != set(REQUIRED_COLUMNS):
        raise ValueError("data.yaml phai khai bao dung 5 bang M5 trong files")

    directory = _find_raw_dir(config, data_dir)
    tables = {}
    for name, filename in config["files"].items():
        path = directory / filename
        columns = pd.read_csv(path, nrows=0).columns
        absent = set(REQUIRED_COLUMNS[name]) - set(columns)
        if absent:
            preview = ", ".join(sorted(absent)[:8])
            raise ValueError(f"{filename}: thieu {len(absent)} cot bat buoc: {preview}")

        options = {}
        if name.startswith("sales_train_"):
            options["dtype"] = defaultdict(
                lambda: "int32", {column: "category" for column in SALES_IDS}
            )
        elif name == "sell_prices":
            options["dtype"] = {"store_id": "category", "item_id": "category"}
        elif name == "calendar":
            options["parse_dates"] = ["date"]
        tables[name] = pd.read_csv(path, nrows=nrows, **options)
    return tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, help="Thu muc chua 5 CSV goc")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--nrows", type=int, help="Doc toi da N dong MOI file de kiem tra nhanh")
    args = parser.parse_args()
    try:
        tables = load_raw_data(args.data_dir, nrows=args.nrows, config_path=args.config)
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.exit(2, f"Loi: {error}\n")
    if args.nrows is not None:
        print(f"SMOKE CHECK: toi da {args.nrows} dong moi file, khong phai toan bo du lieu.")
    for name, table in tables.items():
        print(f"{name:25s} {len(table):>10,} rows x {len(table.columns):,} columns")


if __name__ == "__main__":
    main()
