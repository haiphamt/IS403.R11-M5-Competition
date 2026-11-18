"""Manifest of ALL 30,490 product-store series with their Train demand class (full-data runs).

    python scripts/prepare_full_manifest.py            # writes data/full_series_ids_with_class.csv

Same class rule as the 1,200-series sample (scripts/prepare_sample_split.py): share of zero-sale days on d_1..d_1885 only,
bins <=0.5 / 0.5-0.8 / >0.8. The file is both the --manifest of run_experiment.py and the --classes of analyze_results.py.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from m5.data import load_raw_data  # noqa: E402
from m5.sample import manifest_hash  # noqa: E402
from m5.split import split_wide, to_wide  # noqa: E402

BINS, LABELS = [-0.001, 0.5, 0.8, 1.0], ["low_zero(<=0.5)", "mid_zero(0.5-0.8]", "high_zero(>0.8)"]


def main():
    sales = load_raw_data(table_names=("sales_train_evaluation",))["sales_train_evaluation"]
    zr = (split_wide(to_wide(sales))["train"] == 0).mean(axis=1)
    meta = sales.assign(series_id=to_wide(sales).index)[["series_id", "store_id", "cat_id"]].set_index("series_id")
    out = meta.assign(train_zero_ratio=zr, demand_class=pd.cut(zr, BINS, labels=LABELS).astype(str)).reset_index()
    assert len(out) == 30490 and out["series_id"].is_unique and out["demand_class"].ne("nan").all()
    path = ROOT / "data" / "full_series_ids_with_class.csv"
    out.to_csv(path, index=False)
    print(path, len(out), manifest_hash(out), out["demand_class"].value_counts(normalize=True).round(3).to_dict())


if __name__ == "__main__":
    main()
