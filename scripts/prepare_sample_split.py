"""D02/D03/D07: data schema check, stratified sample, split files, quality report.

Usage: python scripts/prepare_sample_split.py [--data-dir data/raw]
Writes data/sample_series_ids.csv, results/tables/d02_*.csv|json and
data/processed/{train,validation,test}.parquet (wide, one row per series; ignored by Git).
Splits are cut from sales_train_evaluation (d_1..d_1941); the d_1886-1913 days
are identical in sales_train_validation, which is checked here.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from m5.sample import manifest_hash, representativeness, select_stratified  # noqa: E402
from m5.schema import series_id_from_id  # noqa: E402
from m5.split import split_wide, to_wide  # noqa: E402

EXPECTED = {"calendar.csv": (1969, 14), "sales_train_validation.csv": (30490, 1919),
            "sales_train_evaluation.csv": (30490, 1947), "sample_submission.csv": (60980, 29),
            "sell_prices.csv": (6841121, 4)}


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per-stratum", type=int, default=40)
    args = ap.parse_args()
    raw = Path(args.data_dir)

    report = {"files": {}}
    for name, (rows, cols) in EXPECTED.items():
        path = raw / name
        if not path.exists():
            raise SystemExit(f"missing {path}")
        with open(path, "rb") as f:
            n_rows = sum(1 for _ in f) - 1
        n_cols = len(pd.read_csv(path, nrows=0).columns)
        ok = (n_rows, n_cols) == (rows, cols)
        report["files"][name] = {"rows": n_rows, "cols": n_cols, "expected": [rows, cols],
                                 "ok": ok, "sha256": sha256(path)}
        if not ok:
            raise SystemExit(f"{name}: got {(n_rows, n_cols)}, expected {(rows, cols)}")

    sales = pd.read_csv(raw / "sales_train_evaluation.csv")
    val = pd.read_csv(raw / "sales_train_validation.csv")
    cal = pd.read_csv(raw / "calendar.csv")
    report["unique_ids"] = bool(sales["id"].is_unique)
    report["calendar_d_unique"] = bool(cal["d"].is_unique and len(cal) == 1969)
    day_cols = [f"d_{day}" for day in range(1, 1914)]
    wide = to_wide(sales)
    val.index = val["id"].map(series_id_from_id)
    same_series = val.index.is_unique and wide.index.difference(val.index).empty
    report["validation_file_equals_evaluation_d1_1913"] = same_series and all(
        val[day].equals(wide[day].reindex(val.index)) for day in day_cols
    )
    del val
    report["negative_sales"] = int((sales.filter(like="d_") < 0).sum().sum())
    report["missing_sales"] = int(sales.filter(like="d_").isna().sum().sum())

    manifest = select_stratified(sales, per_stratum=args.per_stratum, seed=args.seed)
    out_ids = ROOT / "data" / "sample_series_ids.csv"
    manifest.to_csv(out_ids, index=False)
    report["sample"] = {"seed": args.seed, "per_stratum": args.per_stratum,
                        "n_series": len(manifest), "manifest_sha256": manifest_hash(manifest),
                        "n_strata": int(manifest.groupby(["store_id", "cat_id"]).ngroups)}

    rep = representativeness(wide, manifest["series_id"])
    tables = ROOT / "results" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    rep.to_csv(tables / "d02_sample_representativeness.csv")

    # Train-derived intermittency classes (zero ratio on d_1..d_1885 only)
    train_ids = manifest["series_id"]
    parts = split_wide(wide, train_ids)
    zr = (parts["train"] == 0).mean(axis=1)
    cls = pd.cut(zr, [-0.001, 0.5, 0.8, 1.0], labels=["low_zero(<=0.5)", "mid_zero(0.5-0.8]", "high_zero(>0.8)"])
    full_zr = (split_wide(wide)["train"] == 0).mean(axis=1)
    full_cls = pd.cut(full_zr, [-0.001, 0.5, 0.8, 1.0], labels=cls.cat.categories)
    cmp = pd.DataFrame({"sample_share": cls.value_counts(normalize=True).sort_index(),
                        "all_share": full_cls.value_counts(normalize=True).sort_index()})
    cmp.to_csv(tables / "d02_intermittency_classes.csv")
    manifest.assign(train_zero_ratio=zr.reindex(manifest["series_id"]).to_numpy(),
                    demand_class=cls.reindex(manifest["series_id"]).astype(str).to_numpy()
                    ).to_csv(ROOT / "data" / "sample_series_ids_with_class.csv", index=False)

    proc = ROOT / "data" / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    for name, frame in parts.items():
        frame.reset_index().to_parquet(proc / f"{name}.parquet", index=False)
    report["splits"] = {k: [v.shape[0], v.shape[1], v.columns[0], v.columns[-1]] for k, v in parts.items()}
    (tables / "d07_data_quality_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["sample"], indent=2))
    print(rep.to_string())
    print(cmp.to_string())


if __name__ == "__main__":
    main()
