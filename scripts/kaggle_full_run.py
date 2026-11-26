"""One Kaggle job of the full-data study (all 30,490 series; 5 models: Seasonal Naive, ETS, LightGBM, Croston SBA + hybrid stacking).

    python scripts/kaggle_full_run.py --task validation          # base models, origin d_1885
    python scripts/kaggle_full_run.py --task test                # base models, origin d_1913 (needs tag config-freeze/-v2 in HEAD)
    python scripts/kaggle_full_run.py --task oof --origins 1829  # Train backtests for the stacking meta-learner (one origin per job)

Writes /kaggle/working/<task>_<run>.zip (checkpoints removed) plus data_sha256.json of the five CSVs actually read. The stack
itself (scripts/run_stacking.py) is cheap and runs locally once the Validation/Test runs and both OOF origins are downloaded.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "full_series_ids_with_class.csv"
BASE_MODELS = ["seasonal_naive", "ets", "croston_sba", "lightgbm"]
OOF_MODELS = ["ets", "croston_sba", "lightgbm"]
CSV = ("calendar.csv", "sell_prices.csv", "sales_train_validation.csv", "sales_train_evaluation.csv", "sample_submission.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["validation", "test", "oof"], required=True)
    ap.add_argument("--origins", nargs="+", default=["1829"])
    ap.add_argument("--work", type=Path, default=Path("/kaggle/working") if Path("/kaggle/working").exists() else ROOT / "results" / "kaggle")
    a = ap.parse_args()
    out = a.work / f"full_{a.task}" / ("oof_" + "_".join(a.origins) if a.task == "oof" else "runs")
    out.mkdir(parents=True, exist_ok=True)
    dirs = sorted({p.parent for p in Path("/kaggle/input").rglob("calendar.csv")}) if Path("/kaggle/input").exists() else [ROOT / "data" / "raw"]
    if len(dirs) != 1:
        raise SystemExit(f"cannot identify one M5 data directory: {dirs}")
    hashes = {f: hashlib.sha256((dirs[0] / f).read_bytes()).hexdigest() for f in CSV}
    (out / "data_sha256.json").write_text(json.dumps({"data_dir": str(dirs[0]), "sha256": hashes}, indent=1))
    cpus = os.cpu_count() or 4
    common = ["--manifest", str(MANIFEST), "--workers", str(cpus), "--data-dir", str(dirs[0])]
    if a.task == "oof":
        cmd = [sys.executable, str(ROOT / "scripts" / "backtest_oof.py"), "--models", *OOF_MODELS, "--origins", *a.origins,
               "--n-jobs", str(cpus), "--out", str(out), *common]
    else:
        cmd = [sys.executable, str(ROOT / "scripts" / "run_experiment.py"), "--split", a.task, "--models", *BASE_MODELS,
               "--n-jobs", str(cpus), "--out", str(out), *common]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)
    for ck in out.rglob("ckpt"):
        shutil.rmtree(ck, ignore_errors=True)
    print(shutil.make_archive(str(a.work / f"m5_full_{a.task}{'_' + '_'.join(a.origins) if a.task == 'oof' else ''}"), "zip", out))


if __name__ == "__main__":
    main()
