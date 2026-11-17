"""Bootstrap confidence intervals over series for the macro-RMSSE of the seven core models, and paired
differences against ETS (the top model by point estimate).

    python scripts/bootstrap_ci.py --split test --runs results/runs/tes-A results/runs/tes-B --out results/tables/test

Resamples the 1,200 series with replacement (B draws, seed 42) and recomputes the macro mean. It captures
sampling variation across series only: one 28-day window, and series that share stores and days are not
independent, so the intervals are descriptive and probably too narrow.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PAIRS = [("croston", "lightgbm"), ("croston", "linear_regression"), ("lightgbm", "xgboost"), ("lightgbm", "catboost"), ("xgboost", "catboost")]
ORDER = ["seasonal_naive", "linear_regression", "ets", "croston", "lightgbm", "xgboost", "catboost"]


def bootstrap_means(wide, draws=2000, seed=42):
    """wide: series x model RMSSE (NaN-free). Returns (draws x model) array of macro means over resampled series."""
    rng = np.random.default_rng(seed)
    values = wide.to_numpy(float)
    idx = rng.integers(0, len(values), size=(draws, len(values)))
    return values[idx].mean(axis=1)


def summarize(wide, reference="ets", draws=2000, seed=42):
    means = bootstrap_means(wide, draws, seed)
    cols = list(wide.columns)
    ref = cols.index(reference)
    rows = []
    for j, m in enumerate(cols):
        lo, hi = np.percentile(means[:, j], [2.5, 97.5])
        diff = means[:, j] - means[:, ref]
        d_lo, d_hi = np.percentile(diff, [2.5, 97.5])
        rows.append({"model": m, "macro_rmsse": float(wide[m].mean()), "ci_low": float(lo), "ci_high": float(hi),
                     f"diff_vs_{reference}": float(wide[m].mean() - wide[reference].mean()), "diff_ci_low": float(d_lo),
                     "diff_ci_high": float(d_hi), f"share_draws_better_than_{reference}": float((diff < 0).mean()) if m != reference else float("nan")})
    return pd.DataFrame(rows)


def pair_summary(wide, pairs=PAIRS, draws=2000, seed=42):
    """Paired bootstrap of (first - second) macro-RMSSE: negative means the first model is better."""
    means = bootstrap_means(wide, draws, seed)
    cols = list(wide.columns)
    rows = []
    for a, b in pairs:
        diff = means[:, cols.index(a)] - means[:, cols.index(b)]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        rows.append({"first": a, "second": b, "diff": float(wide[a].mean() - wide[b].mean()), "ci_low": float(lo), "ci_high": float(hi),
                     "share_draws_first_better": float((diff < 0).mean())})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["validation", "test"], required=True)
    ap.add_argument("--runs", nargs="+", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--draws", type=int, default=2000)
    a = ap.parse_args()
    ps = pd.concat([pd.read_csv(r / "metrics_per_series.csv") for r in a.runs])
    wide = ps.pivot(index="series_id", columns="model", values="rmsse")[ORDER]
    if wide.isna().any().any():
        raise ValueError("RMSSE undefined for some series; the macro mean would change between draws")
    out = summarize(wide, draws=a.draws)
    a.out.mkdir(parents=True, exist_ok=True)
    out.round(6).to_csv(a.out / f"{a.split}_bootstrap_ci.csv", index=False)
    pairs = pair_summary(wide, draws=a.draws)
    pairs.round(6).to_csv(a.out / f"{a.split}_bootstrap_pairs.csv", index=False)
    print(out.round(4).to_string(index=False)); print(pairs.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
