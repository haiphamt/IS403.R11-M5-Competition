"""Shared helpers for the optional parts: Optuna tuning (T01), out-of-fold backtests and Ridge stacking (T02)."""

from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_raw_data
from .panel import build_panel

ROOT = Path(__file__).resolve().parents[2]


def load_sample_panel(data_dir=None, manifest=None, max_series=None):
    manifest = manifest or ROOT / "data" / "sample_series_ids.csv"
    ids = list(pd.read_csv(manifest)["series_id"])[:max_series]
    raw = load_raw_data(data_dir, table_names=("sales_train_evaluation", "calendar", "sell_prices"))
    return build_panel(raw["sales_train_evaluation"], raw["calendar"], raw["sell_prices"], ids)


def rmsse_scales(y, origin):
    """One-step naive MSE on d_1..origin after the first non-zero sale; NaN when undefined (M5 definition, as m5.metrics)."""
    scales = np.full(len(y), np.nan)
    for i, row in enumerate(np.asarray(y, dtype=float)[:, :origin]):
        nz = np.flatnonzero(row)
        if nz.size and row.size - nz[0] > 1:
            s = np.mean(np.diff(row[nz[0]:]) ** 2)
            scales[i] = s if s > 0 else np.nan
    return scales


def macro_rmsse(pred, y_true, scales):
    """Mean per-series RMSSE over series where the scale is defined; pred/y_true are (n_series, horizon)."""
    ok = ~np.isnan(scales)
    mse = np.mean((np.asarray(y_true, float) - np.asarray(pred, float)) ** 2, axis=1)
    return float(np.mean(np.sqrt(mse[ok] / scales[ok])))


def fit_ridge_meta(oof, base_cols, target="y_true", alphas=(0.1, 1.0, 10.0, 100.0, 1000.0)):
    """Ridge meta-learner on out-of-fold base forecasts; alpha by leave-one-origin-out RMSE (Train backtests only)."""
    from sklearn.linear_model import Ridge
    X, y, g = oof[list(base_cols)].to_numpy(float), oof[target].to_numpy(float), oof["origin"].to_numpy()
    if len(np.unique(g)) < 2:
        raise ValueError("need rows from at least 2 backtest origins for leave-one-origin-out CV")
    cv = {}
    for a in alphas:
        errs = []
        for o in np.unique(g):
            tr, te = g != o, g == o
            m = Ridge(alpha=a).fit(X[tr], y[tr])
            errs.append(np.mean((m.predict(X[te]) - y[te]) ** 2))
        cv[a] = float(np.sqrt(np.mean(errs)))
    best = min(cv, key=cv.get)
    return Ridge(alpha=best).fit(X, y), best, cv
