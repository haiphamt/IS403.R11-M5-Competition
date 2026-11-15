"""Per-series (local) forecasters with timing and diagnostics, run in chunks.

Each chunk is independent so a long ETS/Croston job can checkpoint every ~50 series.
Timing covers fit + predict of one series only (no I/O); warm-up (numba JIT) is timed
separately by ``warmup``.
"""

import time

import numpy as np

from .seasonal_naive import seasonal_naive

HORIZON = 28
SEASON = 7
CROSTON_ALPHA = 0.1  # statsforecast CrostonClassic default smoothing


def warmup():
    """Compile statsforecast kernels once; returns seconds spent."""
    t0 = time.perf_counter()
    y = np.random.default_rng(0).poisson(1.0, 200).astype(float)
    ets(y)
    croston(y)
    return time.perf_counter() - t0


def croston_components(y, alpha=CROSTON_ALPHA):
    """Reference z (demand size) and p (interval) of classic Croston; forecast = z / p."""
    y = np.asarray(y, dtype=float)
    nz = np.flatnonzero(y)
    if nz.size == 0:
        return 0.0, 1.0
    z, p, q = y[nz[0]], float(nz[0] + 1), 1
    for t in range(nz[0] + 1, len(y)):
        if y[t] > 0:
            z += alpha * (y[t] - z)
            p += alpha * (q - p)
            q = 1
        else:
            q += 1
    return float(z), float(p)


def ets(y):
    from statsforecast.models import AutoETS
    model = AutoETS(season_length=SEASON)  # model selection by AICc (library default)
    model.fit(y)
    fc = np.asarray(model.predict(h=HORIZON)["mean"], dtype=float)
    fitted = getattr(model, "model_", {}) or {}
    par = fitted.get("par", None)
    info = {"ets_method": str(fitted.get("method", "")), "ets_aicc": float(fitted.get("aicc", np.nan))}
    if par is not None:
        names = list(getattr(par, "index", [])) or ["alpha", "beta", "gamma", "phi"][: len(par)]
        for k, v in zip(names, np.asarray(par, dtype=float)):
            info[f"ets_{k}"] = float(v)
    return fc, info


def croston(y):
    from statsforecast.models import CrostonClassic
    model = CrostonClassic()
    model.fit(y)
    fc = np.asarray(model.predict(h=HORIZON)["mean"], dtype=float)
    z, p = croston_components(y)
    return fc, {"croston_z": z, "croston_p": p, "croston_zp": z / p}


def forecast_one(name, y):
    if name == "seasonal_naive":
        return seasonal_naive(y, HORIZON, SEASON), {}
    if name == "ets":
        return ets(y)
    if name == "croston":
        return croston(y)
    raise ValueError(f"unknown model {name!r}")


def run_chunk(name, series_ids, histories):
    """Forecast several series. Returns list of dict(series_id, forecast, fit_predict_s, error, info)."""
    out = []
    for sid, y in zip(series_ids, histories):
        y = np.asarray(y, dtype=float)
        if not np.isfinite(y).all():
            raise ValueError(f"{sid}: history contains NaN/inf")
        t0 = time.perf_counter()
        error = ""
        try:
            fc, info = forecast_one(name, y)
            if fc.shape != (HORIZON,) or not np.isfinite(fc).all():
                raise RuntimeError("invalid forecast")
        except Exception as exc:  # never swap in another model silently: record and fall back visibly
            error = f"{type(exc).__name__}: {exc}"
            fc, info = seasonal_naive(y, HORIZON, SEASON), {}
        out.append({"series_id": sid, "forecast": fc, "fit_predict_s": time.perf_counter() - t0,
                    "fallback": bool(error), "error": error, "info": info})
    return out
