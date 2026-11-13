"""C03 per-series LightGBM pilot; reusable building blocks, not the M04 run."""

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

LAGS = (28, 35, 42)
WINDOWS = (7, 28)


def make_features(history, calendar, weekly_prices, *, state_id, train_end=1885, horizon=28):
    """Build one series with a fixed origin; never consume forecast-period sales.

    Prices must already be filtered to this item/store. Calendar and known
    prices are exogenous covariates. Missing prices stay NaN. Training excludes
    the first 55 days so every lag/rolling window is complete.
    """
    y = np.asarray(history, dtype=float)
    if y.ndim != 1 or len(y) != train_end or train_end < 56:
        raise ValueError("history must contain exactly train_end daily observations (>=56)")
    if not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("sales must be finite and nonnegative")
    if not isinstance(horizon, int) or not 1 <= horizon <= 28:
        raise ValueError("fixed-origin horizon must be between 1 and 28")
    if state_id not in {"CA", "TX", "WI"}:
        raise ValueError("unknown M5 state")
    days = [f"d_{d}" for d in range(1, train_end + horizon + 1)]
    if calendar['d'].duplicated().any():
        raise ValueError("duplicate calendar day")
    cal = calendar.set_index('d').loc[days].reset_index()
    if weekly_prices['wm_yr_wk'].duplicated().any():
        raise ValueError("duplicate weekly price for item/store")
    sales = pd.Series(np.r_[y, np.full(horizon, np.nan)])
    x = pd.DataFrame(index=range(len(days)))
    for lag in LAGS:
        x[f'lag_{lag}'] = sales.shift(lag)
    for window in WINDOWS:
        rolling = sales.shift(28).rolling(window, min_periods=window)
        x[f'rolling_mean_{window}'] = rolling.mean()
        x[f'rolling_std_{window}'] = rolling.std(ddof=1)
    for column in ['wday', 'month', 'year']:
        x[column] = pd.to_numeric(cal[column], errors='raise').to_numpy()
    for column in ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']:
        # One category dictionary shared by training and prediction.
        x[column] = pd.Categorical(cal[column].fillna('__NONE__'))
    x['snap'] = pd.to_numeric(cal[f'snap_{state_id}'], errors='raise').to_numpy()
    price_map = weekly_prices.set_index('wm_yr_wk')['sell_price']
    x['sell_price'] = cal['wm_yr_wk'].map(price_map).to_numpy(dtype=float)
    if np.isinf(x['sell_price']).any() or (x['sell_price'].dropna() <= 0).any():
        raise ValueError("observed prices must be finite and positive")
    x['price_missing'] = x['sell_price'].isna().astype('int8')
    x_train, x_future = x.iloc[55:train_end].copy(), x.iloc[train_end:].copy()
    history_columns = [c for c in x if c.startswith(('lag_', 'rolling_'))]
    if x_train[history_columns].isna().any().any() or x_future[history_columns].isna().any().any():
        raise ValueError("incomplete lag/rolling features")
    return x_train, y[55:].copy(), x_future


def make_model(parameters):
    return LGBMRegressor(**parameters)


def predict_nonnegative(model, features):
    prediction = np.asarray(model.predict(features), dtype=float)
    if prediction.shape != (len(features),) or not np.isfinite(prediction).all():
        raise ValueError("invalid LightGBM predictions")
    return np.maximum(prediction, 0.0)
