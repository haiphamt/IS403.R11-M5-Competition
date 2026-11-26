"""Global (pooled over series) regressors: Linear Regression, LightGBM, XGBoost, CatBoost.

One model is fitted per origin on all (series, day) rows whose target day <= origin.
Features of day t read y only up to t-28, so the 28 forecast days are predicted directly
(non-recursive) from history <= origin. Labels after ``origin`` are masked to NaN here,
so a predictor can never see them even though the panel stores them.
"""

import time

import numpy as np
import pandas as pd

from .features import lag_roll_features
from .panel import Panel

HORIZON = 28
FIRST_DAY_IDX = 55  # day d_56: first day whose lag_42 / rolling_28(shift 28) windows are complete
SEED = 42
LAG_ROLL = ["lag_28", "lag_35", "lag_42", "rolling_mean_7", "rolling_std_7", "rolling_mean_28", "rolling_std_28"]
CAT_COLS = ["dept_id", "cat_id", "store_id", "state_id", "wday", "month",
            "event_name_1", "event_type_1", "event_name_2", "event_type_2"]
NUM_COLS = LAG_ROLL + ["year", "snap", "sell_price", "price_missing"]
FEATURES = NUM_COLS + CAT_COLS
MODELS = ["linear_regression", "lightgbm", "xgboost", "catboost", "lightgbm_noprice", "lightgbm_tuned"]
NOPRICE_EVENT = ["sell_price", "price_missing", "event_name_1", "event_type_1", "event_name_2", "event_type_2"]  # X02 ablation


def _categorical(values, idx, n_series, per_day):
    """Categorical column of n_series x len(idx) rows built from codes (no per-row Python strings: 30,490 series fit in RAM).

    Categories are the sorted string values, as pd.Categorical(strings) would give, so codes match the earlier string build.
    per_day=True: one value per calendar day (tiled over series); False: one value per series (repeated over days)."""
    cats, codes = np.unique(np.asarray(values, dtype=str), return_inverse=True)
    codes = np.tile(codes[idx], n_series) if per_day else np.repeat(codes, len(idx))
    return pd.Categorical.from_codes(codes.astype(np.int16), categories=cats)


def build_table(panel: Panel, origin, horizon=HORIZON):
    """Return (X_train, y_train, X_future, future_keys) for forecast origin ``origin``."""
    y = panel.y[:, :origin + horizon].copy()
    y[:, origin:] = np.nan  # forecast-period labels never enter any feature
    feats = lag_roll_features(y)
    del y
    n = panel.y.shape[0]
    cal = panel.calendar.iloc[:origin + horizon]
    idx = np.arange(FIRST_DAY_IDX, origin + horizon)
    cols = {name: feats.pop(name)[:, idx].ravel() for name in LAG_ROLL}
    cols["year"] = np.tile(cal["year"].to_numpy(float)[idx], n)
    cols["snap"] = panel.snap[:, idx].ravel()
    price = panel.price[:, idx].ravel()
    cols["sell_price"] = price
    cols["price_missing"] = np.isnan(price).astype(float)
    del price
    for c in ("wday", "month", "event_name_1", "event_type_1", "event_name_2", "event_type_2"):
        cols[c] = _categorical(cal[c].fillna("__NONE__").astype(str).to_numpy(), idx, n, per_day=True)
    for c in ("dept_id", "cat_id", "store_id", "state_id"):
        cols[c] = _categorical(panel.meta[c].to_numpy(), idx, n, per_day=False)
    X = pd.DataFrame(cols, copy=False)[FEATURES]
    del cols
    is_train = np.tile(idx + 1 <= origin, n)
    X_train, X_future = X[is_train].reset_index(drop=True), X[~is_train].reset_index(drop=True)
    del X
    if X_train[LAG_ROLL].isna().any().any() or X_future[LAG_ROLL].isna().any().any():
        raise ValueError("lag/rolling features contain NaN")
    y_train = panel.y[:, idx].ravel()[is_train]
    fut_days = idx[idx + 1 > origin] + 1
    keys = pd.DataFrame({"series_id": np.repeat(np.array(panel.series_ids), len(fut_days)), "d_num": np.tile(fut_days, n)})
    return X_train, y_train, X_future, keys


def _as_category(X):
    """Categorical columns with the categories actually present (same codes as building them from strings per frame)."""
    X = X.copy(deep=False)  # only the categorical columns are replaced; no 6 GB copy of the numeric block on full data
    for c in CAT_COLS:
        if c in X.columns:
            X[c] = X[c].cat.remove_unused_categories() if isinstance(X[c].dtype, pd.CategoricalDtype) else pd.Categorical(X[c].astype(str))
    return X


def fit_predict(name, X_train, y_train, X_future, *, n_jobs=8, seed=SEED, lgbm_params=None):
    """Fit one default-parameter model. Returns (prediction, info, fitted model)."""
    info = {}
    t0 = time.perf_counter()
    if name == "linear_regression":
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import Pipeline, make_pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        num = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        pre = ColumnTransformer([("num", num, NUM_COLS),
                                 ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT_COLS)])
        model = Pipeline([("pre", pre), ("lr", LinearRegression())])
        Xt, Xf = X_train, X_future
    elif name in ("lightgbm", "lightgbm_noprice", "lightgbm_tuned"):
        from lightgbm import LGBMRegressor
        if name == "lightgbm_noprice":
            X_train, X_future = X_train.drop(columns=NOPRICE_EVENT), X_future.drop(columns=NOPRICE_EVENT)
        if name == "lightgbm_tuned" and not lgbm_params:
            raise ValueError("lightgbm_tuned needs lgbm_params from the Optuna study")
        extra = lgbm_params if name == "lightgbm_tuned" else {}
        model = LGBMRegressor(random_state=seed, n_jobs=n_jobs, verbosity=-1, **extra)
        Xt, Xf = _as_category(X_train), _as_category(X_future)
    elif name == "xgboost":
        from xgboost import XGBRegressor
        model = XGBRegressor(random_state=seed, n_jobs=n_jobs, enable_categorical=True)
        Xt, Xf = _as_category(X_train), _as_category(X_future)
    elif name == "catboost":
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(random_seed=seed, thread_count=n_jobs, verbose=0,
                                  allow_writing_files=False, cat_features=CAT_COLS)
        Xt, Xf = (X.astype({c: str for c in CAT_COLS}) for X in (X_train, X_future))  # CatBoost reads category values as strings
    else:
        raise ValueError(f"unknown model {name!r}")
    model.fit(Xt, y_train)
    t1 = time.perf_counter()
    pred = np.asarray(model.predict(Xf), dtype=float)
    info["fit_s"], info["predict_s"] = t1 - t0, time.perf_counter() - t1
    if pred.shape != (len(X_future),) or not np.isfinite(pred).all():
        raise RuntimeError(f"{name}: invalid predictions")
    return pred, info, model


def model_parameters(name, model):
    """Full parameter dump of the fitted estimator (shows library defaults actually used)."""
    if name == "linear_regression":
        return model.named_steps["lr"].get_params()
    if name == "catboost":
        return {**model.get_all_params()}
    if name == "xgboost":  # get_params() leaves unset values as None; record what the booster really used
        import json
        return {**model.get_params(), "n_boosted_rounds": model.get_booster().num_boosted_rounds(),
                "booster_config": json.loads(model.get_booster().save_config())}
    return model.get_params()


def importance_or_coefficients(name, model):
    """Tidy table of LR coefficients or boosting feature importance."""
    if name == "linear_regression":
        names = model.named_steps["pre"].get_feature_names_out()
        return pd.DataFrame({"feature": names, "coefficient": model.named_steps["lr"].coef_})
    if name == "catboost":
        imp = model.get_feature_importance()
        kind = "PredictionValuesChange"
    elif name == "xgboost":
        imp = model.feature_importances_
        kind = "gain (normalised)"
    else:
        imp = model.booster_.feature_importance(importance_type="gain")
        kind = "gain"
    names = list(getattr(model, "feature_name_", FEATURES))  # shorter for the no-price ablation
    return pd.DataFrame({"feature": names, "importance": imp, "type": kind}).sort_values(
        "importance", ascending=False)
