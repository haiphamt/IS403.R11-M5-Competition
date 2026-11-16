"""C02: đo thời gian hai mô hình chung trên 10 chuỗi M5 bằng CPU.

Tác giả: Thanh Tiền. Đây là thử nghiệm C02; không thay M05/M06 hay Test.
Chạy: python c02_benchmark.py --data-dir /path/to/M5 --output-dir results/c02
Giả định và cách đọc kết quả: ../benchmarks/c02/README.md.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import gc
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd

TRAIN_END, VAL_END, HORIZON = 1885, 1913, 28
IDS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
CATEGORICAL = IDS[1:] + ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]
CALENDAR = ["wday", "month", "year", "snap_CA", "snap_TX", "snap_WI"]
LAGS = [28, 35, 42]
WINDOWS = [7, 28]
HISTORY_FEATURES = [f"lag_{lag}" for lag in LAGS] + [
    f"rolling_{stat}_{w}" for w in WINDOWS for stat in ["mean", "std"]
]
FEATURES = CATEGORICAL + CALENDAR + ["sell_price", "price_missing"] + HISTORY_FEATURES
REQUIRED_FILES = ["sales_train_validation.csv", "calendar.csv", "sell_prices.csv"]
SOURCE_REVIEW_DATE = "2026-11-13"


@dataclass(frozen=True)
class Config:
    data_dir: str | None = None
    output_dir: str = "results/c02"
    n_series: int = 10
    seed: int = 42
    repeats: int = 3
    threads: int = 2

    def __post_init__(self):
        if min(self.n_series, self.repeats, self.threads) < 1:
            raise ValueError("n_series, repeats, threads must be positive integers")


def require(condition, message):
    """Kiểm tra điều kiện cả khi Python chạy với tùy chọn -O."""
    if not condition:
        raise ValueError(message)


def with_sales_ids(frame):
    """Giữ ID Kaggle; nếu CSV Mcompetitions thiếu ID thì ghép từ sản phẩm/cửa hàng."""
    if "id" not in frame:
        require(frame[["item_id", "store_id"]].notna().all().all(), "Missing item/store IDs")
        frame = frame.copy()
        frame.insert(0, "id", frame.item_id + "_" + frame.store_id + "_validation")
    return frame


def with_calendar_days(calendar):
    """Tính mã d_1, d_2,... từ ngày thực khi lịch Mcompetitions thiếu cột d."""
    if "d" not in calendar:
        require("date" in calendar, "Calendar needs d or date")
        dates = pd.to_datetime(calendar["date"], format="%Y-%m-%d", errors="raise")
        require(dates.notna().all() and dates.is_unique, "Missing/duplicate calendar dates")
        days = (dates - pd.Timestamp("2011-01-29")).dt.days + 1
        require((days >= 1).all(), "Calendar predates M5 d_1 (2011-01-29)")
        calendar = calendar.copy()
        calendar["d"] = "d_" + days.astype(str)
    return calendar


def find_data_dir(explicit=None):
    """Kiểm tra đúng thư mục đã chỉ định; báo lỗi nếu có nhiều bộ M5 để lựa chọn."""
    def complete(path):
        return all((path / name).is_file() for name in REQUIRED_FILES)
    explicit = explicit if explicit is not None else os.environ.get("M5_DATA_DIR")
    if explicit is not None:
        path = Path(explicit).expanduser().resolve()
        require(complete(path), f"Missing one of {REQUIRED_FILES} in {path}")
        return path
    candidates = {p.resolve() for p in [Path("data/raw"), Path("../data/raw")] if complete(p)}
    kaggle = Path("/kaggle/input")
    if kaggle.is_dir():
        candidates.update(p.parent.resolve() for p in kaggle.rglob(REQUIRED_FILES[0]) if complete(p.parent))
    require(len(candidates) == 1,
            f"Found {len(candidates)} M5 directories. Set DATA_DIR explicitly. Candidates: {sorted(map(str, candidates))}")
    return candidates.pop()


def select_series(metadata, n_series=10, seed=42):
    """Luân phiên cửa hàng/nhóm sản phẩm, chọn ngẫu nhiên trong nhóm với seed cố định.

    Với M5 đầy đủ và n=10: mỗi cửa hàng một chuỗi, bao phủ ba nhóm sản phẩm.
    Không dùng nhãn doanh số để chọn. Mẫu đo thời gian này khác mẫu D02.
    """
    require(not metadata["id"].duplicated().any(), "Duplicate sales IDs")
    require(metadata[IDS].notna().all().all(), "Missing sales identifiers")
    require(0 < n_series <= len(metadata), "Invalid sample size")
    rng = np.random.default_rng(seed)
    stores = rng.permutation(sorted(metadata.store_id.unique()))
    categories = rng.permutation(sorted(metadata.cat_id.unique()))
    pools = {}
    for (store, cat), group in metadata.groupby(["store_id", "cat_id"], sort=True):
        pools[store, cat] = list(rng.permutation(sorted(group.id.tolist())))
    selected, round_id = [], 0
    while len(selected) < n_series:
        progress = False
        for i, store in enumerate(stores):
            cat = categories[(i + round_id) % len(categories)]
            pool = pools.get((store, cat), [])
            if pool:
                selected.append(pool.pop())
                progress = True
                if len(selected) == n_series:
                    break
        round_id += 1
        if not progress and round_id % len(categories) == 0:
            # Nếu nhóm không có dữ liệu, chọn phần còn thiếu theo seed đã cố định.
            remaining = sorted(set(metadata.id) - set(selected))
            selected.extend(rng.permutation(remaining)[:n_series - len(selected)].tolist())
    return metadata.set_index("id").loc[sorted(selected)].reset_index()[IDS]


def load_sample(directory, cfg):
    """Quét doanh số hai lượt, lọc giá theo từng phần; không đọc nhãn Test."""
    sales_path = directory / REQUIRED_FILES[0]
    columns = pd.read_csv(sales_path, nrows=0).columns.tolist()
    day_cols = [f"d_{d}" for d in range(1, VAL_END + 1)]
    source_ids = [c for c in IDS if c != "id" or "id" in columns]
    require(set(source_ids + day_cols).issubset(columns), "sales_train_validation.csv has missing columns")
    metadata = with_sales_ids(pd.read_csv(sales_path, usecols=source_ids, dtype=str))
    selected = select_series(metadata, cfg.n_series, cfg.seed)
    selected_ids = set(selected.id)
    parts = []
    dtype = {**{c: str for c in IDS}, **{c: "int32" for c in day_cols}}
    scanned = 0
    for chunk in pd.read_csv(sales_path, usecols=source_ids + day_cols, dtype=dtype, chunksize=512):
        chunk = with_sales_ids(chunk)
        scanned += len(chunk)
        part = chunk[chunk.id.isin(selected_ids)]
        if len(part):
            parts.append(part.copy())
    sales = pd.concat(parts, ignore_index=True).sort_values("id").reset_index(drop=True)
    require(len(sales) == cfg.n_series and not sales.id.duplicated().any(), "Selected series missing/duplicated")
    require((sales[day_cols].to_numpy() >= 0).all(), "Negative sales")
    calendar = pd.read_csv(directory / "calendar.csv")
    calendar_day_derived = "d" not in calendar
    calendar = with_calendar_days(calendar)
    needed_cal = ["d", "wm_yr_wk"] + CALENDAR + CATEGORICAL[5:]
    require(set(needed_cal).issubset(calendar.columns), "Missing calendar columns")
    calendar = calendar[calendar.d.isin(day_cols)][needed_cal].copy()
    require(len(calendar) == VAL_END and not calendar.d.duplicated().any(), "Missing/duplicate calendar days")
    pairs = pd.MultiIndex.from_frame(selected[["store_id", "item_id"]])
    prices_parts, price_rows = [], 0
    for chunk in pd.read_csv(directory / "sell_prices.csv", chunksize=250_000):
        require(set(["store_id", "item_id", "wm_yr_wk", "sell_price"]).issubset(chunk.columns), "Missing price columns")
        price_rows += len(chunk)
        mask = pd.MultiIndex.from_frame(chunk[["store_id", "item_id"]]).isin(pairs)
        prices_parts.append(chunk.loc[mask].copy())
    prices = pd.concat(prices_parts, ignore_index=True)
    info = {"sales_rows_scanned": scanned, "sales_columns": len(columns),
            "prices_rows_scanned": price_rows, "selected_price_rows": len(prices),
            "calendar_rows_used": len(calendar), "full_panel_series": len(metadata),
            "input_adaptations": {"sales_id_derived_from_item_store": "id" not in columns,
                                  "calendar_d_derived_from_date": calendar_day_derived,
                                  "csv_files_modified": False},
            "input_files": {name: {"path": str(directory / name),
                                    "bytes": (directory / name).stat().st_size} for name in REQUIRED_FILES}}
    return sales, calendar, prices, selected, info


def create_features(sales, calendar, prices):
    """Chuyển sang bảng dài và tạo đặc trưng chỉ từ lịch sử được phép sử dụng."""
    day_cols = [f"d_{d}" for d in range(1, VAL_END + 1)]
    frame = sales.melt(id_vars=IDS, value_vars=day_cols, var_name="d", value_name="sales")
    frame["day_num"] = frame.d.str[2:].astype("int16")
    frame = frame.sort_values(["id", "day_num"]).reset_index(drop=True)
    frame["sales"] = frame.sales.astype("float32")
    grouped = frame.groupby("id", sort=False)["sales"]
    for lag in LAGS:
        frame[f"lag_{lag}"] = grouped.shift(lag).astype("float32")
    for w in WINDOWS:
        for stat in ["mean", "std"]:
            # Dịch 28 ngày trước khi tính trung bình/độ lệch chuẩn mẫu (ddof=1).
            frame[f"rolling_{stat}_{w}"] = grouped.transform(
                lambda x, w=w, stat=stat: getattr(x.shift(HORIZON).rolling(w, min_periods=w), stat)()
            ).astype("float32")
    frame = frame.merge(calendar, on="d", how="left", validate="many_to_one")
    require(frame[CALENDAR + ["wm_yr_wk"]].notna().all().all(), "Unmatched calendar")
    frame = frame.merge(prices, on=["store_id", "item_id", "wm_yr_wk"],
                        how="left", validate="many_to_one")
    frame["sell_price"] = frame.sell_price.astype("float32")
    require((frame.sell_price.dropna() > 0).all(), "Non-positive observed price")
    frame["price_missing"] = frame.sell_price.isna().astype("int8")
    # Giữ NaN ở giá và lịch sử chưa đủ: cả hai mô hình xử lý được biến số bị thiếu.
    for c in CATEGORICAL:
        frame[c] = frame[c].astype("object").fillna("__NO_EVENT__").astype(str)
    # Sắp theo thời gian, dùng cùng thứ tự dữ liệu cho hai mô hình.
    return frame.sort_values(["day_num", "id"]).reset_index(drop=True)


def audit_features(sales, calendar, prices, frame):
    """Đổi toàn bộ nhãn Validation; đặc trưng phải giữ nguyên để kiểm tra rò rỉ."""
    corrupted = sales.copy()
    val_cols = [f"d_{d}" for d in range(TRAIN_END + 1, VAL_END + 1)]
    corrupted[val_cols] = corrupted[val_cols] + 1_000_000
    alternative = create_features(corrupted, calendar, prices)
    pd.testing.assert_frame_equal(frame[FEATURES], alternative[FEATURES], check_exact=True)
    val = frame[frame.day_num > TRAIN_END]
    require(len(val) == len(sales) * HORIZON, "Incorrect validation size")
    require(not frame.duplicated(["id", "day_num"]).any(), "Duplicate series/day")
    require(int(val.day_num.max()) - min(LAGS) <= TRAIN_END, "Unavailable target used by lags")
    # Đối chiếu thủ công lag/rolling ở ngày đầu và cuối Validation của từng chuỗi.
    history = sales.set_index("id")
    for sid, group in val.groupby("id", sort=False):
        for _, row in group.iloc[[0, -1]].iterrows():
            day = int(row.day_num)
            for lag in LAGS:
                require(row[f"lag_{lag}"] == history.at[sid, f"d_{day-lag}"], "Lag misalignment")
            for w in WINDOWS:
                values = history.loc[sid, [f"d_{d}" for d in range(day-HORIZON-w+1, day-HORIZON+1)]].to_numpy(dtype=float)
                require(np.isclose(row[f"rolling_mean_{w}"], values.mean(), rtol=1e-5), "Rolling mean misalignment")
                require(np.isclose(row[f"rolling_std_{w}"], values.std(ddof=1), rtol=1e-5), "Rolling std misalignment")
    return {"validation_target_perturbation": "PASS", "lag_rolling_boundary_checks": "PASS",
            "unique_series_day": "PASS", "max_target_source_day_for_validation": VAL_END - HORIZON,
            "price_nan_preserved": True,
            "price_availability_assumption": "M5 contemporaneous weekly prices treated as known exogenous inputs; no future price imputation"}


def rmsse_scale(train):
    """Tính mẫu số RMSSE từ sai phân doanh số Train, bỏ các số 0 đầu chuỗi."""
    train = np.asarray(train, dtype=np.float64)
    nonzero = np.flatnonzero(train > 0)
    active = train[nonzero[0]:] if len(nonzero) else np.array([])
    scale = float(np.mean(np.diff(active) ** 2)) if len(active) >= 2 else 0.0
    fallback = scale <= 0
    # Quy ước scale=1 cho chuỗi suy biến và đánh dấu; đây không phải điểm M5 chính thức.
    return (1.0 if fallback else scale), fallback


def evaluate(val, prediction, histories):
    """Tính MAE/RMSE/R² chung và trung bình RMSSE không trọng số giữa các chuỗi."""
    y = val.sales.to_numpy(dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.float64)
    require(pred.shape == y.shape and np.isfinite(pred).all(), "Invalid predictions")
    error = y - pred
    denom = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - float(np.sum(error ** 2)) / denom if denom > 0 else float(np.all(error == 0))
    metrics = {"MAE": float(np.mean(np.abs(error))), "RMSE": float(np.sqrt(np.mean(error ** 2))), "R2_pooled": r2}
    scored = val[["id", "sales"]].copy()
    scored["prediction"] = pred
    series_metrics = []
    for sid, group in scored.groupby("id", sort=True):
        scale, fallback = rmsse_scale(histories[sid])
        mse = float(np.mean((group.sales.to_numpy(dtype=float) - group.prediction.to_numpy()) ** 2))
        series_metrics.append({"id": sid, "RMSSE": float(np.sqrt(mse / scale)),
                               "scale_used": scale, "scale_fallback": fallback})
    metrics["RMSSE_mean"] = float(np.mean([r["RMSSE"] for r in series_metrics]))
    metrics["scale_fallback_count"] = sum(r["scale_fallback"] for r in series_metrics)
    return metrics, series_metrics


def encode_features(train, val, model_name):
    """XGBoost chỉ lấy danh mục từ Train; nhóm mới ở Validation được chuyển thành NaN."""
    x_train, x_val = train[FEATURES].copy(), val[FEATURES].copy()
    unknown = {}
    for c in CATEGORICAL:
        unknown[c] = int((~x_val[c].isin(x_train[c].unique())).sum())
        if model_name == "XGBoost":
            dtype = pd.CategoricalDtype(categories=sorted(x_train[c].unique()))
            x_train[c] = x_train[c].astype(dtype)
            # Che nhóm chưa xuất hiện trước khi ép kiểu để tương thích các bản pandas mới.
            x_val[c] = x_val[c].where(x_val[c].isin(dtype.categories), None).astype(dtype)
        else:
            x_train[c], x_val[c] = x_train[c].astype(str), x_val[c].astype(str)
    return x_train, x_val, unknown


def build_model(name, cfg):
    """Giữ mặc định số cây/độ sâu/tốc độ học; đặt CPU, seed và số luồng để đo."""
    if name == "XGBoost":
        from xgboost import XGBRegressor
        return XGBRegressor(objective="reg:squarederror", tree_method="hist", device="cpu",
                            enable_categorical=True, random_state=cfg.seed, n_jobs=cfg.threads)
    from catboost import CatBoostRegressor
    return CatBoostRegressor(loss_function="RMSE", task_type="CPU", random_seed=cfg.seed,
                             thread_count=cfg.threads, verbose=False, allow_writing_files=False)


def environment():
    """Ghi cấu hình máy và phiên bản; báo rõ gói thiếu trước khi đo thời gian."""
    packages, missing = {}, []
    for name in ["numpy", "pandas", "scikit-learn", "xgboost", "catboost", "psutil"]:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            missing.append(name)
    if missing:
        raise RuntimeError(
            "Runtime hiện tại thiếu thư viện hoặc thông tin bản cài: " + ", ".join(missing)
            + ". Trên Colab, chạy lại ô ở mục 1. Thư viện rồi chạy các mục 2–9 theo thứ tự. "
            "Kết quả cũ lưu trong notebook không chứng minh runtime mới đã cài gói. "
            "Nếu chạy script, cài bằng: python -m pip install -r benchmarks/c02/requirements.txt"
        )
    import psutil
    vm = psutil.virtual_memory()
    cpu = platform.processor()
    if Path("/proc/cpuinfo").is_file():
        cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")), cpu)
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    return {"python": platform.python_version(), "os": platform.platform(), "cpu": cpu,
            "logical_cpus": os.cpu_count(), "affinity_cpus": affinity,
            "ram_total_GiB": vm.total / 2**30, "ram_available_GiB_at_start": vm.available / 2**30,
            "packages": packages,
            "platform": ("Kaggle" if Path("/kaggle").is_dir() else "Colab"
                         if ("google.colab" in sys.modules or os.environ.get("COLAB_RELEASE_TAG")
                             or os.environ.get("COLAB_BACKEND_VERSION")) else "Local/other"),
            "accelerator_used": "CPU for both models"}


def markdown_table(table):
    """Tạo bảng Markdown mà không cần cài thêm thư viện tabulate."""
    return "\n".join(["| " + " | ".join(map(str, table.columns)) + " |",
                      "| " + " | ".join(["---"] * len(table.columns)) + " |"] +
                     ["| " + " | ".join(f"{v:.6f}" if isinstance(v, (float, np.floating)) else str(v) for v in row) + " |"
                      for row in table.itertuples(index=False, name=None)])


def run_benchmark(cfg=Config(), *, source_sha256=None, show_summary=True):
    """Chạy trên đầu vào thật; không thay bằng dữ liệu giả, không huấn luyện/chấm Test."""
    env = environment()  # Lấy phiên bản ngoài khoảng thời gian đo xử lý dữ liệu/mô hình.
    # Nạp hai thư viện trước khi đo; không chạy khởi động hay bỏ lần đo đầu.
    import xgboost
    import catboost
    start_local = datetime.now(timezone(timedelta(hours=7)))
    directory = find_data_dir(cfg.data_dir)
    root = Path(cfg.output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_id = "C02_" + start_local.strftime("%Y%m%dT%H%M%S_%f%z")
    destination = root / run_id
    destination.mkdir()  # Mỗi lần chạy có thư mục riêng, không ghi đè kết quả cũ.
    print(f"{run_id}: {cfg.n_series} chuỗi, {cfg.repeats} lần lặp, CPU {cfg.threads} luồng", flush=True)
    pipeline_start = time.perf_counter()
    t = time.perf_counter()
    sales, calendar, prices, selected, input_info = load_sample(directory, cfg)
    load_s = time.perf_counter() - t
    t = time.perf_counter()
    frame = create_features(sales, calendar, prices)
    feature_s = time.perf_counter() - t
    t = time.perf_counter()
    audit = audit_features(sales, calendar, prices, frame)
    audit_s = time.perf_counter() - t
    train = frame[frame.day_num <= TRAIN_END].copy()
    val = frame[frame.day_num > TRAIN_END].copy()
    require(len(train) == cfg.n_series * TRAIN_END, "Incorrect training size")
    histories = {row["id"]: row[[f"d_{d}" for d in range(1, TRAIN_END+1)]].to_numpy(dtype=float)
                 for _, row in sales.iterrows()}
    records, details, parameters = [], [], {}
    print(f"Train={len(train):,}; Validation={len(val):,}; kiểm tra rò rỉ dữ liệu: PASS", flush=True)
    for rep in range(1, cfg.repeats + 1):
        # Đổi thứ tự hai mô hình để giảm ảnh hưởng của việc luôn chạy trước.
        order = ["XGBoost", "CatBoost"] if rep % 2 else ["CatBoost", "XGBoost"]
        for name in order:
            gc.collect()  # Dọn bộ nhớ ngoài khoảng thời gian đo mô hình.
            total_start = time.perf_counter()
            t = time.perf_counter()
            x_train, x_val, unknown = encode_features(train, val, name)
            encoding_s = time.perf_counter() - t
            model = build_model(name, cfg)
            t = time.perf_counter()
            fit_kwargs = {"cat_features": CATEGORICAL} if name == "CatBoost" else {}
            model.fit(x_train, train.sales.to_numpy(), **fit_kwargs)
            fit_s = time.perf_counter() - t
            t = time.perf_counter()
            predict_kwargs = {"thread_count": cfg.threads} if name == "CatBoost" else {}
            raw_prediction = np.asarray(model.predict(x_val, **predict_kwargs), dtype=float)
            prediction = np.maximum(raw_prediction, 0)
            predict_s = time.perf_counter() - t  # Gồm đưa dự báo âm về 0, chưa tính độ đo.
            t = time.perf_counter()
            metrics, per_series = evaluate(val, prediction, histories)
            score_s = time.perf_counter() - t
            total_s = time.perf_counter() - total_start
            records.append({"model": name, "repeat": rep, "n_series": cfg.n_series,
                            "train_rows": len(train), "validation_rows": len(val),
                            "encoding_s": encoding_s, "fit_s": fit_s, "predict_s": predict_s,
                            "score_s": score_s, "model_total_s": total_s,
                            "negative_predictions_clipped": int((raw_prediction < 0).sum()), **metrics})
            details.append({"model": name, "repeat": rep, "per_series": per_series})
            if rep == 1:
                # missing=np.nan hợp lệ với XGBoost; đổi sang chuỗi khi ghi JSON chuẩn.
                constructor = json.loads(json.dumps(model.get_params(), default=str), parse_constant=str)
                parameters[name] = {"constructor": constructor, "unknown_validation_categories": unknown,
                                    "predict_kwargs": predict_kwargs}
                if name == "XGBoost":
                    parameters[name]["resolved_booster_config"] = json.loads(model.get_booster().save_config())
                    parameters[name]["boosted_rounds"] = model.get_booster().num_boosted_rounds()
                else:
                    parameters[name]["resolved_parameters"] = model.get_all_params()
                    parameters[name]["tree_count"] = model.tree_count_
            print(f"{name} rep {rep}: fit={fit_s:.3f}s; predict={predict_s:.4f}s; RMSSE={metrics['RMSSE_mean']:.5f}", flush=True)
            del model, x_train, x_val
    pipeline_s = time.perf_counter() - pipeline_start
    trials = pd.DataFrame(records)
    timing = ["encoding_s", "fit_s", "predict_s", "score_s", "model_total_s"]
    metric_cols = ["MAE", "RMSE", "RMSSE_mean", "R2_pooled", "scale_fallback_count"]
    summary = trials.groupby("model", sort=False)[timing + metric_cols].median().reset_index()
    for col in ["fit_s", "predict_s", "model_total_s"]:
        summary[col + "_min"] = summary.model.map(trials.groupby("model")[col].min())
        summary[col + "_max"] = summary.model.map(trials.groupby("model")[col].max())
    summary.insert(1, "n_series", cfg.n_series)
    summary.insert(2, "repeats", cfg.repeats)
    shared = {"load_s": load_s, "features_s": feature_s, "audit_s": audit_s,
              "pipeline_wall_s_all_repeats": pipeline_s,
              "scope": "pipeline includes shared work, audits and all model repetitions; excludes imports, environment collection and output writes"}
    if source_sha256 is None and "__file__" in globals():
        source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    metadata = {"run_id": run_id, "task": "C02", "run_started_Asia_Bangkok": start_local.isoformat(),
                "status": "completed",
                "source_sha256": source_sha256, "source_review_date": SOURCE_REVIEW_DATE,
                "config": asdict(cfg), "environment": env, "inputs": input_info, "checks": audit,
                "shared_runtime": shared, "features": FEATURES, "categorical": CATEGORICAL,
                "parameters": parameters, "per_series_metrics": details,
                "policies": {"split": {"train": [1, TRAIN_END], "validation": [TRAIN_END+1, VAL_END]},
                             "training_rows_with_missing_lags": "retained; numeric NaN handled natively",
                             "forecast": "one pooled global model per algorithm; non-recursive fixed 28-day horizon",
                             "rmsse": "unweighted mean of series RMSSE; scale=1 fallback explicitly flagged; not WRMSSE",
                             "r2": "pooled across all series/day rows; constant target: perfect=1, otherwise=0",
                             "model_defaults": "untuned library defaults except explicit objective, CPU, categorical support, seed, threads, logs/file writing",
                             "price": audit["price_availability_assumption"],
                             "selection": "seeded round-robin store/category timing sample, not D02 representative sample",
                             "test": "not loaded or scored", "repeats": "fresh model each repetition; same seed; alternating order; no warm-up discarded"}}
    selected.to_csv(destination / "selected_series.csv", index=False)
    trials.to_csv(destination / "runtime_trials.csv", index=False)
    summary.to_csv(destination / "runtime_summary.csv", index=False)
    (destination / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=str, allow_nan=False), encoding="utf-8")
    packages = env["packages"]
    (destination / "requirements_observed.txt").write_text("\n".join(f"{k}=={v}" for k, v in packages.items()) + "\n", encoding="utf-8")
    report = (f"# C02 — {run_id}\n\n"
              f"Chạy CSV thật trên **{env['platform']}**, CPU {cfg.threads} luồng, {cfg.n_series} chuỗi. "
              f"Train d_1–d_{TRAIN_END}; Validation d_{TRAIN_END+1}–d_{VAL_END}. Không dùng Test.\n\n"
              "## Kết quả đo\n\nTrung vị qua các lần khởi tạo mô hình mới, seed cố định, không tinh chỉnh; thời gian tính bằng giây.\n\n"
              + markdown_table(summary) + "\n\n## Xử lý dữ liệu dùng chung (tính một lần)\n\n"
              + markdown_table(pd.DataFrame([{k: v for k, v in shared.items() if k != 'scope'}]))
              + "\n\nRMSSE_mean là trung bình không trọng số, khác WRMSSE. Chuỗi suy biến dùng scale=1 và được đánh dấu trong metadata; "
              "R² tính chung trên các dòng Validation. Giả định biết trước giá theo tuần của M5. "
              "Đặc trưng trễ chưa đủ lịch sử và giá thiếu được giữ NaN.\n\n"
              "Kết quả chỉ mô tả mẫu đã chọn trong môi trường được ghi lại.\n")
    (destination / "run_report.md").write_text(report, encoding="utf-8")
    if show_summary:
        print("\n" + markdown_table(summary), flush=True)
        print(f"\nArtifacts: {destination}", flush=True)
    return {"summary": summary, "trials": trials, "selected": selected,
            "metadata": metadata, "output_dir": destination}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir", default="results/c02")
    parser.add_argument("--n-series", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    try:
        run_benchmark(Config(**vars(args)))
    except (ValueError, OSError) as exc:
        parser.exit(2, f"C02 failed: {exc}\n")


if __name__ == "__main__":
    main()
