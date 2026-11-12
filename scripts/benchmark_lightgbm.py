"""C03: train/predict LightGBM on exactly ten original M5 series, CPU only."""

import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
import uuid

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from m5.data import _find_raw_dir
from m5.lightgbm_pilot import make_features, make_model, predict_nonnegative


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results' / 'tables')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    started = perf_counter()
    config_path = ROOT / 'configs' / 'data.yaml'
    model_path = ROOT / 'configs' / 'models' / 'lightgbm_c03.yaml'
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    sample = yaml.safe_load((ROOT / 'configs' / 'sample.yaml').read_text(encoding='utf-8'))
    params = yaml.safe_load(model_path.read_text(encoding='utf-8'))['parameters']
    if config['splits']['train'] != [1, 1885] or config['splits']['validation'] != [1886, 1913] or config['forecast_horizon'] != 28:
        raise ValueError('C03 requires the agreed Train/Validation split and 28-day horizon')
    # The pilot uses three original files. Do not require unrelated Test/submission
    # files merely to time a Train-only fit. S06 still checks all five separately.
    pilot_config = dict(config, files={key: config['files'][key] for key in
                        ['calendar', 'sell_prices', 'sales_train_validation']})
    directory = _find_raw_dir(pilot_config, args.data_dir)
    # Select only training columns: neither Validation nor Test labels enter memory.
    sales_path = directory / config['files']['sales_train_validation']
    columns = ['id', 'item_id', 'store_id', 'state_id'] + [f'd_{d}' for d in range(1, 1886)]
    sales = pd.read_csv(sales_path, nrows=10, usecols=columns)
    if len(sales) != 10 or sales['id'].duplicated().any():
        raise ValueError('need exactly 10 distinct original M5 series')
    calendar = pd.read_csv(directory / config['files']['calendar'])
    calendar = calendar[calendar['d'].isin([f'd_{d}' for d in range(1, 1914)])].copy()
    keys = pd.MultiIndex.from_frame(sales[['store_id', 'item_id']])
    price_parts = []
    for chunk in pd.read_csv(directory / config['files']['sell_prices'], chunksize=250_000):
        selected = pd.MultiIndex.from_frame(chunk[['store_id', 'item_id']]).isin(keys)
        price_parts.append(chunk.loc[selected])
    prices = pd.concat(price_parts, ignore_index=True)
    if prices.empty:
        raise ValueError('no matching prices; check the input M5 files')
    try:
        colab_available = importlib.util.find_spec('google.colab') is not None
    except ModuleNotFoundError:
        colab_available = False
    runtime = ('colab' if colab_available or 'COLAB_RELEASE_TAG' in os.environ else
               'kaggle' if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ else 'local')
    run_id = f'c03_lightgbm_{runtime}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}'
    rows, forecasts, feature_seconds = [], [], []
    effective = None
    warmup_seconds = 0.0
    for series_index, record in sales.iterrows():
        history = record[[f'd_{d}' for d in range(1, 1886)]].to_numpy(dtype=float)
        series_prices = prices[(prices['item_id'] == record['item_id']) & (prices['store_id'] == record['store_id'])]
        tick = perf_counter()
        x, y, future = make_features(history, calendar, series_prices, state_id=record['state_id'])
        feature_seconds.append(perf_counter() - tick)
        if series_index == 0:
            tick = perf_counter()
            warm = make_model(params)
            warm.fit(x, y)
            predict_nonnegative(warm, future)
            warmup_seconds = perf_counter() - tick
        reference = None
        for repeat in range(1, args.repeats + 1):
            tick = perf_counter()
            model = make_model(params)
            model.fit(x, y)
            fit_seconds = perf_counter() - tick
            tick = perf_counter()
            prediction = predict_nonnegative(model, future)
            predict_seconds = perf_counter() - tick
            if reference is not None:
                np.testing.assert_allclose(prediction, reference, rtol=1e-10, atol=1e-10)
            reference = prediction.copy()
            rows.append(dict(run_id=run_id, series_id=record['id'], repeat=repeat,
                             train_rows=len(y), feature_count=x.shape[1],
                             fit_seconds=fit_seconds, predict_seconds=predict_seconds,
                             total_seconds=fit_seconds + predict_seconds))
        effective = model.get_params()
        for day, value in zip(range(1886, 1914), reference):
            forecasts.append(dict(run_id=run_id, series_id=record['id'], day=f'd_{day}', prediction=float(value)))
        print(f'{series_index + 1}/10: {record["id"]}', flush=True)
    timings = pd.DataFrame(rows)
    means = timings.groupby('series_id', sort=False)[['fit_seconds', 'predict_seconds', 'total_seconds']].mean()
    means.columns = ['mean_' + column for column in means.columns]
    means['feature_seconds_once'] = feature_seconds
    means['run_id'] = run_id
    seconds = float(means['mean_total_seconds'].mean())
    try:
        commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    files = [Path(__file__), ROOT / 'src/m5/lightgbm_pilot.py', ROOT / 'src/m5/data.py', config_path, model_path, ROOT / 'configs/sample.yaml']
    summary = dict(task='C03', run_id=run_id, status='measured_pending_review', runtime=runtime,
        recorded_at=datetime.now(timezone(timedelta(hours=7))).isoformat(),
        python=platform.python_version(), packages={p: importlib.metadata.version(p) for p in ['lightgbm','numpy','pandas','scikit-learn','PyYAML']},
        os=platform.platform(), processor=platform.processor(), logical_cpu_count=os.cpu_count(),
        git_commit=commit, source_sha256={str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in files},
        input_subset_sha256={name:hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest() for name,frame in [('sales_train_only',sales),('calendar_through_validation',calendar),('selected_weekly_prices',prices)]},
        source='https://www.kaggle.com/competitions/m5-forecasting-accuracy/data',
        series_selection='first 10 rows of original sales_train_validation.csv, comparable with C04 pilot; not D02 stratified sample',
        series_ids=sales['id'].tolist(), n_series=10, repeats=args.repeats,
        train_range=[1,1885], effective_training_target_range=[56,1885], forecast_range=[1886,1913],
        horizon=28, validation_labels_used=False, test_labels_used=False,
        model_strategy='one independent model per series; no global-model scalability claim',
        model_parameters=effective, prediction_postprocessing='clip negative predictions to zero; no rounding',
        timing_scope='fit includes estimator construction and dataset conversion; predict includes validation and clipping; excludes feature preparation, CSV I/O, warm-up, checks and output writes',
        warmup='one fit/predict on the first selected series, excluded from repeat timings',
        warmup_seconds=warmup_seconds, feature_seconds_total=sum(feature_seconds),
        mean_fit_predict_seconds_per_series=seconds,
        median_fit_predict_seconds_per_series=float(means['mean_total_seconds'].median()),
        configured_sample_n_series=int(sample['n_series']),
        estimated_sample_fit_predict_seconds=seconds * int(sample['n_series']),
        estimate_caveat='linear extrapolation for this per-series CPU configuration only; excludes I/O/features; first 10 series are not representative; not a measured full-sample or global-model runtime',
        sample_decision='unchanged; C05 must combine evidence from all models',
        wall_seconds_before_export=perf_counter()-started,
        reviewer='Thanh Tiền', approval='pending; do not auto-mark Xong')
    output = args.output_dir / run_id
    output.mkdir(parents=True, exist_ok=False)
    timings.to_csv(output / 'runtime_repeats.csv', index=False)
    means.to_csv(output / 'runtime_by_series.csv', index_label='series_id')
    pd.DataFrame(forecasts).to_csv(output / 'validation_forecasts.csv', index=False)
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(means.to_string())
    print(f'OUTPUT_DIR={output}')


if __name__ == '__main__':
    main()
