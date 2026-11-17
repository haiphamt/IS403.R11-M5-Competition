# IS403.R11 – M5 Competition (Nhóm 6)

Đồ án môn **Phân tích dữ liệu kinh doanh**, lớp **IS403.R11**.
So sánh độ chính xác và thời gian chạy của các mô hình dự báo số lượng bán
theo ngày cho từng sản phẩm tại từng cửa hàng trong **28 ngày tiếp theo**.

## Phạm vi thực nghiệm

- Dữ liệu: M5 Forecasting – Accuracy của Walmart.
- Mẫu chính thức: **1.200 chuỗi**, chọn theo cửa hàng và ngành hàng, seed **42**
  (cấu hình đã khóa tại tag `config-freeze`, xem `configs/freeze.yaml`).
- Mô hình lõi: Seasonal Naive, Linear Regression, ETS, Croston,
  LightGBM Default, XGBoost và CatBoost.
- Phần tùy chọn (T01 Optuna, T02 Stacking, X01 Croston SBA, X02 ablation LightGBM): mã nguồn đã có (PR #24);
  **Validation** của T01, X01, X02 đã chạy (`results/tuning/`, `results/runs/val-20261116T015146Z-2ee03bd`);
  Stacking cũng đã chạy trên Validation (macro-RMSSE 0,7519, kém LightGBM Tuned 0,7447 nhưng RMSE gộp thấp nhất; dự báo ngoài mẫu ở `results/oof/`);
  **Test của phần tùy chọn chưa chạy** (cần tag `config-freeze-v2`, chưa tạo).
- Độ đo theo checklist: **MAE, RMSE, RMSSE, R²**, kèm thời gian chạy.
  Phạm vi này được phân biệt với WRMSSE của cuộc thi M5 gốc.

## Kết quả chính (Test d_1914–d_1941, 1.200 chuỗi, chạy một lần sau `config-freeze`)

| Mô hình | RMSSE (macro) | MAE (macro) | fit+predict (s) |
| --- | ---: | ---: | ---: |
| Seasonal Naive | 0,997 | 1,157 | 0,0 |
| Linear Regression | 0,776 | 0,980 | 25,2 |
| ETS | 0,752 | 0,953 | 1026,2 |
| Croston | 0,764 | 0,986 | 1,4 |
| LightGBM | 0,768 | 0,984 | 9,7 |
| XGBoost | 0,768 | 0,987 | 15,7 |
| CatBoost | 0,768 | 0,984 | 2080,3 |

Thời gian: mô hình cục bộ là tổng thời gian huấn luyện và dự báo của từng chuỗi (3 tiến trình), mô hình toàn cục là thời gian
huấn luyện và dự báo trên toàn mẫu (CPU 4 lõi trên Kaggle), không so trực tiếp. Chênh lệch giữa nhóm đầu rất nhỏ và chỉ dựa
trên một cửa sổ 28 ngày. Về chi phí chạy, Croston đạt RMSSE trung bình 0,764 trong khoảng 1,4 giây, gần bằng ETS (0,752)
nhưng nhanh hơn khoảng 750 lần (1.026,2 giây so với 1,4 giây); với hàng bán không đều, đây là lựa chọn nên thử trước các mô hình cây.

- Báo cáo: [`report/draft_report.md`](report/draft_report.md); slide: `report/slides/m5_nhom6.pptx`; hỏi đáp: `report/qa.md`.
- Dự báo và metrics theo `run_id`: `results/runs/`; bảng/hình: `results/tables/{validation,test}/`, `results/figures/`.
- Smoke reproduction: `results/tables/f10_smoke_reproduction.md` (tập con, không phải tái lập toàn bộ).
- Giới hạn: một cửa sổ Test, mẫu 1.200 chuỗi, giả định giá tương lai, khác biệt input giữa mô hình cục bộ và global,
  chưa đo chi phí tồn kho. Bản thảo và slide do AI soạn từ kết quả thật; chưa có thành viên nào review.

## Cấu trúc thư mục

```text
.
├── configs/
│   ├── data.yaml       # Tên file, đường dẫn và khoảng Train/Validation/Test
│   ├── sample.yaml     # Mẫu 1.200 chuỗi, seed 42 (đã khóa, xem configs/freeze.yaml)
│   ├── features.yaml   # Đặc trưng lõi, lùi ít nhất 28 ngày
│   ├── freeze.yaml     # Cấu hình đã khóa trước khi mở nhãn Test (tag config-freeze)
│   └── models/         # Cấu hình các mô hình (kể cả lightgbm_tuned.json từ Optuna)
├── data/
│   ├── raw/            # Dữ liệu gốc, không đưa lên Git
│   ├── processed/      # Dữ liệu Parquet, không đưa lên Git
│   └── sample_series_ids.csv  # Danh sách 1.200 chuỗi của mẫu chính thức
├── notebooks/          # Notebook Colab/Kaggle: 02/03 (lõi), 04/05 (tùy chọn), EDA
├── paper/              # Tóm tắt bài báo: summary.md (S04)
├── report/
│   ├── draft_report.md # Bản thảo báo cáo
│   ├── qa.md           # Hỏi đáp bảo vệ
│   ├── slides/         # Bộ slide (m5_nhom6.pptx)
│   ├── figures/        # Hình cho phần phương pháp
│   └── handoff/        # Giải thích kết quả của từng chủ mô hình
├── results/
│   ├── runs/           # Dự báo và chỉ số theo run_id
│   ├── tuning/         # Optuna (T01): trials, best_params, tuning_summary
│   ├── figures/        # Hình phân tích kết quả
│   └── tables/         # Bảng độ đo và thời gian chạy
├── scripts/            # run_experiment, collect_runs, analyze_results, build_report, build_slides, tune_lightgbm, backtest_oof, run_stacking
├── src/
│   └── m5/             # data, panel, features, metrics, univariate, global_models, optional, evaluate, schema
├── tests/              # Kiểm thử đơn vị (metrics, chống rò rỉ, mô hình, phần tùy chọn)
├── .gitignore
└── requirements.txt
```

## Bắt đầu

```bash
git clone https://github.com/haiphamt/IS403.R11-M5-Competition.git
cd IS403.R11-M5-Competition
python -m pip install -r requirements.txt
```

Danh sách thư viện ban đầu phục vụ S01. Nhóm cần ghi nhận và chốt phiên bản
sau khi kiểm tra môi trường Colab/Kaggle thực tế.

Tải dữ liệu từ [Kaggle M5 Forecasting – Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data)
và đặt các file gốc vào `data/raw/`. Giữ nguyên tên file tải về.

| Tên file gốc | Vai trò |
| --- | --- |
| `calendar.csv` | Lịch, sự kiện và SNAP |
| `sell_prices.csv` | Giá theo sản phẩm, cửa hàng và tuần |
| `sales_train_validation.csv` | Doanh số `d_1`–`d_1913` |
| `sales_train_evaluation.csv` | Doanh số `d_1`–`d_1941` |
| `sample_submission.csv` | Mẫu định dạng dự báo 28 ngày của cuộc thi |

`sample_submission.csv` không chứa nhãn Test. Không ghép chồng hai file
`sales_train_*`: chúng có phần lịch sử trùng nhau. D03 dùng
`sales_train_evaluation.csv` để chia theo các khoảng ngày dưới đây.

## Quy tắc dữ liệu và đánh giá

| Tập dữ liệu | Khoảng ngày |
| --- | --- |
| Train | `d_1` đến `d_1885` |
| Validation | `d_1886` đến `d_1913` |
| Test | `d_1914` đến `d_1941` |

- Xuất riêng `train.parquet`, `validation.parquet`, `test.parquet` ở bước D03.
- Chỉ các lần chạy thử 10 chuỗi C01–C04 và C06 được đọc dữ liệu gốc.
  Các lần chạy mô hình chính thức dùng dữ liệu đã xử lý được bàn giao.
- Dùng Validation để chọn cấu hình. Chỉ chạy Test sau khi có tag
  `config-freeze`; không dùng Test để chọn tham số.
- Các đặc trưng trễ cho dự báo 28 ngày phải tránh sử dụng giá trị tương lai.
  Giá thiếu trước ngày bán đầu tiên được giữ NaN và đánh dấu `price_missing`.
- Mỗi lần chạy chính thức cần lưu cấu hình, mã nhận diện `run_id`,
  kết quả dự báo, độ đo và thời gian chạy.

## Bàn giao công việc

- **S01:** bộ khung repo, README, `.gitignore` và `requirements.txt`.
- **S03:** đã có ba tệp YAML; cỡ mẫu 1.200 chuỗi đã chốt sau C05 và khóa trong `configs/freeze.yaml`.
- **S04:** Thanh Tiền đưa bản tóm tắt bài báo vào `paper/summary.md`.
- **S05:** bộ đọc tại `src/m5/data.py`; kiểm tra trên dữ liệu thật và
  môi trường được giao bằng các ô ở mục 7.2 hoặc 7.3.

Các thư mục còn `.gitkeep` là vị trí bàn giao cho những bước sau.

## 7. Đọc dữ liệu và kiểm tra môi trường

### 7.1. Máy cá nhân

Chạy từ thư mục repo sau khi cài thư viện:

```bash
python src/m5/data.py
```

Bộ đọc ưu tiên đường dẫn truyền bằng `--data-dir`, sau đó đến biến môi trường
`M5_DATA_DIR`, rồi `data/raw/` của repo và các thư mục con của `/kaggle/input`.
Nếu có nhiều bộ M5 trong Kaggle Input, truyền rõ đường dẫn để tránh đọc nhầm.
Ví dụ:

```bash
python src/m5/data.py --data-dir "D:/datasets/M5"
python src/m5/data.py --data-dir "D:/datasets/M5" --nrows 10
```

`--nrows 10` chỉ đọc tối đa 10 dòng **mỗi file** để kiểm tra nhanh việc đọc CSV.
Không dùng chế độ này làm bằng chứng đọc toàn bộ file ở S06 hoặc đo thời gian
huấn luyện 10 chuỗi ở C01–C04/C06.

Kiểm tra mã bằng bộ CSV giả lập nhỏ, không cần tải dữ liệu M5:

```bash
python -m unittest discover -s tests -v
```

### 7.2. Google Colab

Tải và giải nén đủ 5 file gốc vào một thư mục Google Drive của bạn.
Trong notebook Colab mới, chạy ô chuẩn bị:

```python
from google.colab import drive
drive.mount("/content/drive")

!git clone https://github.com/haiphamt/IS403.R11-M5-Competition.git /content/IS403.R11-M5-Competition
%cd /content/IS403.R11-M5-Competition
!python -m pip install -q pandas PyYAML
```

Chạy ô đọc dữ liệu; sửa `DATA_DIR` đúng với thư mục bạn đã giải nén:

```python
import sys
sys.path.insert(0, "src")
from m5.data import load_raw_data

DATA_DIR = "/content/drive/MyDrive/M5"
raw = load_raw_data(DATA_DIR)
for name, table in raw.items():
    print(f"{name}: {table.shape[0]:,} rows x {table.shape[1]:,} columns")
```

Nếu repo đã tồn tại trong phiên notebook, dùng `git pull --ff-only` trong thư
mục repo để cập nhật thay vì chạy lại ô clone.

### 7.3. Kaggle Notebook

Trong notebook của bạn, thêm bộ dữ liệu cuộc thi **M5 Forecasting – Accuracy**
vào **Input** và bật Internet để lấy mã nguồn. Chạy ô chuẩn bị:

```python
!git clone https://github.com/haiphamt/IS403.R11-M5-Competition.git /kaggle/working/IS403.R11-M5-Competition
%cd /kaggle/working/IS403.R11-M5-Competition
!python -m pip install -q pandas PyYAML
```

Chạy ô đọc dữ liệu; bộ đọc tự tìm đủ 5 file dưới `/kaggle/input`:

```python
import sys
sys.path.insert(0, "src")
from m5.data import load_raw_data

raw = load_raw_data()
for name, table in raw.items():
    print(f"{name}: {table.shape[0]:,} rows x {table.shape[1]:,} columns")
```

Không cần GPU để đọc CSV. Bộ đọc giữ nguyên giá thiếu và không thực hiện
chọn mẫu, chia tập, tạo đặc trưng hay huấn luyện mô hình.

Có thể dùng notebook chung
[`00_data_check_and_seasonal_naive.ipynb`](notebooks/00_data_check_and_seasonal_naive.ipynb)
để ghi số dòng/cột, thông tin môi trường và kết quả C04 thành một ZIP nhỏ.
[Mở notebook bằng Google Colab](https://colab.research.google.com/github/haiphamt/IS403.R11-M5-Competition/blob/main/notebooks/00_data_check_and_seasonal_naive.ipynb).
Nếu nút chọn tệp trong ô kết quả không hoạt động, dùng **Files → Upload to
session storage** của Colab để tải ZIP vào `/content` trước khi chạy ô nạp
dữ liệu. Notebook tự nhận một ZIP M5 hợp lệ đã có trong phiên.

## 8. Chạy thử Seasonal Naive (C04)

```bash
python scripts/benchmark_seasonal_naive.py
```

Mã mô hình ở `src/m5/seasonal_naive.py`: lặp lại 7 ngày cuối của lịch sử
để dự báo 28 ngày. Công thức tham khảo:
[Seasonal naïve, Forecasting: Principles and Practice](https://otexts.com/fpp3/toolbox.html#seasonal-na%C3%AFve-method).

Lần thử dùng 10 chuỗi đầu từ dữ liệu gốc, chỉ lấy `d_1`–`d_1885` làm lịch sử.
Không đọc nhãn Test để chọn cấu hình hoặc tính sai số. Danh sách 10 chuỗi
này phục vụ đo thời gian, không thay cho mẫu phân tầng D02.

Mỗi chuỗi được làm nóng một lần rồi đo trung bình 1.000 lần gọi hàm dự báo.
File `results/tables/c04_seasonal_naive_runtime_<môi trường>.csv` ghi thời gian
từng chuỗi; file JSON cùng tên `summary` ghi môi trường và ước lượng cho
1.200 chuỗi. Số đo chỉ gồm gọi hàm dự báo và kiểm tra đầu vào, không gồm
đọc CSV, lưu file hay toàn bộ pipeline.

(Lịch sử C04) Sau C05, mẫu 1.200 chuỗi đã được chốt; xem `configs/freeze.yaml`.

## 9. Pilot thời gian LightGBM (C03)

Chạy tuần tự [`01_lightgbm_c03.ipynb`](notebooks/01_lightgbm_c03.ipynb)
trên Colab CPU. Notebook lấy mã từ nhánh `thuytien`, tải ba CSV nguồn đã chỉ
rõ và giữ nguyên file gốc. Với bản Drive thiếu khóa `d`/`id`, notebook kiểm tra
chuỗi ngày liên tục và ID item/store rồi tạo bản đầu vào riêng; phép chuyển đổi
và hash được ghi trong metadata. Không đưa các CSV lớn vào Git.

```bash
python scripts/benchmark_lightgbm.py --data-dir /duong/dan/M5 --repeats 3
```

Pilot fit riêng 10 chuỗi đầu, chỉ dùng doanh số `d_1`–`d_1885` để dự báo 28 ngày.
`src/m5/lightgbm_pilot.py` tạo lag 28/35/42, rolling 7/28 đã shift 28,
lịch/sự kiện/SNAP/giá; giữ giá thiếu NaN kèm cờ. Dùng thông tin lịch/giá đã biết
theo protocol. Tham số học mặc định và điều khiển CPU/seed nằm trong
`configs/models/lightgbm_c03.yaml`. Không tuning, không dùng nhãn Validation/Test.

Run Colab ngày 12/11/2026 tại
[`results/tables/c03_lightgbm_colab_20261112T164342Z_82d16d27/`](results/tables/c03_lightgbm_colab_20261112T164342Z_82d16d27):
30 dòng đo, bảng trung bình 10 chuỗi, summary và metadata nguồn/môi trường.
Hình ở [`results/figures/`](results/figures/c03_lightgbm_colab_20261112T164342Z_82d16d27/runtime_by_series.png).
Thời gian fit + predict trung bình **0,1885 giây/chuỗi**; ngoại suy 1.200 chuỗi
**226,20 giây**, không gồm I/O, tạo feature, warm-up và ghi file.
Đây là pilot thời gian, không phải lần chạy Validation M04 hay thời gian đo
toàn mẫu. (Lịch sử C03) Mẫu 1.200 chuỗi sau đó đã được chốt ở C05.

Kiểm thử: `python -m unittest discover -s tests -v`.
Các kiểm thử dùng CSV giả chỉ kiểm tra mã, không thay kết quả M5 thật.

## 9a. Chạy thử ETS và Croston (C01)

```bash
python scripts/benchmark_ets_croston.py
```

Mã mô hình ở `src/m5/ets_croston.py`, dùng `statsforecast`: `AutoETS` chọn
cấu hình ETS theo AICc, `CrostonClassic` cho nhu cầu gián đoạn; cả hai chu kỳ
7 ngày, dự báo 28 ngày, không cắt giá trị âm về 0.

Cách chọn chuỗi và phạm vi dữ liệu giống mục 8. Mỗi chuỗi được làm nóng một
lần rồi đo trung bình `--repeats` lần (mặc định 3) gồm huấn luyện và dự báo.
File `results/tables/c01_ets_croston_runtime_<môi trường>.csv` ghi thời gian
từng chuỗi và mô hình; file JSON `summary` ghi môi trường, phiên bản
`statsforecast` và ước lượng cho 1.200 chuỗi, một lượt và Validation + Test.
Notebook chạy trên Colab/Kaggle:
[`01_ets_croston_pilot.ipynb`](notebooks/01_ets_croston_pilot.ipynb).

## 10. Ghi nhận kết quả kiểm tra (S06)

Mỗi người lưu liên kết notebook, ngày chạy, nền tảng, phiên bản Python/pandas,
CPU/RAM và toàn bộ bảng số dòng/số cột vào bằng chứng của task S06.
Khi dùng bộ Kaggle đầy đủ, đối chiếu:

| Bảng | Số dòng | Số cột |
| --- | ---: | ---: |
| calendar | 1.969 | 14 |
| sell_prices | 6.841.121 | 4 |
| sales_train_validation | 30.490 | 1.919 |
| sales_train_evaluation | 30.490 | 1.947 |
| sample_submission | 60.980 | 29 |

Ghi lỗi đường dẫn, file thiếu hoặc sai cột tại đây hoặc trong ghi chú task
để cả nhóm sửa chung. Việc chạy thành công các kiểm thử với CSV giả lập
không thay thế bằng chứng chạy trên dữ liệu thật ở Colab/Kaggle.

## 11. Pipeline thí nghiệm đầy đủ (7 mô hình lõi)

Môi trường của các lần chạy chính: Python 3.13 (CPU trên Kaggle), cài bằng `pip install -r requirements.txt`
(trên máy có proxy TLS dùng thêm `--use-feature=truststore`). Phiên bản thực dùng
của mỗi run được ghi trong `environment.json`. Giữ đúng bản ghim: `xgboost==3.4.1` (cần Python ≥ 3.12) cho dự báo khác với 3.2.0 (macro-RMSSE Validation 0,7500 so với 0,7493), nên tái lập chính xác XGBoost cần đúng phiên bản.

```bash
python scripts/prepare_sample_split.py                    # D02/D03/D07: mẫu 1.200 chuỗi, split parquet, báo cáo dữ liệu
python scripts/run_experiment.py --split validation       # origin d_1885, 7 mô hình -> results/runs/<run_id>/
python scripts/collect_runs.py --split validation --runs results/runs/<run_a> results/runs/<run_b> --out results/tables/validation
python scripts/analyze_results.py --split validation --tables results/tables/validation --figures results/figures/validation
# sau khi chốt cấu hình và tạo tag config-freeze:
python scripts/run_experiment.py --split test             # origin d_1913, refit Train+Validation
```

- Mẫu và manifest: `data/sample_series_ids.csv` (seed 42, 40 chuỗi x 30 tầng, hash trong
  `results/tables/d07_data_quality_report.json`).
- Dò ngược `git_sha` trong `environment.json` của các run cũ (20b22f7, 4cc19f7, 4994dda không còn trong lịch sử đã viết lại):
  xem `results/pre_rewrite_sha_map.csv`: mỗi SHA cũ có commit tương đương trong lịch sử hiện tại với cùng tree hash
  (kiểm: `git rev-parse <commit>^{tree}` bằng cột `tree_hash_of_old_sha`). Chỉ `4994dda` còn thẻ `archive/pre-rewrite-4994dda` (kiểm tra: `git diff --stat archive/pre-rewrite-4994dda config-freeze -- src scripts/run_experiment.py`,
  không có dòng nào = giống hệt). Thẻ của `20b22f7` và `4cc19f7` đã gỡ vì lịch sử cũ có dòng đồng tác giả AI; mã của hai SHA này
  đã kiểm giống bản khóa trước khi gỡ (ghi trong bảng map).
- Trường `git_dirty: true` trong `environment.json` của 4 run lõi chỉ do thư mục `results/<run_id>` được tạo trước khi kiểm tra
  (không phải mã bị sửa dở); từ run `val-20261116T015146Z-2ee03bd` trường này bỏ qua `results/`.
- Mô hình cục bộ theo chuỗi: Seasonal Naive, ETS (AutoETS, chu kỳ 7, AICc), Croston
  (`src/m5/univariate.py`). Mô hình global trên toàn mẫu: Linear Regression, LightGBM, XGBoost,
  CatBoost (`src/m5/global_models.py`) với lag 28/35/42, rolling mean/std 7/28 sau shift 28, lịch,
  SNAP theo bang, giá tuần (giữ NaN + `price_missing`); `item_id` không dùng làm biến phân loại.
- Tham số "Default" là mặc định thư viện; chỉ đặt seed, số luồng, `enable_categorical`
  (XGBoost) và `cat_features` (CatBoost). Tham số lưu ở `model_parameters.json`; riêng XGBoost ở các run cũ, giá trị `None` nghĩa là dùng mặc định của thư viện
  (100 cây, `eta` 0,3, `max_depth` 6); run mới ghi thêm cấu hình thực của booster.
- Mọi dự báo bị cắt tại 0 đồng loạt (không làm tròn); số dự báo âm trước khi cắt ghi trong
  `timings.json`. Chuỗi lỗi được fallback và ghi ở `diag_fallbacks.csv`, không đổi tên mô hình.
- Split `test` bị chặn nếu chưa có tag `config-freeze`.
- Resume: chạy lại cùng lệnh với `--run-id <id>`; ETS/Croston lưu checkpoint 50 chuỗi trong `ckpt/`
  (Git bỏ qua `*.pkl`). Chạy trên Kaggle/Colab bằng
  [`notebooks/02_m5_full_run.ipynb`](notebooks/02_m5_full_run.ipynb) (6 mô hình nhanh) và
  [`notebooks/02b_m5_catboost_run.ipynb`](notebooks/02b_m5_catboost_run.ipynb) (CatBoost, chậm trên CPU). Test dùng `03_m5_test_run.ipynb` và `03b_m5_catboost_test_run.ipynb`; hai notebook này clone đúng tag `config-freeze` (mã đã khóa) chứ không phải đầu nhánh. Cả bốn notebook in cảnh báo nếu phiên bản thư viện khác `requirements.txt`, và ô cuối chỉ lấy run vừa tạo của split đang chạy.
- Thời gian trong bảng: mô hình cục bộ = tổng fit+predict từng chuỗi (một luồng mỗi chuỗi);
  mô hình global = fit+predict toàn mẫu nhiều luồng; không dùng để so trực tiếp nếu khác phần cứng.

### Phần tùy chọn (T01, T02, X01, X02)

Thứ tự chạy (Validation trước, Test sau và chỉ sau khi có tag `config-freeze-v2`). Các notebook clone nhánh `m5-end-to-end`, nên
PR chứa `configs/models/lightgbm_tuned.json` (PR #25) phải được merge vào nhánh này trước khi chạy `05_m5_optional_test.ipynb`.

1. `notebooks/04_m5_optional_validation.ipynb` (Kaggle): Optuna, Croston SBA, ablation, dự báo ngoài mẫu (OOF) của LR/LightGBM/XGBoost;
   `04b_m5_optional_catboost_oof.ipynb`: OOF của CatBoost.
2. Chép `results/tuning/best_params.json` thành `configs/models/lightgbm_tuned.json` (đã có trong kho) trước khi gắn tag.
3. `python scripts/run_stacking.py --split validation --oof results/oof/*.parquet --lgbm tuned --base-runs <các run Validation>`.
4. Chỉ khi nhóm chốt cấu hình: tạo tag `config-freeze-v2` (cập nhật `configs/freeze.yaml`), rồi chạy `05_m5_optional_test.ipynb`
   và `run_stacking.py --split test`. Nhãn Test của 7 mô hình lõi đã mở trước đó nên Test này không còn là dữ liệu giữ lại hoàn toàn.


## Tài nguyên gốc

- [Cuộc thi M5 Forecasting – Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy)
- [Dữ liệu, benchmark và mã nguồn M5 methods](https://github.com/Mcompetitions/M5-methods)

Repo chứa mã nguồn và các kết quả nhỏ cần cho báo cáo. Dữ liệu gốc,
dữ liệu trung gian dung lượng lớn, môi trường ảo và thông tin đăng nhập
được loại khỏi Git bằng `.gitignore`.
