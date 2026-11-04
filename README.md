# IS403.R11 – M5 Competition (Nhóm 6)

Đồ án môn **Phân tích dữ liệu kinh doanh**, lớp **IS403.R11**.
So sánh độ chính xác và thời gian chạy của các mô hình dự báo số lượng bán
theo ngày cho từng sản phẩm tại từng cửa hàng trong **28 ngày tiếp theo**.

## Phạm vi thực nghiệm

- Dữ liệu: M5 Forecasting – Accuracy của Walmart.
- Mẫu mặc định: **1.200 chuỗi**, chọn theo cửa hàng và ngành hàng, seed **42**.
  Cỡ mẫu được chốt sau khi nhóm đo thời gian chạy thực tế.
- Mô hình lõi: Seasonal Naive, Linear Regression, ETS, Croston,
  LightGBM Default, XGBoost và CatBoost.
- Phần có thể cắt khi thiếu thời gian: LightGBM Tuned bằng Optuna và Stacking.
- Độ đo theo checklist: **MAE, RMSE, RMSSE, R²**, kèm thời gian chạy.
  Phạm vi này được phân biệt với WRMSSE của cuộc thi M5 gốc.

## Cấu trúc thư mục

```text
.
├── configs/
│   ├── data.yaml       # Tên file, đường dẫn và khoảng Train/Validation/Test
│   ├── sample.yaml     # Mẫu tạm 1.200 chuỗi, chưa chốt C05
│   ├── features.yaml   # Đặc trưng lõi, lùi ít nhất 28 ngày
│   └── models/         # Cấu hình các mô hình
├── data/
│   ├── raw/            # Dữ liệu gốc, không đưa lên Git
│   └── processed/      # Dữ liệu Parquet, không đưa lên Git
├── notebooks/          # Notebook Colab/Kaggle và EDA
├── paper/              # Tóm tắt bài báo: summary.md (S04)
├── report/
│   ├── figures/        # Hình cho phần phương pháp
│   └── handoff/        # Giải thích kết quả của từng chủ mô hình
├── results/
│   ├── figures/        # Hình phân tích kết quả
│   └── tables/         # Bảng độ đo và thời gian chạy
├── src/
│   └── m5/
│       └── data.py     # Bộ đọc 5 file gốc dùng chung (S05)
├── tests/
│   └── test_data.py    # Kiểm tra bộ đọc và tính nhất quán của cấu hình
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
- **S03:** đã có ba file YAML. `sample.yaml` vẫn là cấu hình tạm;
  chỉ chốt cỡ mẫu sau khi có bằng chứng thời gian chạy ở C05.
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

Giữ nguyên mẫu tạm 1.200 chuỗi cho đến khi C05 có số đo của ETS/Croston,
XGBoost/CatBoost, LightGBM và Linear Regression.

## 9. Chạy thử ETS và Croston (C01)

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

## Tài nguyên gốc

- [Cuộc thi M5 Forecasting – Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy)
- [Dữ liệu, benchmark và mã nguồn M5 methods](https://github.com/Mcompetitions/M5-methods)

Repo chứa mã nguồn và các kết quả nhỏ cần cho báo cáo. Dữ liệu gốc,
dữ liệu trung gian dung lượng lớn, môi trường ảo và thông tin đăng nhập
được loại khỏi Git bằng `.gitignore`.
