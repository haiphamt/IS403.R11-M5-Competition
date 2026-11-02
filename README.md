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
│   └── m5/             # Mã xử lý dữ liệu, độ đo và mô hình
├── tests/              # Kiểm tra chia dữ liệu, độ đo và rò rỉ dữ liệu
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
- **S03:** bổ sung `configs/data.yaml`, `configs/sample.yaml`,
  `configs/features.yaml`.
- **S04:** Thanh Tiền đưa bản tóm tắt bài báo vào `paper/summary.md`.
- **S05:** bổ sung bộ đọc dữ liệu tại `src/m5/data.py`, chạy được trên
  Colab và Kaggle.

Các thư mục có `.gitkeep` hiện chỉ là vị trí bàn giao, chưa có pipeline
huấn luyện hoàn chỉnh. Cập nhật lệnh chạy tại README khi từng bước được bàn giao.

## Tài nguyên gốc

- [Cuộc thi M5 Forecasting – Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy)
- [Dữ liệu, benchmark và mã nguồn M5 methods](https://github.com/Mcompetitions/M5-methods)

Repo chứa mã nguồn và các kết quả nhỏ cần cho báo cáo. Dữ liệu gốc,
dữ liệu trung gian dung lượng lớn, môi trường ảo và thông tin đăng nhập
được loại khỏi Git bằng `.gitignore`.
