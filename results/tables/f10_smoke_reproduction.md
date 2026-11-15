# F10 – Smoke reproduction (môi trường sạch, tập con nhỏ)

**Đây là smoke, không phải tái lập toàn bộ thực nghiệm.** Mục đích: kiểm tra pipeline chạy từ đầu đến cuối sau khi clone mới.

- Ngày chạy: 15/11/2026. Máy: Windows 11, 8 luồng CPU (khác Kaggle 4 lõi).
- Clone mới từ GitHub, nhánh `m5-end-to-end` tại SHA `4994dda`, thư mục tạm không chứa kết quả cũ.
- Venv mới (Python 3.11), cài bằng `pip` với phiên bản mới nhất tương thích (không dùng bản ghim Kaggle vì các phiên bản ghim dành cho Python 3.13):
  numpy 2.4.6, pandas 2.3.3, scikit-learn 1.9.1, statsforecast 2.1.1, lightgbm 4.7.0, xgboost 3.2.0, catboost 1.2.10.
- Bước 1: `python -m unittest discover -s tests` → **42 test OK** (25 giây).
- Bước 2: `python scripts/run_experiment.py --split validation --max-series 12 --workers 2 --chunk 6` (12 chuỗi đầu của manifest, 7 mô hình, origin d_1885, **không dùng nhãn Test**) → run `val-20261115T095419Z-4994dda`, `validate_predictions` đạt, không fallback.

| Mô hình | MAE | RMSE | RMSSE | R² | fit+predict (s) |
|---|---|---|---|---|---|
| Seasonal Naive | 1,631 | 2,246 | 1,315 | −1,179 | 0,002 |
| Linear Regression | 1,345 | 1,763 | 0,982 | −0,220 | 3,33 |
| ETS | 1,226 | 1,682 | 0,937 | −0,048 | 19,69 |
| Croston | 1,249 | 1,689 | 0,939 | −0,054 | 0,037 |
| LightGBM | 1,290 | 1,683 | 0,956 | −0,118 | 1,23 |
| XGBoost | 1,313 | 1,674 | 0,956 | −0,127 | 2,27 |
| CatBoost | 1,298 | 1,701 | 0,959 | −0,124 | 218,81 |

Giới hạn: 12 chuỗi không đại diện cho mẫu 1.200 chuỗi; các con số này không dùng trong báo cáo kết quả; phiên bản thư viện và phần cứng khác với run chính; chưa có người trong nhóm tự chạy lại (M09 vẫn là việc của nhóm).
