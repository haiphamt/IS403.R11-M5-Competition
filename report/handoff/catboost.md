# CatBoost (mặc định thư viện, khai báo biến phân loại) — chủ mô hình: Thanh Tiền

Bản nháp do AI soạn từ bảng/hình thật (F09); chủ mô hình cần đọc lại, sửa và xác nhận. Khuôn 4 ý.

1. **Mạnh hơn ai, ở độ đo/nhóm nào:** Tốt hơn Seasonal Naive khoảng 23% macro-RMSSE (Test 0,768 so 0,997) và có RMSE gộp Test thấp nhất trong ba mô hình boosting (1,929 so LightGBM 1,939, XGBoost 1,964).
2. **Yếu ở đâu:** Chậm nhất: khoảng 2.080 giây fit trên Kaggle CPU 4 lõi (1.000 vòng, độ sâu 6), gấp ~200 lần LightGBM; ở Validation macro-RMSSE 0,763 là kém nhất nhóm học máy, và không vượt ETS/Croston ở nhóm gián đoạn mạnh (Test 0,804).
3. **Vì sao (cơ chế):** Boosting có thứ tự và thống kê mục tiêu cho biến phân loại; theo thước đo `PredictionValuesChange`, `rolling_mean_28` (28,5%), `rolling_mean_7` (15,6%) và `sell_price` (10,0%) đóng góp nhiều nhất (thước đo này khác `gain` của LightGBM/XGBoost nên không so mức tập trung giữa các mô hình); độ chính xác không tốt hơn dù tốn nhiều thời gian hơn.
4. **Bằng chứng:** `results/runs/tes-20261115T095405Z-4994dda/` (run Test), `val-20261115T073404Z-20b22f7/diag_catboost_importance.csv`, `results/tables/test/test_runtime_tradeoff.csv`; hình `results/figures/m10_feature_importance.png`, `results/figures/test/test_accuracy_runtime.png`.
