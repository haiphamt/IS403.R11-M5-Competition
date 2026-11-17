# XGBoost (mặc định thư viện, enable_categorical) — chủ mô hình: Thanh Tiền

Bản nháp do AI soạn từ bảng/hình thật (F09); chủ mô hình cần đọc lại, sửa và xác nhận. Khuôn 4 ý.

1. **Mạnh hơn ai, ở độ đo/nhóm nào:** Tốt hơn Seasonal Naive khoảng 23% macro-RMSSE (Test 0,768 so 0,997), hơn Linear Regression 0,008, ngang LightGBM và CatBoost; fit nhanh (≈16 giây) và gần như không sinh dự báo âm (22 trên 33.600).
2. **Yếu ở đâu:** RMSE gộp Test cao nhất nhóm học máy (1,964; LightGBM 1,939, CatBoost 1,929) và kém ETS/Croston ở nhóm gián đoạn mạnh (Test 0,804 so 0,780/0,785).
3. **Vì sao (cơ chế):** Boosting cộng cây có chính quy hóa; importance (gain) lệch rất nhiều vào `rolling_mean_28` (~76%) và `rolling_mean_7` (~7%), nên mô hình chủ yếu học lại mức doanh số trượt; các đặc trưng giá, tháng, cửa hàng đóng góp dưới 2% mỗi biến theo gain của run Validation; chưa có ablation cho XGBoost nên không suy ra rằng giá không quan trọng.
4. **Bằng chứng:** `results/runs/val-20261115T072831Z-20b22f7/diag_xgboost_importance.csv`, `results/runs/tes-20261115T090359Z-4cc19f7/`, `results/tables/test/test_overall.csv`; hình `results/figures/m10_feature_importance.png`, `m10_boosting_illustration.png` (sơ đồ minh họa, không phải cây học được).
