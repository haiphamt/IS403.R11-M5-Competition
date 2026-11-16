# LightGBM (mặc định thư viện, mô hình global) — chủ mô hình: Thủy Tiên

Bản nháp do AI soạn từ bảng/hình thật (F09); chủ mô hình cần đọc lại, sửa và xác nhận. Khuôn 4 ý.

1. **Mạnh hơn ai, ở độ đo/nhóm nào:** Tốt hơn Seasonal Naive khoảng 23% macro-RMSSE (Test 0,768 so 0,997) và hơn Linear Regression 0,008; ngang XGBoost và CatBoost (cùng 0,768), RMSE gộp Test 1,939 thấp hơn XGBoost (1,964) nhưng cao hơn ETS (1,913).
2. **Yếu ở đâu:** Không vượt ETS và Croston ở nhóm chuỗi gián đoạn mạnh (Test 0,806 so 0,780 và 0,785) và thứ hạng so với ETS không đổi giữa hai cửa sổ (ETS luôn nhỉnh hơn); khi bỏ thông tin giá/sự kiện chưa được đo (ablation X02 chưa có ở bản này).
3. **Vì sao (cơ chế):** Cộng dần cây để giảm phần dư; nhưng gain tập trung vào `rolling_mean_28` (~70%) và `rolling_mean_7` (~16%), tức dự báo chủ yếu là mức doanh số trượt, nên khó vượt một bộ làm mượt tốt như ETS; giá (`sell_price`) và lịch chỉ đóng góp nhỏ.
4. **Bằng chứng:** `results/runs/val-20261115T072831Z-20b22f7/diag_lightgbm_importance.csv`, `results/tables/test/test_overall.csv`, `test_by_demand_class.csv`; hình `results/figures/m10_feature_importance.png`.
