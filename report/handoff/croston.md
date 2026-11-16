# Croston (cổ điển, α = 0,1) — chủ mô hình: Tuấn Vũ

Bản nháp do AI soạn từ bảng/hình thật (F09); chủ mô hình cần đọc lại, sửa và xác nhận. Khuôn 4 ý.

1. **Mạnh hơn ai, ở độ đo/nhóm nào:** Chỉ kém ETS 0,012 macro-RMSSE (Test 0,764 so 0,752) nhưng gần như không tốn thời gian (≈1,4 giây tổng); tốt hơn LightGBM/XGBoost/CatBoost và Linear Regression ở nhóm gián đoạn mạnh (Test 0,785 so 0,804–0,814).
2. **Yếu ở đâu:** Yếu nhất trong nhóm ở nhóm ít số 0 (Test 0,710 so ETS 0,686) và bỏ qua chu kỳ tuần: dự báo là một đường phẳng z/p nên lệch nhiều nhất vào giữa tuần (thứ Ba đến thứ Năm MAE 0,93–0,94, xếp cuối trong sáu mô hình không tính Seasonal Naive, ETS 0,88–0,89); cuối tuần lại xếp thứ hai.
3. **Vì sao (cơ chế):** Tách ước lượng làm mượt của lượng bán khác 0 (z) và của khoảng cách giữa hai lần có nhu cầu (p), dự báo ŷ = z/p; hợp nhu cầu gián đoạn nhưng không có thành phần mùa; ví dụ chuỗi rất thưa có p vài trăm ngày nên ŷ ≈ 0,003–0,010.
4. **Bằng chứng:** `results/runs/val-20261115T072831Z-20b22f7/diag_croston.csv` (z, p theo chuỗi), `results/tables/test/test_overall.csv`, `test_by_demand_class.csv`, `test_mae_by_weekday.csv`; hình `results/figures/test/test_rmsse_by_demand_class.png`.
