# ETS (AutoETS, chu kỳ 7, AICc) — chủ mô hình: Tuấn Vũ

Bản nháp do AI soạn từ bảng/hình thật (F09); chủ mô hình cần đọc lại, sửa và xác nhận. Khuôn 4 ý.

1. **Mạnh hơn ai, ở độ đo/nhóm nào:** Đứng đầu macro-RMSSE ở cả Validation (0,729) và Test (0,752), tốt hơn Croston 0,012 và LightGBM/XGBoost/CatBoost 0,016 trên Test; mạnh nhất ở nhóm chuỗi gián đoạn mạnh (Test 0,780 so 0,804–0,814 của các mô hình global) và nhóm ít số 0 (0,686).
2. **Yếu ở đâu:** Tốn thời gian: khoảng 1.026 giây CPU tổng (≈355 giây thực với 3 tiến trình), gấp ~100 lần LightGBM; ở ngày dự báo thứ 14, Linear Regression, LightGBM, XGBoost và Croston thấp hơn (MAE 1,007–1,022 so 1,043) nhưng CatBoost cao hơn (1,062); 474 dự báo âm phải cắt về 0.
3. **Vì sao (cơ chế):** AutoETS chọn toàn cấu trúc sai số cộng; α trung bình 0,055 (trung vị 0,034) nên mức dự báo được làm mượt rất mạnh, phù hợp chuỗi thưa nhiều nhiễu; γ gần 0 nên chu kỳ tuần được giữ gần như cố định; ~41% chuỗi (496/1.200) không chọn thành phần mùa.
4. **Bằng chứng:** `results/runs/val-20261115T072831Z-20b22f7/diag_ets.csv` (cấu trúc, α/β/γ), `results/tables/test/test_overall.csv`, `test_by_demand_class.csv`, `test_runtime_tradeoff.csv`; mục 4.2 của báo cáo.
