# Linear Regression — chủ mô hình: Thùy Trang

Bản nháp do AI soạn từ bảng/hình thật (F09); chủ mô hình cần đọc lại, sửa và xác nhận. Khuôn 4 ý.

1. **Mạnh hơn ai, ở độ đo/nhóm nào:** Tốt hơn Seasonal Naive khoảng 22% macro-RMSSE (Test 0,776 so 0,997) và có MAE macro thấp hơn Croston, LightGBM, XGBoost trên Test (0,980 so 0,984–0,987), nhưng xếp cuối nhóm học/thống kê theo macro-RMSSE.
2. **Yếu ở đâu:** Yếu nhất ở nhóm gián đoạn mạnh (RMSSE Test 0,814 so ETS 0,780) và sinh nhiều dự báo âm nhất (2.032 trên 33.600 điểm Test, 6%, phải cắt về 0).
3. **Vì sao (cơ chế):** Chỉ biểu diễn quan hệ tuyến tính: hệ số (run Validation; biến số đã chuẩn hóa, biến phân loại one-hot, chỉ so sánh được trong cùng mô hình và là liên hệ thống kê, không phải tác động nhân quả) lớn nhất của biến chuẩn hóa là `rolling_mean_28` (1,18) và `rolling_mean_7` (0,67), kèm hệ số âm lớn cho ngày Giáng sinh (−0,82) và các ngày trong tuần; không có tương tác nên không giữ được dự báo không âm cho chuỗi thưa.
4. **Bằng chứng:** `results/runs/val-20261115T072831Z-20b22f7/diag_linear_regression_coefficients.csv`, `results/runs/tes-20261115T090359Z-4cc19f7/`; bảng `test_by_demand_class.csv`; mục 4.2 của báo cáo.
