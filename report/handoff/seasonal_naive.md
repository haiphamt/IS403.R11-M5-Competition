# Seasonal Naive — chủ mô hình: Tuấn Hải

Bản nháp do AI soạn từ bảng/hình thật (F09); chủ mô hình cần đọc lại, sửa và xác nhận. Khuôn 4 ý.

1. **Mạnh hơn ai, ở độ đo/nhóm nào:** Mô hình đối chứng: không mô hình nào thua nó về macro-RMSSE (Test 0,997; Validation 0,978); sáu mô hình còn lại tốt hơn khoảng 22–25%.
2. **Yếu ở đâu:** Yếu ở mọi nhóm: RMSSE Test 0,900 / 1,004 / 1,034 theo nhóm ít / vừa / nhiều số 0 (xấu nhất ở nhóm gián đoạn mạnh) và sai số tăng từ tuần 1 đến tuần 3 rồi giảm nhẹ ở tuần 4 (MAE trung bình theo tuần 1,117; 1,158; 1,198; 1,155; MAE 1,088 ở horizon 1 và 1,367 ở horizon 28); thứ Bảy, Chủ nhật sai nhiều nhất (MAE 1,294; 1,328).
3. **Vì sao (cơ chế):** Seasonal Naive dự báo 28 ngày bằng cách lặp lại doanh số của 7 ngày cuối. Nếu tuần cuối có nhiều ngày bán bằng 0 hoặc có doanh số bất thường, dự báo sẽ lặp lại mẫu hình đó. Macro-RMSSE của mô hình trên tập Test là 0,997.
4. **Bằng chứng:** `results/tables/test/test_overall.csv`, `test_by_demand_class.csv`, `test_mae_by_horizon.csv`, `test_mae_by_weekday.csv`; hình `results/figures/test/test_rmsse_by_demand_class.png`; run `tes-20261115T090359Z-4cc19f7`.
