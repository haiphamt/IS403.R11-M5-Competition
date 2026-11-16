# Seasonal Naive — chủ mô hình: Tuấn Hải

Bản nháp do AI soạn từ bảng/hình thật (F09); chủ mô hình cần đọc lại, sửa và xác nhận. Khuôn 4 ý.

1. **Mạnh hơn ai, ở độ đo/nhóm nào:** Mô hình đối chứng: không mô hình nào thua nó về macro-RMSSE (Test 0,997; Validation 0,978); sáu mô hình còn lại tốt hơn khoảng 22–25%.
2. **Yếu ở đâu:** Yếu ở mọi nhóm: RMSSE Test 0,900 / 1,004 / 1,034 theo nhóm ít / vừa / nhiều số 0 (xấu nhất ở nhóm gián đoạn mạnh) và sai số tăng theo horizon (MAE 1,088 ở horizon 1 lên 1,367 ở horizon 28); thứ Bảy, Chủ nhật sai nhiều nhất (MAE 1,294; 1,328).
3. **Vì sao (cơ chế):** Lặp lại nguyên tuần cuối nên sao chép luôn nhiễu và các số 0 của tuần đó; với doanh số thấp (nhiều 0/1) phương sai dự báo gần bằng phương sai dữ liệu, nên RMSSE xấp xỉ 1.
4. **Bằng chứng:** `results/tables/test/test_overall.csv`, `test_by_demand_class.csv`, `test_mae_by_horizon.csv`, `test_mae_by_weekday.csv`; hình `results/figures/test/test_rmsse_by_demand_class.png`; run `tes-20261115T090359Z-4cc19f7`.
