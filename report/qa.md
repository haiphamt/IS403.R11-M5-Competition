# Câu hỏi – đáp dự kiến khi bảo vệ (đồ án M5, Nhóm 6)

Các câu trả lời dựa trên cấu hình và mã trong repo. Số liệu cụ thể xem `report/draft_report.md`
và `results/tables/`; các số dưới đây lấy từ bảng nguồn nêu kèm.

**1. Vì sao horizon 28 ngày?**
Đúng với cuộc thi M5 (hai kỳ dự báo 28 ngày) và với chu kỳ bổ sung hàng ngắn hạn của bán lẻ.
Cả bốn mô hình global dự báo trực tiếp cả 28 ngày từ cùng một origin, nên đặc trưng phải lùi ≥ 28 ngày.

**2. Vì sao lag ≥ 28 (28/35/42) và rolling shift 28?**
Ngày dự báo xa nhất là T+28, nên chỉ quan sát đến T mới chắc chắn có. Lag 28 của ngày T+28 là y(T);
lag nhỏ hơn 28 sẽ dùng nhãn chưa biết. `tests/test_no_leakage.py` làm hỏng mọi nhãn sau origin
và kiểm tra feature/target không đổi.

**3. RMSSE khác WRMSSE thế nào?**
RMSSE ở đây tính từng chuỗi: RMSE chia cho sai số naïve một bước (sau leading zeros, trên lịch sử
huấn luyện), rồi lấy trung bình macro trên mẫu. WRMSSE của M5 gộp 42.840 chuỗi ở 12 mức phân cấp,
có trọng số theo doanh thu 28 ngày cuối của Train. Hai số không so trực tiếp với bảng xếp hạng M5.

**4. R² âm / chuỗi hằng nghĩa là gì?**
R² âm nghĩa là dự báo kém hơn việc dùng trung bình của nhãn đánh giá; hợp lệ, không bị ép về 0.
Chuỗi nhãn hằng (SST = 0) không xác định R²; evaluator để NaN, ghi lý do và đếm số chuỗi bị bỏ
(`n_r2_undefined`), không dùng epsilon.

**5. Vì sao ETS và AICc?**
AutoETS chọn trong họ ETS theo tiêu chí thông tin hiệu chỉnh cho mẫu hữu hạn, chu kỳ 7 để bắt mùa tuần.
Kết quả chọn cấu trúc (error/trend/season) và α/β/γ nằm ở `diag_ets.csv`.

**6. Croston: z và p?**
z là ước lượng làm mượt của lượng bán khi có nhu cầu, p là ước lượng làm mượt khoảng cách giữa hai
lần có nhu cầu; dự báo phẳng ŷ = z/p. Cài đặt tham chiếu trong `m5.univariate.croston_components`
được kiểm tra khớp dự báo của statsforecast (smoothing 0,1).

**7. Hệ số Linear Regression nói gì?**
Hệ số của biến đã chuẩn hóa (điền thiếu bằng trung vị rồi chuẩn hóa, khớp trên dữ liệu huấn luyện trước ngày dự báo) cho thấy liên hệ
điều kiện giữa đặc trưng và doanh số; biến phân loại one-hot bỏ nhóm gốc. Không phải tác động nhân quả.

**8. Boosting cộng cây như thế nào?**
F_m(x) = F_{m-1}(x) + η f_m(x): mỗi cây mới học phần sai số còn lại của mô hình hiện tại. Sơ đồ minh họa
khác với cây thật học được; importance thật nằm ở `diag_*_importance.csv`.

**9. Default so với Tuned?**
Core dùng tham số mặc định thư viện (chỉ đặt seed/luồng/khai báo categorical). Optuna (30 lượt, chỉ nhãn Validation) cho
macro-RMSSE Validation 0,7447 so với 0,7496 của Default; chênh lệch đo trên chính cửa sổ chọn nên hơi lạc quan. Trên Test (chạy 18/11/2026)
Tuned đạt 0,7700, không hơn Default (0,7680; chênh +0,0020, khoảng tin cậy [−0,0009; +0,0054] chứa 0): lợi thế trên Validation biến mất.

**10. Giá tương lai?**
Pipeline dùng giá tuần của ngày cần dự báo (bảng giá M5 công bố đến d_1969). Trong vận hành giá tương lai
chỉ dùng được nếu được lên kế hoạch trước; đây là giả định phải nêu, không phải kết luận nhân quả về giá.

**11. Vì sao chọn mẫu 1.200 chuỗi (seed 42, 40 × 30 tầng)?**
Phân tầng store × category để giữ cấu trúc; thống kê đại diện chỉ dùng Train
(`results/tables/d02_sample_representativeness.csv`). Mẫu không chọn lại sau khi thấy Test.

**12. Local hay global?**
Seasonal Naive/ETS/Croston là local (một mô hình mỗi chuỗi, chỉ dùng lịch sử chuỗi đó). Linear Regression,
LightGBM, XGBoost, CatBoost là global (một mô hình cho cả mẫu, dùng thêm giá/lịch/SNAP/nhóm).
Khác biệt input là một yếu tố gây nhiễu khi so thứ hạng; pilot LightGBM theo chuỗi (C03) không dùng để suy ra thời gian global.

**13. Stacking có rò rỉ không?**
Chỉ hợp lệ khi dự báo đưa vào meta là ngoài mẫu của mô hình cơ sở. Không fit meta hay chọn alpha bằng Test;
score trên chính tập huấn luyện meta không phải đánh giá độc lập. Trạng thái: trên Validation stacking đạt macro-RMSSE 0,7519 (kém LightGBM Tuned 0,7447 và ETS 0,7293, nhưng RMSE gộp thấp nhất 2,0251); trên Test (18/11/2026) macro-RMSSE 0,7689, ngang LightGBM (0,7680) và kém ETS (0,7520), RMSE gộp 1,928 thấp nhất trong các mô hình global nhưng cao hơn ETS (1,913).

**14. Vì sao mô hình thắng/thua?**
Trên tập kiểm tra, ETS chính xác nhất (RMSSE trung bình 0,752). Croston gần bằng (0,764) nhưng chỉ chạy khoảng 1,4 giây, so với
khoảng 1.026 giây của ETS; ba mô hình LightGBM, XGBoost, CatBoost ngang nhau (0,768).
Với hàng bán chậm (trên 80% số ngày không bán), ETS và Croston (0,780 và 0,785) tốt hơn ba mô hình cây (0,804–0,806), nên với
nhóm hàng này dùng mô hình làm mượt đơn giản là đủ và tốn ít tài nguyên hơn.
Phần liên hệ tồn kho mới là gợi ý vì nhóm chưa đo chi phí tồn kho. Nguồn: `results/tables/test/test_overall.csv`, `test_by_demand_class.csv`.

**15. Smoke khác full reproduction thế nào?**
Smoke chạy cùng pipeline trên tập nhỏ (F10: 12 chuỗi, 7 mô hình, 42 bài kiểm thử) để kiểm tra chạy được end-to-end; không tái lập
các con số toàn mẫu và không dùng Test để phát triển.
