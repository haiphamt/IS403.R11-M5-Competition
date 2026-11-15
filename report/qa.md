# Câu hỏi – đáp dự kiến khi bảo vệ (đồ án M5, Nhóm 6)

Các câu trả lời dựa trên cấu hình và mã trong repo. Số liệu cụ thể xem `report/draft_report.md`
và `results/tables/`; không trích số khi chưa có bảng nguồn.

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
Hệ số của biến đã chuẩn hóa (median impute + StandardScaler fit trên lịch sử fold) cho thấy liên hệ
điều kiện giữa đặc trưng và doanh số; biến phân loại one-hot bỏ nhóm gốc. Không phải tác động nhân quả.

**8. Boosting cộng cây như thế nào?**
F_m(x) = F_{m-1}(x) + η f_m(x): mỗi cây mới học phần sai số còn lại của mô hình hiện tại. Sơ đồ minh họa
khác với cây thật học được; importance thật nằm ở `diag_*_importance.csv`.

**9. Default so với Tuned?**
Core dùng tham số mặc định thư viện (chỉ đặt seed/luồng/khai báo categorical). Tuning bằng Optuna là P1,
chỉ chạy khi không chặn lõi; xem phần phạm vi trong báo cáo.

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
score trên chính tập huấn luyện meta không phải đánh giá độc lập. Trạng thái làm/cắt xem báo cáo.

**14. Vì sao mô hình thắng/thua?**
Trả lời bằng bảng theo nhóm nhu cầu/category/bang và đồ thị theo horizon/thứ trong tuần của run đã khóa;
nêu cơ chế (ví dụ Croston phẳng hợp nhu cầu gián đoạn, Seasonal Naive nhạy nhiễu tuần cuối).

**15. Smoke khác full reproduction thế nào?**
Smoke chạy cùng pipeline trên subset nhỏ (ví dụ 24 chuỗi) để kiểm tra chạy được end-to-end; không tái lập
các con số toàn mẫu và không dùng Test để phát triển.
