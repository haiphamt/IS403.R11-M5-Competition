# Câu hỏi – đáp dự kiến khi bảo vệ (đồ án M5, Nhóm 6)

Các câu trả lời dựa trên cấu hình và mã trong repo. Số liệu xem `report/draft_report.md` và `results/full/tables/`;
các số dưới đây lấy từ bảng nguồn nêu kèm (nghiên cứu chính: toàn bộ 30.490 chuỗi, 5 mô hình).

**1. Vì sao chỉ 5 mô hình?**
Pilot trên 1.200 chuỗi đã chạy 11 mô hình (Bảng P1). Nhóm giữ mỗi cách tiếp cận một đại diện: Seasonal Naive (mốc), ETS (làm mượt mũ),
Croston SBA (hàng bán thưa), LightGBM (học máy global) và stacking lai (kết hợp). XGBoost, CatBoost ngang LightGBM nhưng chậm hơn
(CatBoost ước tính khoảng 14 giờ CPU trên toàn bộ dữ liệu); Croston cổ điển kém SBA; Linear Regression kém nhất nhóm học máy; LightGBM Tuned
không giữ được lợi thế trên Test của pilot.

**2. Vì sao chạy toàn bộ 30.490 chuỗi thay vì mẫu?**
Bỏ sai số do lấy mẫu (mẫu pilot lệch về chuỗi gián đoạn: 43,3% so với 39,1%) và tính được WRMSSE 12 cấp như M5, vì WRMSSE cần mọi chuỗi đáy.
Thứ tự bốn mô hình cơ sở trên Test giống pilot; macro-RMSSE toàn bộ cao hơn pilot khoảng 0,006–0,007.

**3. Vì sao horizon 28 ngày, lag ≥ 28 (28/35/42) và rolling shift 28?**
Đúng kỳ dự báo của M5. LightGBM dự báo trực tiếp cả 28 ngày từ một origin, nên ngày xa nhất T+28 chỉ chắc chắn biết y đến T; lag nhỏ hơn 28 sẽ
dùng nhãn chưa biết. `tests/test_no_leakage.py` làm hỏng mọi nhãn sau origin và kiểm tra đặc trưng, target không đổi.

**4. RMSSE khác WRMSSE thế nào?**
RMSSE tính từng chuỗi: RMSE chia cho sai số naïve một bước trên lịch sử Train (bỏ số 0 đầu), rồi lấy trung bình macro trên 30.490 chuỗi.
WRMSSE cộng dự báo lên 12 cấp (42.840 chuỗi), trọng số theo doanh thu 28 ngày cuối trước origin, rồi lấy trung bình 12 cấp (`scripts/wrmsse.py`).
Thứ hạng khác nhau: Croston SBA tốt ở cấp đáy nhưng WRMSSE Test 1,095 (kém nhất), LightGBM kém nhất cấp đáy nhưng WRMSSE 0,722.

**5. Điểm WRMSSE có so được với bảng xếp hạng M5 không?**
Cửa sổ Test (d_1914–d_1941) trùng giai đoạn chấm public leaderboard của M5, nên cùng thước đo; nhưng điểm này không phải bài nộp thi và chỉ một cửa sổ.
Kiểm tra cài đặt: Seasonal Naive cho 0,870, dự báo toàn 0 cho 5,447.

**6. Croston SBA khác Croston thế nào?**
Croston làm mượt lượng bán z và khoảng cách p chỉ khi có nhu cầu, dự báo phẳng z/p. Syntetos–Boylan chỉ ra z/p bị chệch lên nên nhân (1 − α/2);
với α = 0,1 hệ số là 0,95. Ở pilot SBA tốt hơn Croston ở cả Validation và Test.

**7. Stacking lai hoạt động thế nào, có rò rỉ không?**
ŷ = max(0, β0 + β1·ETS + β2·SBA + β3·LightGBM), β khớp bằng Ridge trên dự báo ngoài mẫu trong Train (fit đến mốc 1829 và 1857, dự báo 28 ngày
kế tiếp, nhãn tới d_1885). Bộ gộp khớp một lần rồi áp nguyên cho Validation và Test, nên không dùng nhãn Validation/Test. Trọng số: ETS 0,543,
LightGBM 0,371, SBA 0,138 (`diag_stacking_coefficients.csv`).

**8. Mô hình nào tốt nhất?**
Trên Test: stacking lai theo cả macro-RMSSE (0,757, nhỉnh hơn ETS 0,758) và WRMSSE (0,668). Trên Validation thứ tự đổi: ETS nhỉnh hơn stacking
ở cấp đáy và LightGBM có WRMSSE thấp nhất (0,591). Kết luận vững: ETS và stacking tốt hơn Croston SBA và LightGBM ở cấp đáy khoảng 0,01–0,02.

**9. Vì sao LightGBM kém ETS ở cấp đáy nhưng tốt hơn ở cấp tổng hợp?**
Importance tập trung vào `rolling_mean_28` (68,5%) và `rolling_mean_7` (18,2%): dự báo là mức doanh số trượt điều chỉnh theo lịch, học chung mọi chuỗi.
Cộng lại theo cửa hàng/bang, mức và chu kỳ tuần chung này rất chính xác (cấp 1: 0,610 trên Test). Ở từng chuỗi thưa, làm mượt riêng của ETS tốt hơn.

**10. R² âm / chuỗi hằng nghĩa là gì?**
R² âm nghĩa là dự báo kém hơn trung bình của chính nhãn đánh giá; với chuỗi thưa 28 ngày có phương sai nhỏ nên điều này phổ biến.
812 chuỗi Test có nhãn hằng nên R² không xác định; evaluator để NaN, ghi lý do và đếm riêng, không dùng epsilon.

**11. Chi phí tính toán?**
Trên Kaggle 4 lõi: Seasonal Naive dưới 1 giây, Croston SBA khoảng 10 giây, LightGBM khoảng 4 phút (cộng gần 1 phút dựng 56 triệu dòng đặc trưng),
ETS khoảng 2,5 giờ thực. Stacking tốn thêm khoảng 12 giây nhưng cần đủ ba mô hình cơ sở và hai backtest.

**12. Vì sao ETS cho dự báo âm?**
ETS chọn toàn cấu trúc sai số cộng; với chuỗi thưa, mức cộng mùa vụ có thể âm. 16.474 điểm dự báo Test (1,9%) được cắt về 0, như mọi mô hình.

**13. Giá tương lai?**
Pipeline dùng giá tuần của ngày cần dự báo (M5 công bố giá đến d_1969). Trong vận hành chỉ dùng được nếu giá được lên kế hoạch trước.

**14. Test có còn "mù" không?**
Cấu hình được khóa bằng tag `config-freeze` (commit `c8ba5c8`) trước khi chạy Test toàn bộ, và Test chạy một lần. Tuy vậy nhãn Test của 1.200 chuỗi
pilot (3,9% số chuỗi) đã được xem trước khi chọn năm mô hình; nhóm nêu rõ điều này trong phần giới hạn.

**15. Local hay global?**
Seasonal Naive, ETS, Croston SBA là local (một mô hình mỗi chuỗi, chỉ dùng lịch sử chuỗi đó). LightGBM là global (một mô hình cho mọi chuỗi,
dùng thêm giá/lịch/SNAP/nhóm). Stacking gộp cả hai. Khác biệt input là yếu tố gây nhiễu khi so thứ hạng.

**16. Smoke khác full reproduction thế nào?**
Smoke chạy cùng pipeline trên tập nhỏ (F10: 12 chuỗi, 7 mô hình, 42 bài kiểm thử ở lần chạy đó) để kiểm tra chạy được end-to-end; không tái lập
các con số và không dùng Test để phát triển. Nghiên cứu chính chạy lại toàn bộ trên Kaggle (`notebooks/06a–06d`).

**17. Đồ án đóng góp gì so với M5?**
Đồ án đối chiếu năm cách tiếp cận trên cùng 30.490 chuỗi và cùng kỳ dự báo 28 ngày, báo cáo sai số theo mức độ gián đoạn, theo 12 cấp WRMSSE
và ghi lại thời gian tính toán. Trong nhóm chuỗi có trên 80% số ngày bán bằng 0 ở tập huấn luyện, ETS, stacking và Croston SBA có macro-RMSSE Test
thấp hơn LightGBM; ở các cấp tổng hợp thì ngược lại. Đóng góp nằm ở cách so sánh và phân tích theo nhóm và theo cấp.

**18. Vì sao bảng chính chưa dùng Tweedie?**
Bảng chính dùng cấu hình đã khóa trước khi chạy Test, trong đó LightGBM dùng hàm mất mát bình phương mặc định.
Tweedie là một lựa chọn huấn luyện đã được dùng tại M5. Nhóm có thể đánh giá lựa chọn này trên nhiều cửa sổ giữ lại
để kiểm mức cải thiện trước khi đưa vào cấu hình chính thức.
