# S04 - Tóm tắt bài báo M5 Accuracy

**Người thực hiện: Thanh Tiền.** Makridakis, S., Spiliotis, E. & Assimakopoulos, V. (2022), *M5 accuracy competition: Results, findings, and conclusions*, *International Journal of Forecasting*, 38(4), 1346–1364. [DOI](https://doi.org/10.1016/j.ijforecast.2021.11.013).

## Bài toán và dữ liệu

Bài báo tổng kết cuộc thi M5 Accuracy: dự báo lượng bán hằng ngày của Walmart trong **28 ngày tiếp theo**, hỗ trợ lập kế hoạch bán hàng và tồn kho. Dữ liệu gồm **3.049 sản phẩm, 10 cửa hàng, 30.490 chuỗi sản phẩm–cửa hàng**, tổng cộng **42.840 chuỗi ở 12 cấp phân cấp**; đầu vào có lịch sử bán, lịch/sự kiện, SNAP và giá theo tuần. Thách thức chính là nhu cầu gián đoạn, nhiều ngày bán bằng 0, biến động mạnh và yêu cầu dự báo nhất quán giữa các cấp. Đây là bài tổng kết nhiều lời giải. [§§2–3]

## Phương pháp nổi bật và kiểm chứng LightGBM

Cuộc thi xếp hạng bằng **WRMSSE**: RMSSE chuẩn hóa sai số theo lịch sử Train, kết hợp trọng số doanh thu 28 ngày cuối Train và trọng số bằng nhau giữa 12 cấp. Bộ đối chuẩn có **24 cấu hình**, gồm thống kê, học máy và tổ hợp. [§2.2; §4.3]

Đội vô địch **YJ_STU (YeonJun In) dùng LightGBM**, xác nhận nhận định của S04. Đội xây dựng **220 mô hình** theo cửa hàng, cửa hàng–ngành hàng và cửa hàng–nhóm hàng, kết hợp dự báo đệ quy và không đệ quy. Mỗi chuỗi lấy trung bình đều **6 dự báo phù hợp**. Đội dùng Tweedie loss, đặc trưng định danh/lịch/giá/doanh số và kiểm định trên bốn cửa sổ 28 ngày. Hạng nhì kết hợp LightGBM với điều chỉnh từ N-BEATS; hạng ba dùng ensemble mạng nơ-ron có LSTM. [§4.2, tr. 1354–1355]

## Kết quả và kết luận chính

YJ_STU đạt **WRMSSE 0,520**, cải thiện **22,4%** so với đối chuẩn thống kê tốt nhất **ES_bu (0,671)**. Ở cấp sản phẩm–cửa hàng, điểm là **0,884 so với 0,915**, cải thiện khoảng **3,4%**. Chỉ **415/5.507 đội (7,5%)** vượt ES_bu: dùng học máy không tự bảo đảm tốt hơn đối chuẩn. [Bảng 2, tr. 1352; §3]

Các lời giải dẫn đầu cho thấy giá trị của học chung giữa nhiều chuỗi (*cross-learning*), biến ngoại sinh, kiểm định theo thời gian và kết hợp dự báo. Phân tích này chưa cô lập tác dụng của từng thành phần. [§4.3]

## Hạn chế và liên hệ đồ án

Kết quả gắn với Walmart và WRMSSE; điểm tổng thể tốt không đồng nghĩa tốt nhất ở mọi cấp hoặc mọi chuỗi, cũng không trực tiếp đo chi phí tồn kho. Chỉ **17/50 đội dẫn đầu** cung cấp mô tả phương pháp. Cần kiểm chứng khả năng khái quát và chi phí triển khai trên dữ liệu khác. [§§5.2–5.3]

Đồ án nhóm có thể vận dụng mô hình học chung, đặc trưng lịch/giá và kiểm định theo thời gian khi so sánh các mô hình. Thứ hạng M5 không thay thế kết quả nhóm; cần phân biệt WRMSSE cuộc thi với MAE/RMSE/RMSSE/R² và runtime của nhóm.

**Nguồn kiểm chứng:** các mục/trang chỉ dẫn thuộc bài báo ở DOI trên; mã lời giải được công bố tại [Mcompetitions/M5-methods](https://github.com/Mcompetitions/M5-methods).
