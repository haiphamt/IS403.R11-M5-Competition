# Nghiên cứu liên quan: LightGBM và dự báo bán lẻ


## 1. Ke và cộng sự (2017) — LightGBM: A Highly Efficient Gradient Boosting Decision Tree

- **Tác giả/năm:** Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, Tie-Yan Liu (2017), NeurIPS 30. [Bài gốc và abstract](https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html).
- **Bài toán:** giảm chi phí huấn luyện GBDT trên dữ liệu nhiều quan sát và nhiều đặc trưng; đây là bài phương pháp nền tảng, không phải thực nghiệm M5.
- **Đề xuất:** GOSS giữ các quan sát gradient lớn và lấy mẫu nhóm gradient nhỏ; EFB gộp các đặc trưng gần như loại trừ nhau để giảm số chiều.
- **Kết quả/giới hạn:** tác giả báo cáo tăng tốc đến hơn 20 lần trên các bộ dữ liệu công khai với độ chính xác gần tương đương. Không suy ra cùng mức tăng tốc trên M5; pilot nhóm dùng GBDT mặc định, không mặc nhiên bật GOSS.

## 2. Makridakis, Spiliotis và Assimakopoulos (2022) — M5 accuracy competition: Results, findings, and conclusions

- **Tác giả/năm:** Spyros Makridakis, Evangelos Spiliotis, Vassilios Assimakopoulos (2022), International Journal of Forecasting. [DOI bài gốc](https://doi.org/10.1016/j.ijforecast.2021.11.013).
- **Bài toán:** dự báo doanh số phân cấp Walmart trong 28 ngày; cuộc thi đánh giá trên 42.840 chuỗi gồm các mức tổng hợp, với 30.490 chuỗi đáy.
- **Đề xuất/phạm vi:** tổng kết tổ chức, tiêu chí RMSSE/WRMSSE, kết quả và phương pháp dự thi; đây là nghiên cứu tổng kết, không giới thiệu riêng một thuật toán mới.
- **Kết quả/giới hạn:** bảng kết quả cho thấy cả top 50 cải thiện hơn 14% so với benchmark tốt nhất, đội thắng cải thiện 22,4%. Đây là WRMSSE của thiết lập cuộc thi; không đồng nhất với MAE/RMSE/RMSSE/R² trên mẫu 1.200 chuỗi của nhóm và không quy toàn bộ lợi ích cho một mô hình đơn lẻ.

## 3. Bandara và cộng sự (2022) — A fast and scalable ensemble of global models with long memory and data partitioning for the M5 forecasting competition

- **Tác giả/năm:** Kasun Bandara, Hansika Hewamalage, Rakshitha Godahewa, Puwasala Gamakumara (2022), International Journal of Forecasting 38(4), 1400–1404. [Hồ sơ nghiên cứu và abstract của tác giả](https://research.monash.edu/en/publications/a-fast-and-scalable-ensemble-of-global-models-with-long-memory-an/), [DOI](https://doi.org/10.1016/j.ijforecast.2021.11.004).
- **Bài toán:** dự báo nhu cầu bán lẻ M5, khai thác học chéo giữa các chuỗi thay vì chỉ fit từng chuỗi độc lập.
- **Đề xuất:** framework ensemble các mô hình global; bài và bộ từ khóa xác định LightGBM, pooled regression, long memory và phân hoạch dữ liệu là các thành phần/chủ đề chính.
- **Kết quả/giới hạn:** abstract tác giả báo vị trí 17 ở track Accuracy, thuộc top 1%. Hồ sơ/abstract không cung cấp đủ số runtime để trích một hệ số tăng tốc; không lấy thứ hạng của ensemble global làm bằng chứng cho pilot LightGBM từng chuỗi của nhóm.

## Liên hệ với thí nghiệm

Ba bài lần lượt cung cấp cơ chế LightGBM, bối cảnh đánh giá bán lẻ M5 và cách dùng LightGBM trong mô hình học chéo/ensemble. Nhóm cần kiểm chứng thực nghiệm riêng trên cùng split, mẫu và metric; không sao chép thứ hạng hoặc mức cải thiện của cuộc thi làm kết quả của đồ án. Chỉ ghi các kết quả có nguồn xác minh ở trên; không bổ sung số liệu từ phần bài chưa truy cập được.
