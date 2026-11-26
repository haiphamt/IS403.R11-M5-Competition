# Nội dung slide báo cáo đồ án M5 (Nhóm 6, IS403.R11)

Số liệu lấy từ `report/draft_report.md` (toàn bộ 30.490 chuỗi, 5 mô hình, Test d_1914–d_1941).
Mỗi slide gồm: **Trên slide** (chữ cần đưa lên), **Hình**, **Lời nói** (ý cần nói), **Thời lượng**.
Tổng phần chính 16 slide, khoảng 16 phút; phần dự phòng (backup) để trả lời hỏi đáp.
Nếu bị giới hạn 15 phút: gộp S15 vào S13 và nói S6 trong 30 giây.

## Quy tắc phân bổ thời gian (theo yêu cầu của thầy)

- **Lướt nhanh, không giảng lại lý thuyết** (đã học): ETS (Chương 4–5), hồi quy và Ridge (Chương 3), Gradient Descent, MAE/RMSE.
- **Dành thời gian giải thích** (chưa học): Croston SBA, LightGBM (boosting), stacking, RMSSE và WRMSSE.
- Thầy sẽ hỏi: ETS xác định cấu trúc và tham số bằng gì; LightGBM kết hợp các cây thế nào; độ đo có ý nghĩa gì; vì sao dùng phương pháp mới. Cả bốn câu đều có đáp án ngay trên slide bên dưới.

---

## S1. Trang bìa (0:15)

**Trên slide:** So sánh các phương pháp dự báo nhu cầu bán lẻ trên toàn bộ dữ liệu M5 (Walmart). Nhóm 6, môn Phân tích dữ liệu kinh doanh, IS403.R11. Tên năm thành viên: Tuấn Hải, Tuấn Vũ, Thanh Tiền, Thủy Tiên, Thùy Trang.

**Lời nói:** Chào thầy và các bạn. Nhóm em so sánh năm cách dự báo doanh số theo ngày trên toàn bộ dữ liệu của cuộc thi M5.

---

## S2. Bài toán và ý nghĩa (0:45)

**Trên slide:**
- **Đầu vào:** lịch sử bán theo ngày của từng sản phẩm tại từng cửa hàng, lịch/sự kiện, SNAP, giá tuần.
- **Đầu ra:** 28 dự báo điểm không âm cho 28 ngày tiếp theo, mỗi cặp sản phẩm–cửa hàng.
- **Quy mô:** 30.490 chuỗi (3 bang, 10 cửa hàng, 3 ngành hàng).
- Dự báo thấp thì thiếu hàng; dự báo cao thì tồn kho.
- **Câu hỏi nghiên cứu:** mô hình nào chính xác hơn, đổi lại tốn bao nhiêu thời gian, và thứ hạng có đổi khi nhìn ở các cấp tổng hợp không?

**Hình:** `report/latex_figures/vi_eda_daily_sales.pdf` (doanh số theo ngày).

**Lời nói:** Bài toán là dự báo số lượng bán 28 ngày tới cho từng cặp sản phẩm–cửa hàng. Doanh nghiệp dùng để quyết định bổ sung hàng.

---

## S3. Dữ liệu và thách thức (1:00)

**Trên slide:**
- **68,4%** số ngày có lượng bán bằng 0; **39,1%** số chuỗi có hơn 80% ngày bán 0 (nhu cầu gián đoạn).
- Hậu quả: độ đo phần trăm (MAPE) chia cho 0, không dùng được.
- Mẫu hình khác nhau theo mặt hàng, cửa hàng; cuối tuần cao hơn khoảng 20% mức trung bình tuần.
- **Chuỗi thời gian nên phải chia theo thời gian**, không xáo trộn; đặc trưng trễ phải tránh dùng thông tin tương lai (rò rỉ nhãn).
- 30.490 chuỗi: mô hình riêng từng chuỗi (local) có chi phí tính toán lớn.

**Lời nói:** Có bốn thách thức: nhiều số 0, các chuỗi khác nhau, nguy cơ rò rỉ nhãn, và chi phí tính toán. Thách thức đầu quyết định cả việc chọn mô hình lẫn việc chọn độ đo.

---

## S4. Bài báo M5 và hướng giải quyết đề xuất (1:00)

**Trên slide:**
- **Bài báo:** Makridakis, Spiliotis, Assimakopoulos (2022), *M5 accuracy competition*, Int. J. Forecasting.
- Kết quả của cuộc thi: đội thắng dùng LightGBM, WRMSSE 0,520, tốt hơn 22,4% so với đối chuẩn thống kê tốt nhất. Ở cấp sản phẩm–cửa hàng chỉ tốt hơn khoảng 3,4%; chỉ 7,5% đội vượt đối chuẩn.
- **Trước đó đã dùng:** thống kê (làm mượt mũ ETS; Croston cho nhu cầu gián đoạn), học máy (gradient boosting), kết hợp nhiều mô hình.
- **Hướng của nhóm:** đặt năm mô hình đại diện cho năm cách tiếp cận trên cùng dữ liệu, cùng kỳ 28 ngày, cùng độ đo.

| Cách tiếp cận | Mô hình |
|---|---|
| Mốc ngây thơ | Seasonal Naive |
| Làm mượt mũ | ETS |
| Hàng bán thưa | Croston SBA |
| Học máy global | LightGBM |
| Kết hợp | Stacking lai (ETS + Croston SBA + LightGBM) |

**Lời nói:** Bài báo M5 cho thấy học máy không tự động tốt hơn thống kê ở cấp từng chuỗi. Vì vậy nhóm không chạy đua điểm số mà so sánh có kiểm soát năm cách tiếp cận. Đây không phải tái lập hệ thống đội thắng.

---

## S5. Thiết kế thực nghiệm (1:00)

**Trên slide:**
- **Chia theo thời gian:** Train d_1–1885 · Validation d_1886–1913 · Test d_1914–1941.
- Dùng Validation để chọn cấu hình; **khóa cấu hình** (tag `config-freeze`) rồi mới chạy Test, **một lần**.
- Đặc trưng trễ ≥ 28 ngày; có kiểm thử tự động xác nhận không rò rỉ nhãn.
- **Hai giai đoạn:** pilot 1.200 chuỗi với 11 mô hình, chọn ra 5 mô hình, rồi chạy toàn bộ 30.490 chuỗi.
- Lý do bỏ: XGBoost, CatBoost ngang LightGBM nhưng chậm hơn (CatBoost ước tính khoảng 14 giờ CPU); Linear Regression kém nhất nhóm học máy.

**Lời nói:** Mấu chốt là Test chỉ mở một lần sau khi khóa cấu hình. Nhóm cũng nêu rõ: nhãn Test của 1.200 chuỗi pilot đã được xem trước, nên Test không còn hoàn toàn mù.

---

## S6. Năm mô hình và liên hệ với môn học (0:45)

**Trên slide:**

| Mô hình | Loại | Trong môn học |
|---|---|---|
| Seasonal Naive | local | mốc so sánh |
| ETS | local | **đã học** (Chương 4, 5) |
| Croston SBA | local | **mới** |
| LightGBM | global | **mới** (nền tảng Gradient Descent đã học) |
| Stacking (Ridge) | kết hợp | **mới**; Ridge là hồi quy Chương 3 thêm phạt L2 |

- *Local:* một mô hình cho mỗi chuỗi, chỉ thấy lịch sử chuỗi đó. *Global:* một mô hình cho mọi chuỗi, thêm giá, lịch, SNAP, nhóm hàng.

**Lời nói:** Phần kế thừa từ môn học nhóm nói nhanh. Em dành thời gian cho ba phần mới: Croston SBA, LightGBM và stacking.

---

## S7. ETS (đã học, nói nhanh) (0:45)

**Trên slide:**
- Họ làm mượt mũ: **E**rror, **T**rend, **S**easonal; chu kỳ mùa = 7 ngày.
- Ví dụ ETS(A,N,A): mức `l_t = l_{t-1} + α·e_t`, mùa `s_t = s_{t-7} + γ·e_t`.
- **Cấu trúc (E,T,S) chọn tự động cho từng chuỗi bằng AICc** (AutoETS, `statsforecast`); không chọn thủ công. Tham số α, β, γ ước lượng tự động khi khớp.
- Kết quả chọn: ETS(A,N,A) 52,4%; ETS(A,N,N) 35,8%; có xu hướng tắt dần khoảng 11%. α trung bình 0,066: làm mượt rất mạnh.
- Điểm yếu: có thể dự báo âm (1,9% điểm Test, cắt về 0) và **chậm: khoảng 2,5 giờ** cho 30.490 chuỗi.

**Hình:** `report/latex_figures/structure_autoets.pdf`.

**Lời nói:** Nếu thầy hỏi tham số xác định bằng gì: cấu trúc bằng AICc, tự động từng chuỗi; α, β, γ do thư viện ước lượng khi khớp. Nhóm không chọn tay.

---

## S8. Croston SBA (mới, giải thích kỹ) (1:30)

**Trên slide:**
- **Vấn đề:** làm mượt thông thường trên chuỗi nhiều số 0 bị các số 0 kéo nhiễu.
- **Ý tưởng Croston (1972):** chỉ cập nhật khi có bán (y_t > 0), tách thành hai ước lượng:
  - `z ← z + α(y_t − z)`: lượng bán khi có bán
  - `p ← p + α(q − p)`: khoảng cách (ngày) giữa hai lần bán; q là số ngày từ lần bán trước
  - Dự báo phẳng: `ŷ = z / p` (lượng bán trung bình mỗi ngày)
- **SBA (Syntetos–Boylan 2005):** `z/p` bị chệch lên, nhân hệ số `(1 − α/2)`:
  `ŷ = (1 − α/2) · z/p`, với α = 0,1 thì hệ số = 0,95.
- **Vì sao chọn:** chuyên cho hàng bán thưa; ở pilot SBA tốt hơn Croston cổ điển ở cả Validation và Test.
- **Kết quả:** macro-RMSSE 0,769, gần ETS (0,758), chỉ **khoảng 10 giây** (nhanh hơn ETS khoảng 3.350 lần).
- **Điểm yếu:** dự báo là đường phẳng, không có chu kỳ tuần, nên kém ở các cấp tổng hợp.

**Hình:** `report/latex_figures/structure_croston_sba.pdf`.

**Lời nói:** Ví dụ ngắn có thể đưa vào backup (S18). Điểm cần nhớ: Croston biến bài toán nhiều số 0 thành hai con số dễ ước lượng — bán bao nhiêu mỗi lần và bao lâu bán một lần.

---

## S9. LightGBM (mới, giải thích kỹ) (1:45)

**Trên slide:**
- **Gradient boosting (Friedman 2001):** xây dần các cây nhỏ, mỗi cây học **phần sai số còn lại** của các cây trước, rồi **cộng** vào, không lấy trung bình.
  - `F_0` = trung bình của y
  - Bước m = 1..100: phần dư `r = y − F_{m−1}(x)` (chính là gradient âm của hàm mất mát bình phương); khớp cây `f_m` (tối đa 31 lá) lên `r`
  - `F_m(x) = F_{m−1}(x) + η · f_m(x)`, η = 0,1
  - Dự báo cuối `ŷ = F_100(x)`
- Nối với môn học: đây là Gradient Descent trong không gian hàm.
- **LightGBM (Ke 2017):** nhanh nhờ GOSS (giữ mẫu có gradient lớn), EFB (gộp đặc trưng loại trừ nhau), mọc cây theo lá.
- **Cấu hình:** tham số mặc định thư viện (100 cây, learning rate 0,1, 31 lá, hàm mất mát bình phương). Không tuning, không early stopping.
- **Global:** một mô hình cho cả 30.490 chuỗi (56 triệu dòng), 21 đặc trưng: lag 28/35/42, trung bình và độ lệch chuẩn trượt 7/28, thứ, tháng, năm, sự kiện, SNAP, giá, nhóm hàng.
- **Vì sao chọn:** học chéo giữa các chuỗi, dùng được giá và lịch; là họ mô hình của phần lớn đội đứng đầu M5.
- **Importance (gain):** `rolling_mean_28` 68,5%, `rolling_mean_7` 18,2%.

**Hình:** `results/full/figures/m10_boosting_illustration.png`; `report/latex_figures/structure_lightgbm.pdf`.

**Lời nói:** Nếu thầy hỏi các cây kết hợp thế nào: tuần tự và cộng dồn, mỗi cây có trọng số cố định η = 0,1 và học phần phần dư, khác rừng ngẫu nhiên là các cây độc lập rồi lấy trung bình.
**Không nói nhóm dùng Tweedie:** nhóm dùng hàm mất mát bình phương mặc định. Tweedie chỉ là hướng phát triển.

---

## S10. Stacking lai (mới, giải thích kỹ) (1:15)

**Trên slide:**
- **Ý tưởng:** thay vì chọn một mô hình, học trọng số để gộp ba dự báo:
  `ŷ = max(0, β₀ + β₁·ETS + β₂·SBA + β₃·LightGBM)`
- **Bộ gộp (meta-learner):** hồi quy **Ridge** (hồi quy tuyến tính có phạt L2). Ridge vì ETS và SBA tương quan cao; phạt L2 giữ hệ số ổn định.
- **Khớp không rò rỉ:** fit ba mô hình đến mốc 1829 và 1857, dự báo 28 ngày kế tiếp (nhãn tới d_1885), được 1.707.440 dòng dự báo ngoài mẫu. Khớp Ridge một lần rồi áp cho Validation và Test.
- **Trọng số học được:** ETS 0,543 · LightGBM 0,371 · Croston SBA 0,138 (hệ số chặn ≈ 0).
- **Vì sao chọn:** local mạnh ở chuỗi thưa, global mạnh ở mức tổng hợp và chu kỳ tuần: hai nguồn lỗi bổ sung nhau.

**Hình:** `report/latex_figures/structure_stacking.pdf`.

**Lời nói:** Trọng số cho thấy bộ gộp gần như trung bình có trọng số, chủ yếu dựa vào ETS và LightGBM; SBA nhận ít vì giống ETS.

---

## S11. Tiêu chí đánh giá, phần 1: MAE, RMSE, RMSSE (1:15)

**Trên slide:** với `e_h = y_{T+h} − ŷ_{T+h}`, H = 28:

| Độ đo | Công thức | Ý nghĩa |
|---|---|---|
| MAE | `(1/H) Σ |e_h|` | sai số trung bình, cùng đơn vị (sản phẩm/ngày) |
| RMSE | `√((1/H) Σ e_h²)` | phạt nặng sai số lớn |
| RMSSE | `RMSE / √(trung bình (y_t − y_{t−1})² trên Train)` | sai số **chuẩn hóa**; so sánh được giữa các chuỗi |
| R² | `1 − SSE/SST` | có thể âm: kém hơn trung bình nhãn |

- **Diễn giải RMSSE:** < 1 nghĩa là tốt hơn dự báo ngây thơ một bước trên Train. Seasonal Naive có RMSSE ≈ 1 vì chính nó cũng là dự báo ngây thơ.
- **Không dùng MAPE:** chia cho giá trị thực, mà 68,4% ngày bán bằng 0.
- Xếp hạng chính: macro-RMSSE (trung bình trên 30.490 chuỗi).

**Lời nói:** Công thức viết lên bảng nếu thầy yêu cầu. RMSSE cho phép cộng điểm các chuỗi có quy mô khác nhau mà không chia cho 0.

---

## S12. Tiêu chí đánh giá, phần 2: WRMSSE (1:00)

**Trên slide:**
- **WRMSSE (độ đo của M5):** cộng dự báo lên **12 cấp** (tổng, bang, cửa hàng, ngành hàng, nhóm hàng, … đến sản phẩm × cửa hàng), tổng 42.840 chuỗi.
- Mỗi chuỗi có RMSSE riêng; **trọng số = doanh thu** (số lượng × giá) 28 ngày cuối trước origin, chuẩn hóa trong từng cấp.
- `WRMSSE = trung bình 12 cấp của Σ w_i · RMSSE_i`
- Ý nghĩa: sản phẩm bán nhiều tiền được tính nặng hơn; đo được cả chất lượng ở mức tổng.
- Kiểm tra cài đặt: dự báo hoàn hảo cho 0; Seasonal Naive cho 0,870; dự báo toàn 0 cho 5,447.

**Lời nói:** WRMSSE khác RMSSE ở hai điểm: có trọng số theo doanh thu và đánh giá ở mọi cấp phân cấp, nên thứ hạng có thể đảo.

---

## S13. Kết quả trên Test (1:15)

**Trên slide:**

| Mô hình | RMSSE (macro) | WRMSSE | MAE (macro) | Thời gian (s) |
|---|---:|---:|---:|---:|
| Seasonal Naive | 0,997 | 0,870 | 1,244 | 0,7 |
| ETS | 0,758 | 0,760 | 1,040 | 35.121 |
| Croston SBA | 0,769 | 1,095 | 1,075 | 10,5 |
| LightGBM | 0,775 | 0,722 | 1,079 | 228,4 |
| **Stacking lai** | **0,757** | **0,668** | 1,050 | 12,1 (+ 3 mô hình cơ sở) |

- Cả bốn mô hình giảm khoảng 22–24% RMSSE so với Seasonal Naive.
- Kết luận vững: {ETS, stacking} tốt hơn {Croston SBA, LightGBM} khoảng 0,01–0,02 ở cả hai cửa sổ.
- **Thận trọng:** chênh lệch stacking–ETS (0,0015) đổi chiều trên Validation, nên không coi là bền.

**Hình:** `report/latex_figures/vi_score_validation_test.pdf`.

**Lời nói:** Stacking tốt nhất trên Test, nhưng em nhấn mạnh phần đáng tin là nhóm dẫn đầu, không phải thứ tự trong nhóm.

---

## S14. Theo cấp tổng hợp và theo nhóm nhu cầu (1:00)

**Trên slide:**

| WRMSSE Test | Cấp 1 (tổng) | Cấp 12 (sản phẩm × cửa hàng) |
|---|---:|---:|
| ETS | 0,687 | 0,845 |
| Croston SBA | 1,186 | 0,862 |
| LightGBM | 0,610 | 0,857 |
| Stacking lai | 0,505 | 0,834 |

- Croston SBA tốt ở cấp đáy nhưng **kém nhất ở cấp tổng hợp** vì dự báo phẳng, cộng lại vẫn phẳng, mất chu kỳ tuần.
- LightGBM kém ở cấp đáy nhưng **tốt ở cấp tổng hợp** vì học chung mức và mùa vụ.
- Nhóm hàng bán thưa (> 80% ngày 0): ETS 0,795, stacking 0,796, SBA 0,801, **LightGBM 0,820**.

**Hình:** `results/full/figures/test/test_rmsse_by_demand_class.png`.

**Lời nói:** Đây là lý do stacking có ý nghĩa: hai loại mô hình sai ở hai chỗ khác nhau.

---

## S15. Chi phí tính toán (0:30)

**Trên slide:**
- Seasonal Naive < 1 s · Croston SBA khoảng 10 s · LightGBM khoảng 4 phút · ETS khoảng 2,5 giờ (4 tiến trình).
- Stacking: bước gộp 12 s nhưng cần đủ ba mô hình cơ sở và hai backtest, nên tổng lớn hơn ETS.
- Nếu cần nhanh, Croston SBA là lựa chọn nên thử trước cho hàng bán thưa.

**Hình:** `report/latex_figures/vi_accuracy_runtime_test.pdf`.

---

## S16. Kết luận, giới hạn, hướng phát triển (1:15)

**Trên slide:**
- **Kết luận:** (Q1) mọi mô hình giảm khoảng 22–24% so với mốc; (Q2) hàng bán thưa: mô hình local và stacking hơn LightGBM; (Q3) Croston SBA gần như không tốn thời gian, ETS tốn nhất; (Q5) thứ hạng đổi mạnh ở cấp tổng hợp. Stacking lai cân bằng nhất trên Test.
- **Giới hạn:** một cửa sổ Test 28 ngày; giả định biết giá tương lai; local và global khác input; chưa đo chi phí tồn kho; chỉ dự báo điểm.
- **Hướng phát triển:** kiểm định cuốn chiếu nhiều mùa; khớp stacking theo WRMSSE; thêm chu kỳ tuần cho Croston; thử hàm mất mát Tweedie và tuning cho LightGBM; mô phỏng quyết định tồn kho; dự báo xác suất.

**Lời nói:** Xin cảm ơn thầy và các bạn, nhóm sẵn sàng trả lời câu hỏi.

---

# Slide dự phòng (chỉ mở khi được hỏi)

- **B1. Ví dụ số Croston** (α = 0,1, chuỗi `0,0,4,0,0,0,2,0,3`): sau lần bán 3 có z = 3,72, p = 2,99; Croston ŷ ≈ 1,244; SBA ≈ 0,95 × 1,244 ≈ 1,182 (thư viện có thể khởi tạo hơi khác).
- **B2. Ví dụ số stacking:** ETS = 1,0; SBA = 0,8; LightGBM = 1,2 thì ŷ = 0,5429·1,0 + 0,1377·0,8 + 0,3713·1,2 − 0,005 ≈ 1,094.
- **B3. Kết quả Validation:** macro-RMSSE: Seasonal Naive 0,976; ETS 0,737; SBA 0,759; LightGBM 0,757; stacking 0,738. WRMSSE: LightGBM 0,591 (thấp nhất), stacking 0,715.
- **B4. Bootstrap:** khoảng tin cậy 95% Test của stacking − ETS: [−0,0020; −0,0009]; chỉ phản ánh biến thiên giữa chuỗi trong một cửa sổ.
- **B5. Bảng pilot 1.200 chuỗi** (Bảng P1 trong báo cáo): lý do giữ và bỏ từng mô hình.
- **B6. 21 đặc trưng của LightGBM** và cách chứng minh không rò rỉ (`tests/test_no_leakage.py`).
- **B7. AICc:** `AICc = AIC + 2k(k+1)/(n−k−1)`, k là số tham số, n là số quan sát; cân bằng độ khớp và độ phức tạp.

---

# Chuẩn bị hỏi đáp: các câu thầy có thể hỏi

| Câu hỏi | Trả lời ngắn |
|---|---|
| ETS xác định cấu trúc và tham số bằng gì? Tự động hay thủ công? | **Tự động** từng chuỗi: AutoETS chọn (E,T,S) theo AICc, chu kỳ 7; α, β, γ do thư viện ước lượng khi khớp. Nhóm không chọn tay. |
| LightGBM kết hợp các cây thế nào? | Cộng dồn tuần tự: `F_m = F_{m−1} + 0,1·f_m`, mỗi cây khớp phần dư của tổng các cây trước (gradient âm của hàm bình phương); 100 cây, 31 lá. Khác rừng ngẫu nhiên (lấy trung bình cây độc lập). |
| Vì sao dùng LightGBM, Croston SBA, stacking? | LightGBM: học chéo giữa 30.490 chuỗi, dùng giá và lịch, là họ mô hình của đội thắng M5. Croston SBA: chuyên cho nhu cầu gián đoạn (68,4% ngày 0), rất rẻ. Stacking: local và global sai ở chỗ khác nhau nên gộp có lợi. |
| RMSSE, WRMSSE có ý nghĩa gì? Sao không dùng MAPE? | RMSSE: sai số chia cho sai số ngây thơ của chính chuỗi, < 1 là tốt hơn ngây thơ, so được giữa chuỗi. WRMSSE: thêm trọng số doanh thu và 12 cấp. MAPE chia cho 0 khi bán 0. |
| Sao Ridge mà không hồi quy tuyến tính thường? | ETS và SBA tương quan cao, hệ số hồi quy thường sẽ bất ổn; phạt L2 làm ổn định. Alpha chọn bằng leave-one-origin-out (kết quả gần như không nhạy). |
| Stacking có rò rỉ không? | Không: khớp trên dự báo ngoài mẫu trong Train (mốc 1829, 1857; nhãn tới d_1885), khớp một lần rồi áp cho Validation và Test. |
| Sao không dùng ARIMA, SVR, RNN? | Nhóm chưa chạy các mô hình này, nên không khẳng định chúng kém hơn. Nhóm chọn mỗi cách tiếp cận một đại diện từ pilot; ETS đã tốn khoảng 2,5 giờ cho 30.490 chuỗi nên ngân sách tính toán cũng là một ràng buộc. |
| Có dùng Tweedie không? | **Không.** LightGBM dùng hàm mất mát bình phương mặc định, đã khóa trước khi chạy Test. Tweedie là hướng phát triển (đội thắng M5 dùng). |
| Mô hình nào tốt nhất? | Stacking trên Test (RMSSE 0,757, WRMSSE 0,668); nhưng trên Validation ETS nhỉnh hơn ở cấp đáy và LightGBM có WRMSSE thấp nhất. Kết luận vững là nhóm dẫn đầu, không phải thứ tự trong nhóm. |

Thêm 18 câu hỏi đáp chi tiết tại `report/qa.md` và cẩm nang `report/KIEN_THUC_DO_AN.md`.

---

# Lưu ý đối chiếu với đoạn phân tích đã dán

Đoạn phân loại "đã học / mới" trong tin nhắn có hai điểm **không khớp với đồ án thật**, đừng đưa lên slide:

1. **Tweedie:** đoạn đó viết LightGBM dùng hàm mất mát Tweedie. Đồ án dùng cấu hình mặc định (hàm bình phương), Tweedie chỉ ở mục hướng phát triển.
2. **Hồi quy tuyến tính bội "Tầng 1":** Linear Regression chỉ có trong pilot và đã bị loại; bản stacking cuối là ETS + Croston SBA + LightGBM, gộp bằng Ridge. Hồi quy chỉ xuất hiện ở bộ gộp Ridge.
3. Cũng không có SVR, ARIMA, RNN, LSTM trong đồ án; các ví dụ đó là phần nhắc chung của thầy, không áp dụng cho bài này.
