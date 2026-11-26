# CẨM NANG KIẾN THỨC ĐỒ ÁN M5 (NHÓM 6, IS403.R11)

> **Mục đích.** Một người chưa biết gì về đồ án đọc file này xong phải hiểu: đề bài là gì, dữ liệu ra sao, nhóm làm
> gì và vì sao làm vậy, từng mô hình hoạt động thế nào, các độ đo nghĩa là gì, kết quả ra sao, và trả lời được các
> câu thầy cô có thể hỏi. Mọi con số đã được đối chiếu với các bảng trong `results/full/tables/`.
>
> **Nguồn của file này:** `report/draft_report.md` (bản thảo gốc của số liệu), `report/qa.md`, `README.md`,
> `configs/*.yaml`, mã trong `src/m5/` và `scripts/`, các bảng trong `results/`. File **không** lấy từ bản LaTeX trên
> Overleaf; câu chữ bản LaTeX có thể khác, nhưng số liệu được sinh từ cùng kết quả.
>
> **Cách đọc theo thời gian.**
> - 2 phút: đọc mục 0 và mục 15 (bảng số cần nhớ).
> - 20 phút: đọc mục 0–8.
> - Trước buổi bảo vệ: đọc hết, rồi tập trả lời mục 13 (hỏi đáp) mà không nhìn đáp án.

---

## 0. Tóm tắt trong 1 phút

**Đề bài.** Dự báo số lượng bán **theo ngày** của **từng sản phẩm tại từng cửa hàng** Walmart trong **28 ngày tiếp theo**
(dữ liệu cuộc thi M5 Forecasting – Accuracy). Có **30.490** chuỗi như vậy.

**Khó ở đâu.** Phần lớn ngày bán bằng 0 (68,4% số ngày trên toàn bộ dữ liệu huấn luyện). Gọi là *nhu cầu gián đoạn*.
Nên các độ đo theo phần trăm (MAPE) không dùng được và các phương pháp thông thường dễ sai.

**Nhóm làm gì.** So sánh **5 mô hình** đại diện cho 5 cách tiếp cận trên **toàn bộ 30.490 chuỗi**, cùng cách chia dữ
liệu theo thời gian, cùng 28 ngày dự báo, cùng độ đo:

| # | Mô hình | Là gì (một câu) |
|---|---|---|
| 1 | **Seasonal Naive** | Lặp lại 7 ngày cuối cùng. Mốc đối chuẩn, không học gì. |
| 2 | **ETS** | Làm mượt mũ có mùa vụ tuần, chọn cấu trúc tự động cho từng chuỗi. |
| 3 | **Croston SBA** | Chuyên cho hàng bán thưa: dự báo = (lượng bán trung bình) / (khoảng cách giữa hai lần bán) × hệ số hiệu chỉnh. |
| 4 | **LightGBM** | Một mô hình học máy (cây tăng cường gradient) chung cho mọi chuỗi, dùng lịch/giá/SNAP/đặc trưng trễ. |
| 5 | **Stacking lai** | Trung bình có trọng số học được của ETS + Croston SBA + LightGBM (hồi quy Ridge). |

**Kết quả cốt lõi (Test, 28 ngày d_1914–d_1941, chạy một lần).**

| Mô hình | RMSSE macro | WRMSSE | Thời gian |
|---|---:|---:|---|
| Seasonal Naive | 0,997 | 0,870 | 0,7 giây |
| ETS | 0,758 | 0,760 | ≈ 35.121 giây CPU (≈ 2,5 giờ thực, 4 tiến trình) |
| Croston SBA | 0,769 | 1,095 | ≈ 10,5 giây |
| LightGBM | 0,775 | 0,722 | ≈ 228 giây |
| **Stacking lai** | **0,757** | **0,668** | 12,1 giây (+ chi phí ba mô hình cơ sở) |

**Năm thông điệp (Q1–Q5 của báo cáo):**
1. Cả bốn mô hình đều giảm khoảng **22–24%** sai số so với Seasonal Naive.
2. Hàng **bán thưa** (> 80% ngày bán 0): mô hình cục bộ (ETS, Croston SBA) và stacking tốt hơn rõ LightGBM.
3. Chi phí khác nhau cực lớn: Croston SBA ≈ 10 giây, ETS ≈ 35.000 giây (gấp ≈ 3.350 lần) mà sai số chỉ hơn 0,011.
4. Cơ chế giải thích được: ETS làm mượt rất mạnh; Croston SBA cập nhật theo lần có nhu cầu; LightGBM chủ yếu dựa vào mức doanh số trượt 28 ngày.
5. Ở **các cấp tổng hợp** (WRMSSE) thứ hạng đổi: LightGBM tốt, Croston SBA **kém nhất** vì dự báo phẳng không có chu kỳ tuần.

**Lời nói chính xác nhất về "mô hình tốt nhất".** Trên Test, stacking đứng đầu theo cả hai độ đo, nhưng chênh với ETS ở cấp
đáy chỉ 0,0015 và **đổi chiều trên Validation**. Kết luận vững: ETS và stacking tốt hơn Croston SBA và LightGBM ở cấp đáy khoảng
0,01–0,02, và bốn mô hình đều tốt hơn rõ Seasonal Naive. Đừng tuyên bố "stacking thắng chắc chắn".

---

## 1. Bối cảnh: cuộc thi M5 và bài toán

### 1.1. M5 là gì
M5 (Makridakis 5) là cuộc thi dự báo do giáo sư Spyros Makridakis tổ chức trên Kaggle. Phần **Accuracy** yêu cầu dự báo doanh số
Walmart trong **28 ngày**. Thông tin từ bài tổng kết [1] (xem `paper/summary.md`):

- Dữ liệu: **3.049 sản phẩm × 10 cửa hàng = 30.490 chuỗi đáy**, cộng các cấp tổng hợp thành **42.840 chuỗi ở 12 cấp**.
- Xếp hạng bằng **WRMSSE** (mục 7.5).
- Đội thắng (YJ_STU) dùng **LightGBM**, 220 mô hình, trung bình 6 dự báo, hàm mất mát Tweedie: WRMSSE **0,520**, cải thiện
  **22,4%** so với benchmark thống kê tốt nhất ES_bu (WRMSSE 0,671).
- Ở riêng cấp sản phẩm–cửa hàng, cải thiện chỉ khoảng **3,4%** (0,884 so với 0,915).
- Chỉ **415/5.507 đội (7,5%)** vượt ES_bu. Bài học: **học máy không tự động tốt hơn** mô hình thống kê đơn giản.
- Hạng nhì: LightGBM kết hợp điều chỉnh từ N-BEATS; hạng ba: ensemble mạng nơ-ron có LSTM.

**Quan hệ giữa đồ án và M5.** Đồ án **không** tái lập hệ thống đội thắng. Đây là so sánh thuật toán trong điều kiện tài nguyên
của nhóm. Cửa sổ Test của nhóm (d_1914–d_1941) trùng giai đoạn chấm bảng xếp hạng **công khai** (public leaderboard) của M5; giai
đoạn private (d_1942–d_1969) không dùng. Điểm của nhóm **không phải bài nộp thi**.

### 1.2. Đầu vào, đầu ra của bài toán
- **Đầu vào:** lịch sử bán hàng theo ngày đến ngày T, cộng lịch (thứ, tháng, năm, sự kiện), cờ SNAP theo bang, giá bán theo tuần.
- **Đầu ra:** 28 dự báo điểm **không âm** ŷ(T+1)…ŷ(T+28) cho mỗi chuỗi.
- Nhóm làm **dự báo điểm** (một con số mỗi ngày), không làm dự báo xác suất/khoảng.

### 1.3. Vì sao bài toán khó
- **Nhu cầu gián đoạn (intermittent demand):** 68,4% số ngày bằng 0; 39,1% số chuỗi có trên 80% ngày bán 0. Một chuỗi điển hình
  là `0,0,0,2,0,0,1,0,0,0,3,…`.
- **Chuỗi bắt đầu muộn:** 23.511/30.490 chuỗi có đoạn toàn số 0 ở đầu (sản phẩm chưa được bán/niêm yết), trung vị 452 ngày.
- **Chu kỳ tuần mạnh:** thứ Bảy, Chủ nhật bán cao hơn khoảng 20% so với trung bình tuần; thứ Ba–Năm thấp hơn khoảng 12% (mẫu pilot).
- **Khác biệt giữa chuỗi lớn:** cùng một phương pháp có thể tốt với hàng bán đều, tệ với hàng bán thưa.

---

## 2. Dữ liệu

### 2.1. Năm file gốc (Kaggle, đặt ở `data/raw/`, **không** đưa lên Git)

| File | Nội dung | Kích thước |
|---|---|---|
| `calendar.csv` | Lịch: ngày, thứ, tháng, năm, sự kiện (tên, loại), cờ SNAP của CA/TX/WI, mã tuần `wm_yr_wk` | 1.969 dòng × 14 cột |
| `sell_prices.csv` | Giá theo (cửa hàng, sản phẩm, tuần) | 6.841.121 dòng × 4 cột |
| `sales_train_validation.csv` | Doanh số d_1–d_1913 | 30.490 × 1.919 |
| `sales_train_evaluation.csv` | Doanh số d_1–d_1941 | 30.490 × 1.947 |
| `sample_submission.csv` | Mẫu định dạng nộp bài (không có nhãn) | 60.980 × 29 |

- **Pipeline dùng `sales_train_evaluation.csv`** (đủ đến d_1941). Hai file doanh số trùng nhau ở d_1–d_1913 (đã kiểm), **không ghép chồng**.
- Mã ngày `d_k`: d_1 = 29/01/2011 (thứ Bảy). d_1941 = 22/05/2016. d_1914 = 25/04/2016 (**thứ Hai**), nên 28 ngày Test là đúng 4 tuần thứ Hai→Chủ nhật.
- **Khóa chuỗi:** `series_id` = `item_id_store_id`, ví dụ `HOBBIES_1_001_CA_1`. Mọi phép nối theo khóa, **không theo thứ tự dòng**.

### 2.2. Cấu trúc phân cấp
- 3 **bang**: CA (4 cửa hàng), TX (3), WI (3) → 10 **cửa hàng** (`store_id`).
- 3 **ngành hàng** (`cat_id`): FOODS, HOBBIES, HOUSEHOLD.
- 7 **nhóm hàng** (`dept_id`): FOODS_1/2/3, HOBBIES_1/2, HOUSEHOLD_1/2.
- 3.049 **sản phẩm** (`item_id`).
- Trong báo cáo: *ngành hàng* = `cat_id`, *nhóm hàng* = `dept_id`.

### 2.3. Ba khái niệm lịch/giá cần biết
- **SNAP:** chương trình hỗ trợ tem thực phẩm của Mỹ; cờ 0/1 theo bang cho biết ngày đó được dùng SNAP hay không (ảnh hưởng doanh số thực phẩm).
- **Giá thiếu:** nếu sản phẩm chưa được niêm yết ở một tuần thì không có giá. Nhóm **giữ NaN** và thêm cờ `price_missing`, **không back-fill**.
- **Giả định giá tương lai:** pipeline dùng giá tuần của **chính ngày cần dự báo** (bảng giá M5 công bố đến d_1969). Đây là giả định phải nêu, không phải bằng chứng giá gây ra doanh số. Thực tế chỉ dùng được nếu giá được lên kế hoạch trước.

### 2.4. Nhóm nhu cầu (dùng để phân tích Q2)
Mỗi chuỗi có `train_zero_ratio` = tỷ lệ ngày bán 0 trên Train (d_1–d_1885), chia ba nhóm (`data/full_series_ids_with_class.csv`):

| Nhóm | Điều kiện | Số chuỗi | Tỷ lệ |
|---|---|---:|---:|
| `low_zero` | ≤ 0,5 | 6.530 | 21,4% |
| `mid_zero` | 0,5 – 0,8 | 12.053 | 39,5% |
| `high_zero` | > 0,8 (bán rất thưa) | 11.907 | 39,1% |

**Độ bền của cách chia:** vì tính cả số 0 *trước lần bán đầu tiên*, nhóm `high_zero` bị thổi phồng. Tính lại chỉ sau lần bán
đầu tiên: 7.341 chuỗi (24%) đổi nhóm, `high_zero` giảm 11.907 → 7.486. Kết luận về mô hình **không đổi** (Bảng 7d, mục 8.4).

---

## 3. Thiết kế thí nghiệm

### 3.1. Chia dữ liệu theo thời gian (không xáo trộn)

| Tập | Ngày | Vai trò |
|---|---|---|
| **Train** | d_1 – d_1885 | Huấn luyện, phân nhóm nhu cầu, backtest cho stacking (mốc 1829, 1857) |
| **Validation** | d_1886 – d_1913 | Xem kết quả trước khi khóa cấu hình |
| **Test** | d_1914 – d_1941 | Đánh giá cuối. **Refit trên d_1–d_1913** rồi dự báo, chạy **một lần** |

- **Origin** (mốc dự báo) T: Validation T = d_1885, Test T = d_1913. Chỉ dùng dữ liệu ≤ T để dự báo T+1…T+28.
- Dữ liệu chia theo thời gian vì dự báo thực tế luôn nhìn về tương lai; xáo trộn ngẫu nhiên sẽ làm rò rỉ thông tin.

### 3.2. Quy trình chống "nhìn trộm" đáp án Test
1. Dùng Validation để chọn/kiểm tra cấu hình.
2. **Khóa cấu hình** bằng git tag `config-freeze` (hiện trỏ commit `c8ba5c8`; tag `config-freeze-v2` cũng ở commit này; xem `configs/freeze.yaml`, khối `full_data_v3`).
3. Mã chặn chạy Test nếu: tag không có, không nằm trong HEAD, hoặc cây làm việc có thay đổi chưa commit (ngoài `results/`) (`scripts/run_experiment.py`, hàm `freeze_problem`).
4. Chạy Test **một lần**, không chỉnh tham số dựa trên Test.

**Điểm trung thực phải nêu khi được hỏi:** 1.200 chuỗi pilot (3,9% số chuỗi) đã có nhãn Test được xem **trước khi** chốt 5 mô hình
(mục 3.3), nên Test toàn bộ **không còn hoàn toàn "mù"** với quyết định chọn mô hình. Các tag cũ đã bị gỡ và tạo lại ngày 18/11/2026
tại `c8ba5c8`; các SHA của run pilot ghi ở `configs/freeze.yaml` và `results/pre_rewrite_sha_map.csv`.

### 3.3. Hai giai đoạn: pilot 1.200 chuỗi → toàn bộ dữ liệu

**Giai đoạn 1 – Pilot.** Mẫu phân tầng **1.200 chuỗi** (10 cửa hàng × 3 ngành hàng × 40 chuỗi, seed 42; `data/sample_series_ids.csv`), chạy
**11 mô hình**: 7 lõi (Seasonal Naive, Linear Regression, ETS, Croston, LightGBM, XGBoost, CatBoost) và 4 mở rộng (Croston SBA,
LightGBM tinh chỉnh bằng Optuna, LightGBM bỏ giá/sự kiện, stacking Ridge cũ).

**Giai đoạn 2 – Nghiên cứu chính.** Chọn 5 mô hình, chạy toàn bộ **30.490** chuỗi.

*Bảng P1. Kết quả pilot (macro-RMSSE) và quyết định* (`results/runs/*/metrics_summary.csv`)

| Mô hình (pilot 1.200 chuỗi) | Validation | Test | Quyết định |
|---|---:|---:|---|
| ETS | 0,729 | 0,752 | chọn |
| Croston SBA | 0,743 | 0,762 | chọn |
| Croston | 0,745 | 0,764 | bỏ |
| LightGBM bỏ giá/sự kiện | 0,748 | 0,766 | bỏ |
| CatBoost | 0,763 | 0,768 | bỏ |
| LightGBM | 0,750 | 0,768 | chọn |
| XGBoost | 0,750 | 0,768 | bỏ |
| Stacking (LR + LightGBM Tuned + XGBoost + CatBoost) | 0,752 | 0,769 | chọn, **dựng lại** thành bản lai ETS + SBA + LightGBM |
| LightGBM Tuned | 0,745 | 0,770 | bỏ |
| Linear Regression | 0,759 | 0,776 | bỏ |
| Seasonal Naive | 0,978 | 0,997 | chọn (mốc) |

**Lý do chọn 5 mô hình:**
- Mỗi mô hình đại diện **một cách tiếp cận khác nhau**: mốc ngây thơ, làm mượt mũ, chuyên cho hàng thưa, học máy global, kết hợp.
- Bỏ mô hình **trùng vai trò, không tốt hơn**: XGBoost và CatBoost ngang LightGBM nhưng chậm hơn (CatBoost ước tính **≈ 14 giờ CPU** trên toàn bộ dữ liệu, vượt giới hạn một phiên Kaggle); Croston cổ điển kém SBA ở cả hai tập; Linear Regression kém nhất nhóm học máy.
- Bỏ phương án **không đứng vững trên Test**: LightGBM Tuned hơn bản mặc định ở Validation nhưng không hơn ở Test; "bỏ giá/sự kiện" chỉ là phép thử phụ.
- Stacking được giữ nhưng **dựng lại** (ba mô hình cơ sở cũ đã bị bỏ; kết hợp cục bộ + global là phép thử mới).

**Vì sao chạy toàn bộ dữ liệu:** bỏ sai số do lấy mẫu (mẫu pilot lệch về chuỗi gián đoạn: 43,3% so với 39,1% chuỗi có > 80% ngày 0) và tính được
**WRMSSE 12 cấp** (cần mọi chuỗi đáy). Thứ tự bốn mô hình cơ sở không đổi so với pilot; macro-RMSSE toàn bộ cao hơn pilot khoảng 0,006–0,007.

### 3.4. Local và global (khái niệm rất hay bị hỏi)
- **Mô hình cục bộ (local):** mỗi chuỗi một mô hình riêng, chỉ thấy **lịch sử doanh số của chính chuỗi đó**. Gồm Seasonal Naive, ETS, Croston SBA. 30.490 chuỗi = 30.490 lần fit.
- **Mô hình toàn cục (global):** **một** mô hình học chung từ mọi chuỗi, dùng thêm giá/lịch/SNAP/nhóm hàng và đặc trưng trễ. Là LightGBM.
- **Stacking** gộp cả hai.
- **Hệ quả cần nói rõ:** khi so thứ hạng local và global, ta đang gộp *hiệu ứng thuật toán* và *hiệu ứng thông tin ngoại sinh*; không so sạch hoàn toàn.

---

## 4. Đặc trưng và chống rò rỉ nhãn

### 4.1. Dự báo trực tiếp 28 ngày (direct, không đệ quy)
Mô hình dự báo cả 28 ngày từ **một** origin. Với ngày xa nhất T+28, ta chỉ **chắc chắn biết** doanh số đến T. Vì vậy mọi đặc trưng
dựa trên doanh số phải lùi **ít nhất 28 ngày**. Đây là câu trả lời cho "vì sao lag 28/35/42 mà không lag 1, 7?".

### 4.2. Đặc trưng của LightGBM (21 đặc trưng)
Với chuỗi i, ngày t:

| Nhóm | Đặc trưng | Ý nghĩa |
|---|---|---|
| Trễ | `lag_28`, `lag_35`, `lag_42` = y(t−k) | Doanh số 28/35/42 ngày trước (cùng thứ trong tuần) |
| Trượt | `rolling_mean_7`, `rolling_mean_28`, `rolling_std_7`, `rolling_std_28` | Trung bình và độ lệch chuẩn của cửa sổ 7 hoặc 28 ngày, **sau khi lùi 28 ngày**: rmean_w = (1/w) Σ_{j=0}^{w−1} y(t−28−j) |
| Lịch | `wday`, `month`, `year` | Thứ, tháng, năm |
| Sự kiện | `event_name_1/2`, `event_type_1/2` | Hai cặp (tên, loại) sự kiện |
| SNAP | `snap` | Cờ SNAP của bang của chuỗi |
| Giá | `sell_price`, `price_missing` | Giá tuần, cờ giá thiếu |
| Phân loại | `dept_id`, `cat_id`, `store_id`, `state_id` | Nhóm hàng, ngành hàng, cửa hàng, bang |

- **Không dùng `item_id`** làm biến phân loại (hơn 3.000 giá trị, dễ quá khớp, nặng bộ nhớ). Đây là khác biệt so với `configs/features.yaml`, được ghi nhận trong báo cáo.
- Bảng huấn luyện toàn bộ: ≈ **55,8 triệu dòng** (Validation: 1.830 ngày × 30.490), ≈ **56,6 triệu dòng** (Test: 1.858 ngày × 30.490 = 56.650.420). Ngày đầu tiên dùng là d_56 (khi `lag_42` và cửa sổ trượt đủ dữ liệu).
- Cột phân loại dựng từ mã số thay chuỗi chữ để vừa RAM Kaggle (≈ 30 GB); kiểm tra cho thấy dự báo LightGBM trùng cách dựng cũ (lệch tối đa 1,8·10⁻¹⁵).

### 4.3. Cách chứng minh "không rò rỉ"
- Nhãn từ T+1 trở đi bị **che bằng NaN** trước khi dựng đặc trưng (`global_models.build_table`).
- Hàm `lag_roll_features` **từ chối** lag < 28 (`ValueError`).
- `tests/test_no_leakage.py`: phá hỏng toàn bộ nhãn sau origin rồi xác nhận **đặc trưng và target không đổi**.
- Đánh giá nối dự báo với nhãn **theo khóa** `(series_id, d)`, không theo thứ tự dòng; lược đồ dự báo (`src/m5/schema.py`) kiểm tra đủ 28 ngày × mọi chuỗi, không trùng lặp, không NaN.
- Mô hình cục bộ chỉ nhận `y[:origin]` của chính chuỗi.

### 4.4. Hậu xử lý
Mọi dự báo của mọi mô hình bị **cắt tại 0**: ŷ = max(ŷ, 0). **Không làm tròn.** Số dự báo âm trước khi cắt được ghi ở `timings.json`
(chỉ ETS có: **16.474** trên 853.720 điểm Test, tức 1,9%).

---

## 5. Năm mô hình chi tiết

### 5.1. Seasonal Naive (mốc đối chuẩn)
- **Ý tưởng:** ngày thứ Hai tới giống thứ Hai tuần trước... Lặp lại tuần cuối: ŷ(T+h) = y(T−6+((h−1) mod 7)).
- **Ví dụ:** tuần cuối `[2,0,1,3,0,1,4]` → dự báo 28 ngày là chuỗi này lặp 4 lần.
- **Tham số:** không có. Thời gian < 1 giây cho 30.490 chuỗi.
- **Vì sao RMSSE ≈ 1:** mẫu số của RMSSE là sai số của dự báo ngây thơ một bước trên Train; Seasonal Naive cũng là một dự báo ngây thơ (lặp tuần trước) nên sai số cùng cỡ → RMSSE ≈ 1 (0,997 Test, 0,976 Validation).
- **Điểm yếu:** sao chép luôn nhiễu và các số 0 của tuần cuối.
- Vai trò: **mốc để tính % cải thiện**. Mọi mô hình khác phải thắng nó mới có ý nghĩa.

### 5.2. ETS (AutoETS, statsforecast)
- **ETS = Error, Trend, Seasonal.** Họ mô hình làm mượt mũ. Ký hiệu: A = cộng (additive), N = không có, Ad = cộng có tắt dần (damped), M = nhân.
  - **ETS(A,N,N)** = làm mượt mũ đơn giản: mức l(t) = α·y(t) + (1−α)·l(t−1); dự báo phẳng bằng mức hiện tại.
  - **ETS(A,N,A)** = thêm thành phần mùa vụ tuần cộng (tham số γ).
  - **ETS(A,Ad,A)** = thêm xu hướng tắt dần (β, φ).
- **α (alpha) nhỏ = làm mượt mạnh**: mức dự báo ổn định, ít bị nhiễu ngày hôm qua kéo đi. Hợp chuỗi thưa nhiều nhiễu.
- **Chọn cấu trúc tự động** cho từng chuỗi theo **AICc** (AIC có hiệu chỉnh cho mẫu hữu hạn; chọn mô hình cân bằng độ khớp và độ phức tạp). `season_length = 7`.
- Mỗi chuỗi fit riêng. Không chuỗi nào phải fallback.
- **Thống kê thực nghiệm (run Validation):**

| Cấu trúc | Số chuỗi | Tỷ lệ |
|---|---:|---:|
| ETS(A,N,A) | 15.989 | 52,4% |
| ETS(A,N,N) | 10.917 | 35,8% |
| ETS(A,Ad,A) | 2.319 | 7,6% |
| ETS(A,Ad,N) | 1.103 | 3,6% |
| ETS(A,A,N) | 94 | 0,3% |
| ETS(A,A,A) | 68 | 0,2% |

  α trung bình 0,066 (trung vị 0,041); γ trung vị ≈ 0 (chu kỳ tuần gần như cố định); khoảng 40% chuỗi **không** chọn thành phần mùa. **Toàn bộ cấu trúc chọn đều có sai số cộng.**
- **Dự báo có thể âm** (mô hình cộng) → 16.474 điểm Test bị cắt về 0.
- **Chi phí:** ≈ 35.121 giây CPU tổng, ≈ 8.794 giây thực (≈ 2,5 giờ) với 4 tiến trình. Đắt nhất. Là điểm yếu duy nhất.
- **Vì sao ETS mạnh ở cấp đáy:** làm mượt riêng từng chuỗi + giữ chu kỳ tuần, lại ít nhiễu hơn học máy trên chuỗi thưa.

### 5.3. Croston SBA (hàng bán thưa)
- **Vấn đề:** làm mượt thông thường trên chuỗi nhiều số 0 bị nhiễu bởi các số 0.
- **Ý tưởng Croston (1972):** **chỉ cập nhật khi có nhu cầu** (y > 0). Tách làm hai ước lượng:
  - z = lượng bán *khi có bán* (làm mượt: z ← z + α(y − z))
  - p = *khoảng cách* (số ngày) giữa hai lần có bán (p ← p + α(q − p), q = số ngày từ lần bán trước)
  - Dự báo phẳng: **ŷ = z / p** (lượng bán trung bình mỗi ngày).
- **SBA (Syntetos–Boylan 2005):** z/p bị **chệch lên**; nhân hệ số hiệu chỉnh **(1 − α/2)**. Với α = 0,1 hệ số = **0,95**: ŷ = 0,95 · z/p. (`statsforecast.CrostonSBA`.)
- **Ví dụ minh họa** (α = 0,1, chuỗi `0,0,4,0,0,0,2,0,3`; thư viện có thể khởi tạo hơi khác): lần bán đầu z=4, p=3 → sau lần bán 2: z=3,8, p=3,1 → sau lần bán 3: z=3,72, p=2,99. Croston cổ điển ŷ = 3,72/2,99 ≈ 1,244; SBA ≈ 0,95 × 1,244 ≈ **1,182**.
- **Điểm mạnh:** gần ETS ở nhóm bán thưa, nhưng chỉ ≈ **10 giây** (nhanh hơn ETS ≈ **3.350 lần**).
- **Điểm yếu then chốt:** dự báo là **một đường phẳng**, **không có chu kỳ tuần**. Hậu quả:
  - Ở cấp đáy: kém ETS ở hàng bán đều (chu kỳ tuần rõ).
  - Ở cấp **tổng hợp**: cộng các đường phẳng vẫn là đường phẳng, mất hẳn chu kỳ tuần của doanh số tổng → WRMSSE Test **1,095**, thậm chí tệ hơn Seasonal Naive (0,870). Cấp 1–9: 1,11–1,22.
- Ví dụ từ pilot (Croston cổ điển): chuỗi cực thưa có p vài trăm ngày nên ŷ ≈ 0,003–0,010.

### 5.4. LightGBM (mô hình global)
- **Gradient boosting (Friedman 2001):** xây dựng dần các **cây quyết định nhỏ**, mỗi cây học **phần sai số còn lại** của các cây trước:
  F_m(x) = F_{m−1}(x) + η · f_m(x) (η = learning rate).
- **LightGBM (Ke 2017):** phiên bản nhanh nhờ GOSS (giữ mẫu gradient lớn) và EFB (gộp đặc trưng loại trừ nhau); chia cây theo lá (leaf-wise).
- **Cấu hình nhóm dùng:** **mặc định thư viện** (lightgbm 4.6.0): 100 cây, learning rate 0,1, num_leaves 31, hàm mất mát bình phương (L2). Chỉ đặt `random_state=42`, số luồng 4, khai báo biến phân loại. **Không tuning, không early stopping.**
- **Một mô hình cho mọi 30.490 chuỗi** (56 triệu dòng huấn luyện). fit+predict ≈ 228 giây (Test), cộng ≈ 46 giây dựng bảng đặc trưng.
- **Importance (gain, run Validation):** `rolling_mean_28` **68,5%**, `rolling_mean_7` **18,2%**, `lag_28` 3,9%, `sell_price` 1,9%, `wday` 1,3%, `rolling_std_28` 1,2%, `month` 1,1%, `lag_35` 0,9%.
  → Dự báo chủ yếu là **mức doanh số trượt** điều chỉnh theo lịch. Importance chỉ nói đóng góp vào *dự báo*, **không phải tác động nhân quả** (đừng nói "giá không quan trọng").
- **Vì sao kém ETS ở cấp đáy nhưng tốt ở cấp tổng hợp:** học chung nên nắm rất chính xác mức và mùa vụ của **tổng**; nhưng với từng chuỗi thưa, làm mượt riêng của ETS tốt hơn.

### 5.5. Stacking lai (ETS + Croston SBA + LightGBM)
- **Ý tưởng:** thay vì chọn một mô hình, **học trọng số** để gộp ba dự báo.
  ŷ = max(0, β₀ + β₁·ŷ_ETS + β₂·ŷ_SBA + β₃·ŷ_LGBM)
- **Meta-learner:** hồi quy **Ridge** (hồi quy tuyến tính có phạt L2, chống quá khớp khi các biến tương quan cao như ETS và SBA).
- **Cách khớp β (quan trọng, hay bị hỏi "có rò rỉ không?"):**
  1. Fit ba mô hình cơ sở đến **mốc 1829** rồi dự báo 28 ngày kế (d_1830–d_1857); lặp với **mốc 1857** (d_1858–d_1885). Nhãn chỉ tới d_1885: **không chạm Validation/Test.**
  2. Gộp lại: 2 mốc × 30.490 chuỗi × 28 ngày = **1.707.440 dòng** dự báo ngoài mẫu (OOF = out-of-fold).
  3. Chọn hệ số phạt alpha ∈ {0,1; 1; 10; 100; 1000} bằng **leave-one-origin-out** (huấn luyện trên một mốc, kiểm trên mốc kia).
  4. Bộ gộp khớp **một lần** rồi áp nguyên cho Validation và Test. Điểm Validation/Test không dùng để khớp bộ gộp.
- **Trọng số học được:** ETS **0,543**, LightGBM **0,371**, Croston SBA **0,138**, hệ số chặn −0,005 (≈ 0). Tổng ≈ 1,05 → gần như trung bình có trọng số.
- **Ví dụ:** ETS = 1,0; SBA = 0,8; LightGBM = 1,2 → ŷ = 0,5429·1,0 + 0,1377·0,8 + 0,3713·1,2 − 0,005 ≈ **1,094**.
- **Giải thích trọng số:** ETS và SBA tương quan cao nên SBA nhận trọng số nhỏ; LightGBM mang thông tin global và chu kỳ tuần mà hai mô hình cục bộ thiếu.
- **alpha chọn = 0,1** (biên dưới lưới) nhưng sai số kiểm chéo giữa các alpha gần như bằng nhau (2,23535 đến 2,23536) → lựa chọn này **không ảnh hưởng** kết quả.
- **Chi phí:** bước khớp chỉ ≈ 12 giây, nhưng **cần đủ ba mô hình cơ sở** (và hai backtest) nên tổng chi phí (≈ 35.372 giây CPU) lớn hơn ETS.
- **Hạn chế:** chỉ có hai mốc backtest (chưa kiểm độ ổn định trọng số theo thời gian); bộ gộp tối ưu theo sai số bình phương **cấp đáy**, không tối ưu WRMSSE.

---

## 6. Các độ đo (phải hiểu thật kỹ)

Ký hiệu: e(i,h) = y(i,T+h) − ŷ(i,T+h), H = 28.

### 6.1. MAE
MAE_i = (1/H) Σ|e|. Cùng đơn vị với doanh số (sản phẩm/ngày), dễ hiểu. Ví dụ MAE = 1,04 = trung bình lệch ≈ 1 sản phẩm/ngày.

### 6.2. RMSE
RMSE_i = √((1/H) Σ e²). Phạt **nặng** sai số lớn.

### 6.3. RMSSE (độ đo xếp hạng chính ở cấp đáy)
RMSSE_i = √( MSE dự báo / MSE của dự báo ngây thơ một bước trên Train )

- Mẫu số: (1/(n−s)) Σ_{t=s+1}^{n} (y(t) − y(t−1))², với n = origin, s = ngày bán khác 0 đầu tiên (**bỏ các số 0 ở đầu**).
- Ý nghĩa: **< 1** nghĩa là dự báo tốt hơn dự báo "lấy hôm qua làm ngày mai" trên Train. **Chuẩn hóa theo từng chuỗi** nên cộng/so được giữa chuỗi bán nhiều và bán ít.
- **Ví dụ:** Train `0,0,3,0,2,1` → bỏ 0 đầu còn `3,0,2,1` → hiệu `−3,2,−1` → bình phương `9,4,1` → trung bình 4,667. Nếu MSE dự báo = 1,2 → RMSSE = √(1,2/4,667) ≈ **0,507**.
- **Vì sao không dùng MAPE:** chia cho doanh số thực = 0 → vô nghĩa với hàng thưa.
- **Macro-RMSSE** = **trung bình** RMSSE trên 30.490 chuỗi (mọi chuỗi đều xác định cả hai tập). Đây là **độ đo xếp hạng chính ở cấp đáy.**

### 6.4. R²
R²_i = 1 − SSE/SST trên 28 ngày. **Có thể âm** (dự báo kém hơn lấy trung bình của chính nhãn). Với chuỗi thưa, 28 ngày có phương sai nhỏ nên R² macro đều âm (−0,18 đến −1,37), **không phải lỗi**. **812 chuỗi Test** (1.587 chuỗi Validation) có nhãn **hằng** (toàn 0) → SST = 0 → R² **không xác định**, ghi NaN, loại khỏi trung bình và đếm riêng.

### 6.5. WRMSSE (độ đo của M5, 12 cấp)
1. Cộng dự báo cấp đáy lên **12 cấp** (tổng 42.840 chuỗi):

| Cấp | Gộp theo | Số chuỗi |
|---:|---|---:|
| 1 | Tổng toàn bộ | 1 |
| 2 | Bang | 3 |
| 3 | Cửa hàng | 10 |
| 4 | Ngành hàng | 3 |
| 5 | Nhóm hàng | 7 |
| 6 | Bang × ngành hàng | 9 |
| 7 | Bang × nhóm hàng | 21 |
| 8 | Cửa hàng × ngành hàng | 30 |
| 9 | Cửa hàng × nhóm hàng | 70 |
| 10 | Sản phẩm | 3.049 |
| 11 | Sản phẩm × bang | 9.147 |
| 12 | Sản phẩm × cửa hàng (cấp đáy) | 30.490 |

2. Mỗi chuỗi ở mỗi cấp có RMSSE riêng (mẫu số tính trên chuỗi đã cộng lên).
3. **Trọng số = doanh thu** (số lượng × giá) của **28 ngày cuối trước origin**, chuẩn hóa trong từng cấp.
4. WRMSSE của mỗi cấp = tổng có trọng số các RMSSE; **WRMSSE tổng = trung bình 12 cấp** (`scripts/wrmsse.py`).
- **Kiểm tra cài đặt (để chứng minh code đúng):** dự báo hoàn hảo → 0; Seasonal Naive trên Test → 0,870; dự báo toàn 0 → 5,447.
- **Khác biệt RMSSE vs WRMSSE:** RMSSE macro = trung bình **cấp đáy**, mọi chuỗi như nhau. WRMSSE = trung bình **12 cấp**, có trọng số doanh thu, thưởng độ chính xác ở cấp tổng hợp. Hai độ đo xếp hạng **khác nhau**, và đó chính là Q5.

### 6.6. Macro và pooled
- **Macro:** tính độ đo từng chuỗi rồi **lấy trung bình** (chuỗi nào cũng một phiếu).
- **Pooled:** gộp mọi cặp (chuỗi, ngày) rồi tính một lần (chuỗi bán nhiều chi phối).
- Xếp hạng chính theo **macro-RMSSE**; pooled báo riêng.

### 6.7. Chính sách chốt **trước** khi chạy Test
- Mẫu số RMSSE = 0 → RMSSE không xác định; nhãn hằng → R² không xác định.
- **Không thay bằng epsilon.** Ghi NaN kèm lý do, loại khỏi trung bình, đếm riêng.
- Kết quả: 0 chuỗi RMSSE không xác định; 812 chuỗi (Test) R² không xác định.

### 6.8. Cách đo thời gian (dễ bị hiểu sai)
- **Mô hình cục bộ:** tổng thời gian fit+predict **từng chuỗi** (đơn luồng mỗi chuỗi; chạy 4 tiến trình song song). Số "thực" (wall) = tổng/4 xấp xỉ.
- **LightGBM:** thời gian fit+predict toàn bộ dữ liệu với 4 luồng; dựng đặc trưng và đọc dữ liệu tính riêng.
- **Stacking:** chỉ bước khớp bộ gộp; cộng chi phí ba mô hình cơ sở thì được số "gồm cả mô hình cơ sở".
- Hai nhóm **không cùng phạm vi** nên không so trực tiếp từng giây.
- Phần cứng: Kaggle CPU 4 lõi, ≈ 30 GB RAM, Python 3.13.

---

## 7. Cách làm / quy trình thực nghiệm (từ đầu đến cuối)

### 7.1. Sơ đồ tổng quát
```
Dữ liệu gốc (5 CSV)
  → src/m5/data.py (đọc, kiểm tra)
  → panel.py (ma trận chuỗi × ngày + lịch + giá + SNAP)
  → split.py / schema.py (chia Train/Val/Test; lược đồ dự báo)
  ├─ Mô hình cục bộ: seasonal_naive / ets / croston_sba (univariate.py, theo lô 50 chuỗi, có checkpoint)
  ├─ Mô hình global: features.py + global_models.py (LightGBM)
  └─ Backtest trong Train (mốc 1829, 1857) → OOF → run_stacking.py (Ridge)
  → evaluate.py (MAE, RMSE, RMSSE, R²)   + wrmsse.py (12 cấp)
  → collect_runs.py → analyze_results.py → bootstrap_ci.py → robustness_classes.py
  → build_report.py (bảng) / build_slides.py / latex_figures.py (hình)
```

### 7.2. Bốn job Kaggle (chạy song song)
Mỗi job một notebook gọi `scripts/kaggle_full_run.py --task …`:

| Notebook | Việc | Ghi chú |
|---|---|---|
| `06a_full_validation.ipynb` | Validation (origin d_1885), 4 mô hình cơ sở | nhánh `full-data-5-models` |
| `06b_full_oof_1829.ipynb` | Backtest mốc 1829 | ETS, SBA, LightGBM |
| `06c_full_oof_1857.ipynb` | Backtest mốc 1857 | |
| `06d_full_test.ipynb` | **Test** (origin d_1913) | clone **đúng tag `config-freeze`** (mã đã khóa) |

- ETS mất ≈ 2,5 giờ mỗi job. Ba mô hình còn lại rất nhanh.
- Cả bốn job dùng **cùng mã** (commit `c8ba5c8`), **cùng năm CSV** (SHA-256 trùng, ghi ở `data_sha256.json`), **cùng danh sách 30.490 chuỗi** (hash `56a6267c…`).
- Môi trường: numpy 2.1.3, pandas 2.3.3, scikit-learn 1.6.1, statsforecast 2.1.1, LightGBM 4.6.0 (đúng bản ghim ở `requirements.txt`).
- **Stacking chạy trên máy cục bộ** (chỉ vài giây) sau khi tải về dự báo của Validation/Test và hai file OOF.

### 7.3. Các lệnh chính (README mục 12)
```powershell
python scripts/run_stacking.py --split validation --oof "results/full/oof/oof_*.parquet" --lgbm default --bases ets croston_sba lightgbm --base-runs results/full/runs/val-<run> --manifest data/full_series_ids_with_class.csv --out results/full/runs
python scripts/collect_runs.py --split validation --runs results/full/runs/val-* --out results/full/tables/validation
python scripts/analyze_results.py --split validation --tables results/full/tables/validation --figures results/full/figures/validation --classes data/full_series_ids_with_class.csv
python scripts/bootstrap_ci.py --split validation --runs results/full/runs/val-* --out results/full/tables/validation
python scripts/wrmsse.py --split validation --runs results/full/runs/val-* --out results/full/tables/validation
python scripts/robustness_classes.py --split test --runs results/full/runs/tes-* --out results/full/tables/test --manifest data/full_series_ids_with_class.csv
python scripts/build_report.py
```
(lặp lại với `test`; `run_stacking.py --split test` cần tag `config-freeze-v2`.)

### 7.4. Truy vết (traceability)
Mỗi lần chạy có một `run_id` dạng `tes-20261118T054917Z-c8ba5c8` (split, thời điểm UTC, SHA commit). Thư mục `results/full/runs/<run_id>/` có:
`predictions.parquet` (dự báo), `metrics_per_series.csv`, `metrics_summary.csv`, `timings.json`, `environment.json` (phiên bản thư viện, git SHA, CPU),
`data_sha256.json`, `model_parameters.json`, và file `diag_*` (cấu trúc ETS, importance LightGBM, hệ số stacking).
**Mọi bảng/hình trong báo cáo truy được về `run_id`**; bảng sinh bằng `scripts/build_report.py`, không sửa số bằng tay.

| run_id | split | Mô hình | Mã |
|---|---|---|---|
| `val-20261118T054238Z-c8ba5c8` | validation (origin d_1885) | Seasonal Naive, ETS, Croston SBA, LightGBM | c8ba5c8 |
| `val-20261118T083718Z-867b564-stack` | validation | Stacking lai | 867b564 (cục bộ) |
| `tes-20261118T054917Z-c8ba5c8` | test (origin d_1913) | Seasonal Naive, ETS, Croston SBA, LightGBM | c8ba5c8 |
| `tes-20261118T083859Z-867b564-stack` | test | Stacking lai | 867b564 (cục bộ) |

### 7.5. Kiểm thử
`tests/` có 17 tệp: metrics (RMSSE, R², trường hợp suy biến), **không rò rỉ**, schema/dự báo, seasonal naive, ETS/Croston, WRMSSE, bootstrap, khóa Test (`test_freeze.py`), khóa chạy lại (`test_run_lock.py`)... Chạy: `python -m unittest discover -s tests -v`.
Các kiểm thử dùng dữ liệu giả để kiểm mã; **không thay** bằng chứng chạy trên dữ liệu thật.

### 7.6. Cấu trúc thư mục repo
```
configs/      data.yaml, features.yaml, freeze.yaml (khóa), sample.yaml, models/*.yaml
data/         full_series_ids_with_class.csv (30.490 chuỗi + nhóm nhu cầu), sample_series_ids.csv (pilot 1.200)
notebooks/    06a–06d (nghiên cứu chính), 02–05 (pilot), 00/01 (kiểm tra dữ liệu, C03 LightGBM)
paper/        summary.md (tóm tắt M5), lightgbm_related_work.md
report/       draft_report.md (bản thảo), qa.md (hỏi đáp), slides/m5_nhom6.pptx, handoff/ (giải thích từng mô hình), latex_figures/
results/full/ runs/, oof/, tables/{validation,test}, figures/   ← nghiên cứu chính
results/      runs/, tables/, figures/ ← pilot 1.200 chuỗi
scripts/      run_experiment, kaggle_full_run, backtest_oof, run_stacking, wrmsse, bootstrap_ci, analyze_results, build_report, ...
src/m5/       data, panel, features, global_models, univariate, evaluate, metrics, schema, split, seasonal_naive, optional, sample
tests/        17 tệp kiểm thử
```

---

## 8. Kết quả đầy đủ

### 8.1. Tổng thể (Q1) — macro trên 30.490 chuỗi

**Validation (d_1886–d_1913)**

| Mô hình | MAE | RMSE | RMSSE | R² | So Seasonal Naive (RMSSE) | fit+predict (s) |
|---|---:|---:|---:|---:|---:|---:|
| Seasonal Naive | 1,232 | 1,740 | 0,976 | −1,535 | 0% | 0,7 |
| ETS | 1,039 | 1,346 | 0,737 | −0,242 | −24,5% | 34.832 |
| Croston SBA | 1,085 | 1,391 | 0,759 | −0,345 | −22,2% | 8,7 |
| LightGBM | 1,063 | 1,354 | 0,757 | −0,346 | −22,4% | 255,0 |
| Stacking lai | 1,046 | 1,332 | 0,738 | −0,243 | −24,4% | 15,2 |

**Test (d_1914–d_1941, refit d_1–d_1913, chạy một lần)**

| Mô hình | MAE | RMSE | RMSSE | R² | So Seasonal Naive (RMSSE) | fit+predict (s) |
|---|---:|---:|---:|---:|---:|---:|
| Seasonal Naive | 1,244 | 1,752 | 0,997 | −1,371 | 0% | 0,7 |
| ETS | 1,040 | 1,360 | 0,758 | −0,184 | −23,9% | 35.121 |
| Croston SBA | 1,075 | 1,393 | 0,769 | −0,422 | −22,9% | 10,5 |
| LightGBM | 1,079 | 1,386 | 0,775 | −0,422 | −22,3% | 228,4 |
| Stacking lai | 1,050 | 1,351 | 0,757 | −0,235 | −24,1% | 12,1 |

**Pooled (Test):** RMSE gộp: Seasonal Naive 2,677; ETS 2,174; Croston SBA 2,232; LightGBM 2,209; **Stacking 2,124** (thấp nhất).

**Tỷ lệ chuỗi có MAE thấp hơn Seasonal Naive (Test):** ETS 65,0%; Croston SBA 64,1%; LightGBM 60,0%; Stacking 63,0% → cải thiện đến từ **giảm sai số trung bình**, không phải thắng mọi chuỗi.

### 8.2. Thứ hạng và độ chắc chắn (bootstrap)
- **Bootstrap theo chuỗi:** lấy lại (có hoàn lại) 30.490 chuỗi, **2.000 lần**, seed 42 → khoảng tin cậy 95%.

| Mô hình (Test) | RMSSE | KTC 95% | Chênh so ETS | KTC 95% chênh |
|---|---:|---|---:|---|
| Seasonal Naive | 0,997 | [0,992; 1,002] | +0,2388 | [+0,2359; +0,2417] |
| ETS | 0,758 | [0,755; 0,762] | — | — |
| Croston SBA | 0,769 | [0,765; 0,773] | +0,0107 | [+0,0093; +0,0121] |
| LightGBM | 0,775 | [0,771; 0,779] | +0,0164 | [+0,0152; +0,0177] |
| Stacking lai | 0,757 | [0,753; 0,761] | −0,0015 | [−0,0020; −0,0009] |

- **Chênh lệch cặp (âm = mô hình thứ nhất tốt hơn):** Test: SBA − LightGBM = −0,0057; Stacking − LightGBM = −0,0179; Stacking − SBA = −0,0122 (đều không chứa 0). Validation: **SBA − LightGBM = +0,0020** (LightGBM hơn SBA, ngược Test).
- **Vì sao vẫn không coi là bền vững:** KTC chỉ phản ánh biến thiên **giữa các chuỗi trong một cửa sổ 28 ngày**. Các chuỗi không độc lập (cùng cửa hàng, cùng ngày) nên khoảng thực tế rộng hơn. Trên Validation thứ tự hai cặp đảo: ETS thấp hơn stacking 0,0009; LightGBM thấp hơn SBA 0,0020.
- **Kết luận vững:** (a) bốn mô hình tốt hơn rõ Seasonal Naive; (b) cặp {ETS, stacking} tốt hơn cặp {Croston SBA, LightGBM} khoảng 0,01–0,02 ở **cả hai** cửa sổ.

### 8.3. WRMSSE và các cấp tổng hợp (Q5)

| Mô hình | WRMSSE Val | WRMSSE Test | Cấp 1 (tổng) Test | Cấp 3 (cửa hàng) Test | Cấp 9 Test | Cấp 12 (đáy) Test |
|---|---:|---:|---:|---:|---:|---:|
| Seasonal Naive | 0,923 | 0,870 | 0,751 | 0,799 | 0,892 | 1,100 |
| ETS | 0,724 | 0,760 | 0,687 | 0,731 | 0,790 | 0,845 |
| Croston SBA | 1,121 | 1,095 | 1,186 | 1,144 | 1,118 | 0,862 |
| LightGBM | **0,591** | 0,722 | 0,610 | 0,685 | 0,759 | 0,857 |
| Stacking lai | 0,715 | **0,668** | **0,505** | **0,620** | **0,730** | **0,834** |

Giải thích:
- **Croston SBA:** tốt ở cấp 12 (0,862, gần ETS và LightGBM) nhưng **kém nhất ở mọi cấp tổng hợp** (cấp 1–9: 1,11–1,22 trên Test, tệ hơn cả Seasonal Naive): dự báo phẳng khi cộng lại vẫn phẳng, mất chu kỳ tuần của tổng.
- **LightGBM:** kém nhất trong bốn mô hình ở cấp đáy (Test) nhưng rất tốt ở cấp tổng hợp (cấp 1: 0,610 Test, **0,330** Validation) vì học chung mức và mùa vụ của mọi chuỗi.
- **Stacking:** tận dụng cả hai, WRMSSE Test thấp nhất (0,668). Trên **Validation**, LightGBM có WRMSSE thấp nhất (0,591) và stacking thứ hai (0,715), vì bộ gộp khớp theo sai số bình phương cấp đáy chứ không tối ưu WRMSSE.
- **Thang đối chiếu:** đội thắng M5 đạt 0,520 trên bộ chấm công bố; điểm của nhóm không phải bài nộp thi, chỉ cùng thước đo (cùng cửa sổ public leaderboard).

### 8.4. Theo nhóm nhu cầu, ngành hàng, bang (Q2) — RMSSE macro (Test)

| Nhóm nhu cầu | Seasonal Naive | ETS | Croston SBA | LightGBM | Stacking |
|---|---:|---:|---:|---:|---:|
| `high_zero` (> 0,8; 11.907 chuỗi) | 1,051 | 0,795 | 0,801 | 0,820 | 0,796 |
| `mid_zero` (0,5–0,8; 12.053) | 0,987 | 0,750 | 0,759 | 0,757 | 0,746 |
| `low_zero` (≤ 0,5; 6.530) | 0,918 | 0,708 | 0,730 | 0,725 | 0,705 |

| Ngành hàng | Seasonal Naive | ETS | Croston SBA | LightGBM | Stacking |
|---|---:|---:|---:|---:|---:|
| FOODS | 1,003 | 0,777 | 0,786 | 0,790 | 0,772 |
| HOBBIES | 0,982 | 0,736 | 0,741 | 0,756 | 0,738 |
| HOUSEHOLD | 0,997 | 0,745 | 0,761 | 0,764 | 0,747 |

| Bang | Seasonal Naive | ETS | Croston SBA | LightGBM | Stacking |
|---|---:|---:|---:|---:|---:|
| CA | 1,013 | 0,769 | 0,784 | 0,784 | 0,768 |
| TX | 0,954 | 0,729 | 0,736 | 0,748 | 0,728 |
| WI | 1,019 | 0,774 | 0,783 | 0,789 | 0,771 |

- **Hàng bán rất thưa (> 0,8):** ETS (0,795), stacking (0,796), Croston SBA (0,801) sát nhau và **đều tốt hơn rõ LightGBM (0,820)**.
- **Hàng bán đều (≤ 0,5):** stacking (0,705) và ETS (0,708) dẫn, LightGBM 0,725, **Croston SBA kém nhất (0,730)** vì dự báo phẳng bỏ qua chu kỳ tuần (rõ hơn ở hàng bán đều).
- HOBBIES sai số chuẩn hóa thấp nhất, FOODS cao nhất; TX thấp nhất. Khác biệt giữa nhóm chỉ **mô tả**, không diễn giải nhân quả.
- **Kiểm tra độ bền (nhóm tính sau lần bán đầu tiên, Bảng 7d):** ở `high_zero` mới (7.486 chuỗi): ETS 0,800, stacking 0,805, SBA 0,806, **LightGBM 0,837** → kết luận không đổi.

### 8.5. Theo thứ trong tuần và horizon (Test)
- Mọi mô hình sai nhiều hơn vào **thứ Bảy, Chủ nhật** (ngày bán cao nên sai số tuyệt đối lớn). MAE gộp thứ Bảy: Seasonal Naive 1,390; ETS 1,148; Croston SBA 1,183; LightGBM 1,212; Stacking 1,164. Chủ nhật tương tự (ETS 1,177).
- Sai số **tăng từ tuần 1 đến tuần 3 rồi giảm ở tuần 4** (MAE theo tuần của ETS: 0,969; 1,049; 1,093; 1,047).
- ETS có MAE gộp thấp nhất ở **22/28 horizon**; Croston SBA thấp nhất ở horizon 14.
- MAE theo horizon (ETS): h=1: 0,917; h=7: 1,119; h=14: 1,122; h=21: 1,278; h=28: 1,190.

### 8.6. Chi phí tính toán (Q3)

| Mô hình | Thời gian (Test) | Ghi chú |
|---|---|---|
| Seasonal Naive | 0,7 giây | |
| Croston SBA | 10,5 giây CPU tổng | ≈ 4,5 giây thực |
| LightGBM | 228,4 giây | 4 luồng; + ≈ 46 giây dựng 56 triệu dòng đặc trưng |
| ETS | 35.121 giây CPU tổng | **≈ 8.794 giây thực (≈ 2,5 giờ)** với 4 tiến trình |
| Stacking | 12,1 giây (bước khớp) | Gồm ba mô hình cơ sở: ≈ 35.372 giây |

- **Croston SBA đạt macro-RMSSE 0,769 trong 10,5 giây, kém ETS (0,758) chỉ 0,011, nhưng nhanh gấp ≈ 3.350 lần.** Với hàng bán không đều ở cấp từng chuỗi, đây là lựa chọn nên thử trước.
- Đổi lại: stacking giảm WRMSSE từ 0,760 (ETS) xuống 0,668 nhưng tốn tổng chi phí lớn hơn ETS.
- **Chưa có chi phí tồn kho** nên **không kết luận lợi ích kinh doanh**.

### 8.7. So với pilot 1.200 chuỗi
Trên Test, macro-RMSSE toàn bộ cao hơn pilot khoảng 0,006–0,007: ETS 0,758 vs 0,752; Croston SBA 0,769 vs 0,762; LightGBM 0,775 vs 0,768; Seasonal Naive không đổi (0,997). **Thứ tự bốn mô hình cơ sở giống pilot.** Mẫu pilot lệch nhẹ về chuỗi gián đoạn (43,3% vs 39,1%) nên kết quả toàn bộ dữ liệu đại diện hơn; thứ hạng giữ nguyên cho thấy 1.200 chuỗi đủ để chọn mô hình.

---

## 9. Giải thích sâu: vì sao kết quả như vậy (Q4)

| Hiện tượng | Giải thích |
|---|---|
| ETS tốt ở cấp đáy, đặc biệt hàng thưa | α trung bình nhỏ (0,066) → mức dự báo rất mượt; γ ≈ 0 → chu kỳ tuần ổn định; không bị nhiễu bởi từng số 0. |
| ETS đắt | AutoETS thử nhiều cấu trúc cho từng chuỗi (30.490 lần tìm cấu trúc theo AICc), đơn luồng mỗi chuỗi. |
| Croston SBA rẻ và gần ETS ở hàng thưa | Chỉ cập nhật khi có bán, công thức đóng. |
| Croston SBA tệ ở cấp tổng hợp | Dự báo phẳng không có chu kỳ tuần, cộng lại vẫn phẳng. |
| LightGBM tốt ở cấp tổng hợp | Học chung mức + mùa vụ của mọi chuỗi từ lịch và đặc trưng trễ; sai số từng chuỗi triệt tiêu khi cộng. |
| LightGBM kém ETS ở từng chuỗi thưa | 86,7% gain đến từ hai trung bình trượt → dự báo ≈ mức trượt điều chỉnh lịch, thiếu độ mượt riêng của từng chuỗi; cấu hình mặc định, không tuning. |
| Stacking tốt nhất ở Test | Trộn ETS (mượt, đúng cấp đáy) và LightGBM (mùa vụ chung); hai nguồn sai số khác nhau nên gộp giảm sai số. |
| Stacking ≈ ETS ở cấp đáy | ETS chiếm 54% trọng số và hai mô hình rất tương quan; lợi ích gộp nhỏ ở cấp đáy, lớn ở cấp tổng hợp. |
| Seasonal Naive RMSSE ≈ 1 | Cùng bản chất với mẫu số của RMSSE (dự báo ngây thơ). |

Lưu ý ngôn ngữ: dùng "phù hợp với", "gợi ý" khi giải thích cơ chế; **không** nói "chứng minh" hay nguyên nhân chắc chắn (importance, so sánh nhóm đều là quan sát).

---

## 10. Hình và sơ đồ trong báo cáo
- Hình kết quả (Test): `results/full/figures/test/` — RMSSE theo nhóm nhu cầu, MAE theo horizon, MAE theo thứ, đánh đổi độ chính xác–thời gian, ví dụ thực tế–dự báo.
- Ví dụ thực tế–dự báo chọn **theo quy tắc khách quan**: ba chuỗi ở phân vị 10/50/90 của doanh số trung bình trên Train (không chọn tay để "đẹp").
- Hình importance LightGBM: `results/full/figures/m10_feature_importance.png`.
- **Năm sơ đồ cấu trúc mô hình** cho báo cáo LaTeX (Draw.io, mỗi mô hình một trang, kèm PDF vector): `report/latex_figures/M5_Model_Structures.drawio`; trong Overleaf là `figures/structure_{seasonal_naive,autoets,croston_sba,lightgbm,stacking}.pdf`.
- Báo cáo cuối viết bằng **LaTeX trên Overleaf** (hai project: tiếng Việt và tiếng Anh); `report/draft_report.md` là nguồn số liệu/nội dung.
- Slide: `report/slides/m5_nhom6.pptx` (17 slide).

---

## 11. Giới hạn (phải tự nói ra, đừng đợi bị hỏi)
1. **Một cửa sổ Test 28 ngày**; các chênh lệch nhỏ đổi chiều giữa Validation và Test.
2. **Khác biệt input:** mô hình cục bộ chỉ có lịch sử doanh số; LightGBM có thêm giá/lịch/SNAP/nhóm.
3. **Giả định giá ngày dự báo được biết trước.**
4. **KTC bootstrap** chỉ phản ánh biến thiên giữa chuỗi trong một cửa sổ; chuỗi không độc lập nên khoảng thực tế rộng hơn.
5. **Runtime local và global khác phạm vi**; stacking chạy trên máy khác với các mô hình cơ sở.
6. **Test không còn hoàn toàn mù:** nhãn Test của 1.200 chuỗi pilot (3,9%) đã được xem trước khi chọn 5 mô hình.
7. **Stacking:** chỉ hai mốc backtest, khớp theo sai số cấp đáy, không theo WRMSSE; trọng số chưa kiểm độ ổn định theo thời gian.
8. **Phân nhóm nhu cầu** tính cả ngày 0 trước lần bán đầu tiên (đã kiểm độ bền ở Bảng 7d).
9. **Chưa đo chi phí tồn kho** → không kết luận lợi ích kinh doanh.
10. **LightGBM dùng mặc định thư viện**, không tuning, không hàm mất mát Tweedie. Đây là cấu hình đã khóa trước khi chạy Test.
11. **Bản thảo và slide do AI soạn từ kết quả thật; chưa có review đầy đủ của các thành viên.**
12. Chỉ dự báo điểm, không dự báo xác suất.

---

## 12. Hướng phát triển
- Kiểm định cuốn chiếu (**rolling-origin**) trên nhiều mùa để kiểm độ ổn định thứ hạng (ưu tiên số 1).
- Khớp stacking theo WRMSSE hoặc theo từng cấp; thêm mốc backtest.
- Thêm chu kỳ tuần cho Croston (nhân hệ số mùa) để khắc phục điểm yếu ở cấp tổng hợp.
- Khoảng tin cậy có xét phụ thuộc theo thời gian.
- Mô phỏng quyết định tồn kho với chi phí thực; dự báo xác suất.
- Thử hàm mất mát **Tweedie** cho LightGBM (đội thắng M5 dùng) và tuning, đánh giá trên nhiều cửa sổ.

---

## 13. Hỏi đáp dự kiến (đọc câu hỏi, tự trả lời, rồi đối chiếu)

**1. Đồ án làm gì, trong một câu?**
So sánh năm cách dự báo (mốc, ETS, Croston SBA, LightGBM, stacking lai) cho số lượng bán 28 ngày của toàn bộ 30.490 chuỗi sản phẩm–cửa hàng M5, theo độ chính xác (kể cả WRMSSE 12 cấp) và thời gian chạy.

**2. Vì sao chỉ 5 mô hình?**
Pilot 1.200 chuỗi chạy 11 mô hình. Giữ mỗi cách tiếp cận một đại diện. Bỏ XGBoost, CatBoost (ngang LightGBM nhưng chậm, CatBoost ≈ 14 giờ CPU toàn bộ), Croston cổ điển (kém SBA), Linear Regression (kém nhất nhóm học máy), LightGBM Tuned (không giữ lợi thế trên Test).

**3. Vì sao chạy toàn bộ 30.490 chuỗi?**
Bỏ sai số lấy mẫu (mẫu pilot lệch về hàng gián đoạn) và tính được WRMSSE 12 cấp như M5. Thứ tự bốn mô hình cơ sở giống pilot.

**4. Vì sao horizon 28, lag ≥ 28, rolling shift 28?**
Đúng kỳ dự báo M5. Dự báo trực tiếp cả 28 ngày từ một origin, nên ngày T+28 chỉ chắc chắn biết đến T. Lag nhỏ hơn 28 dùng nhãn chưa biết. `tests/test_no_leakage.py` kiểm.

**5. Làm sao chắc không rò rỉ nhãn?**
Nhãn sau origin bị che NaN; hàm đặc trưng từ chối lag < 28; test làm hỏng nhãn sau origin và xác nhận đặc trưng/target không đổi; đánh giá nối theo khóa; stacking khớp chỉ trên dự báo ngoài mẫu trong Train.

**6. RMSSE khác WRMSSE thế nào?**
RMSSE tính từng chuỗi (RMSE chia sai số ngây thơ một bước trên Train, bỏ số 0 đầu), lấy trung bình macro cấp đáy. WRMSSE cộng dự báo lên 12 cấp (42.840 chuỗi), trọng số theo doanh thu 28 ngày cuối trước origin, trung bình 12 cấp. Thứ hạng khác nhau: Croston SBA tốt cấp đáy nhưng WRMSSE 1,095 (kém nhất); LightGBM kém nhất cấp đáy nhưng WRMSSE 0,722.

**7. Vì sao không dùng MAPE?**
Doanh số thực bằng 0 ở 68,4% số ngày → chia cho 0.

**8. Điểm WRMSSE có so được với bảng xếp hạng M5 không?**
Cùng thước đo và cùng cửa sổ public leaderboard (d_1914–d_1941), nhưng không phải bài nộp thi, chỉ một cửa sổ, và nhóm không tái lập hệ thống đội thắng (0,520). Kiểm tra cài đặt: Seasonal Naive cho 0,870, dự báo toàn 0 cho 5,447.

**9. Croston SBA khác Croston cổ điển thế nào?**
Croston cổ điển cho ŷ = z/p, bị chệch lên. SBA nhân (1 − α/2); với α = 0,1 là 0,95. Ở pilot SBA tốt hơn Croston ở cả Validation (0,743 vs 0,745) và Test (0,762 vs 0,764).

**10. Stacking hoạt động thế nào, có rò rỉ không?**
ŷ = max(0, β₀ + β₁·ETS + β₂·SBA + β₃·LightGBM); β khớp bằng Ridge trên dự báo ngoài mẫu trong Train (mốc 1829 và 1857, nhãn tới d_1885). Khớp một lần, áp cho Validation và Test; không dùng nhãn Validation/Test. Trọng số: ETS 0,543, LightGBM 0,371, SBA 0,138.

**11. Mô hình nào tốt nhất?**
Trên Test: stacking theo cả macro-RMSSE (0,757, nhỉnh ETS 0,758) và WRMSSE (0,668). Trên Validation đổi chiều: ETS nhỉnh stacking ở cấp đáy, LightGBM có WRMSSE thấp nhất (0,591). Kết luận vững: ETS và stacking tốt hơn Croston SBA và LightGBM ở cấp đáy khoảng 0,01–0,02. Không tuyên bố "thắng chắc chắn".

**12. Vì sao chênh 0,0015 giữa stacking và ETS vẫn "có ý nghĩa" theo bootstrap mà bạn lại nói không bền vững?**
KTC bootstrap chỉ phản ánh biến thiên giữa 30.490 chuỗi trong **một** cửa sổ 28 ngày; không phản ánh biến thiên theo thời gian. Trên Validation chiều đảo ngược. Do vậy chênh lệch cỡ 0,001–0,006 không được coi là bền vững.

**13. Vì sao LightGBM kém ETS ở cấp đáy nhưng tốt hơn ở cấp tổng hợp?**
Importance dồn vào `rolling_mean_28` (68,5%) và `rolling_mean_7` (18,2%): dự báo là mức trượt điều chỉnh lịch, học chung mọi chuỗi. Cộng theo cửa hàng/bang, mức và chu kỳ tuần chung này rất chính xác (cấp 1: 0,610). Ở từng chuỗi thưa, làm mượt riêng của ETS tốt hơn.

**14. Vì sao Croston SBA kém nhất ở cấp tổng hợp?**
Dự báo của nó là một đường phẳng cho mỗi chuỗi. Tổng các đường phẳng vẫn phẳng nên mất chu kỳ tuần của doanh số tổng (thứ Bảy, Chủ nhật cao). Cấp 1–9 trên Test: 1,11–1,22.

**15. R² âm / chuỗi hằng nghĩa là gì?**
R² âm = dự báo kém hơn trung bình của chính nhãn đánh giá; với chuỗi thưa 28 ngày có phương sai nhỏ nên phổ biến. 812 chuỗi Test có nhãn hằng nên R² không xác định; ghi NaN, đếm riêng, không dùng epsilon.

**16. Chi phí tính toán?**
Kaggle 4 lõi: Seasonal Naive < 1 giây, Croston SBA ≈ 10 giây, LightGBM ≈ 4 phút (+ ≈ 1 phút dựng đặc trưng), ETS ≈ 2,5 giờ thực. Stacking tốn thêm ≈ 12 giây nhưng cần đủ ba mô hình cơ sở và hai backtest.

**17. Vì sao ETS cho dự báo âm?**
ETS chọn toàn cấu trúc sai số cộng; với chuỗi thưa, mức cộng mùa vụ có thể âm. 16.474 điểm dự báo Test (1,9%) bị cắt về 0, như mọi mô hình.

**18. Vì sao cắt về 0 mà không làm tròn?**
Doanh số không âm nên cắt về 0 là hợp lý và áp cho mọi mô hình; MAE/RMSE/RMSSE làm việc tốt với dự báo thực (kỳ vọng); làm tròn sẽ làm mất thông tin.

**19. Giá tương lai?**
Pipeline dùng giá tuần của ngày cần dự báo (M5 công bố giá đến d_1969). Trong vận hành chỉ dùng được nếu giá được lên kế hoạch trước. Đã nêu là giả định.

**20. Test có còn "mù" không?**
Cấu hình khóa bằng tag `config-freeze` (commit `c8ba5c8`) trước khi chạy Test toàn bộ, Test chạy một lần. Nhưng nhãn Test của 1.200 chuỗi pilot (3,9%) đã xem trước khi chọn 5 mô hình; nhóm nêu rõ trong phần giới hạn.

**21. Local hay global?**
Seasonal Naive, ETS, Croston SBA là local (một mô hình mỗi chuỗi, chỉ dùng lịch sử chuỗi đó). LightGBM là global (một mô hình mọi chuỗi, thêm giá/lịch/SNAP/nhóm). Stacking gộp cả hai. Khác biệt input là yếu tố gây nhiễu khi so thứ hạng.

**22. Vì sao chọn Ridge cho stacking mà không trung bình đơn giản hay mô hình phức tạp?**
Ridge có ít tham số (4), diễn giải được trọng số, phạt L2 xử lý ETS–SBA tương quan cao. Trọng số học được gần như một trung bình có trọng số (tổng ≈ 1,05). Mô hình phức tạp hơn với chỉ hai mốc backtest dễ quá khớp. Đây cũng là hạn chế: chưa so với trung bình đều.

**23. Vì sao chỉ hai mốc backtest (1829, 1857)?**
Mỗi mốc cần fit lại ETS (≈ 2,5 giờ). Hai mốc cho phép leave-one-origin-out và vừa trong thời gian/tài nguyên. Hạn chế đã nêu: chưa kiểm độ ổn định của trọng số theo thời gian.

**24. Vì sao Test refit trên d_1–d_1913 chứ không dùng mô hình Validation?**
Test dự báo từ origin d_1913; mô hình cần thấy dữ liệu mới nhất (cả giai đoạn Validation) như trong thực tế vận hành. Cùng cấu hình, chỉ thêm dữ liệu.

**25. Vì sao không dùng deep learning?**
Ngoài phạm vi (tài nguyên CPU Kaggle, giới hạn thời gian). Đội thắng M5 dùng LightGBM; hạng ba dùng ensemble neural. Là hướng phát triển. Đồ án so sánh năm cách đại diện, không chạy đua.

**26. Vì sao LightGBM dùng cấu hình mặc định?**
Cấu hình đã khóa trước khi chạy Test. Optuna ở pilot (LightGBM Tuned) hơn mặc định ở Validation nhưng **không hơn ở Test** (0,770 vs 0,768) nên không đưa vào.

**27. Vì sao bảng chính chưa dùng Tweedie?**
Bảng chính dùng cấu hình đã khóa trước khi chạy Test, LightGBM dùng hàm mất mát bình phương mặc định. Tweedie là lựa chọn của đội thắng M5; nhóm có thể đánh giá trên nhiều cửa sổ giữ lại để kiểm mức cải thiện trước khi đưa vào cấu hình chính thức.

**28. Đồ án đóng góp gì so với M5?**
Đối chiếu năm cách tiếp cận trên cùng 30.490 chuỗi, cùng horizon và độ đo; báo cáo sai số theo mức độ gián đoạn, theo 12 cấp WRMSSE và thời gian. Phát hiện: ở nhóm > 80% ngày bán 0, ETS, stacking và Croston SBA tốt hơn LightGBM; ở cấp tổng hợp thì ngược lại. Đóng góp nằm ở cách so sánh và phân tích theo nhóm và theo cấp, không phải thuật toán mới.

**29. Smoke khác tái lập đầy đủ thế nào?**
Smoke chạy cùng pipeline trên tập nhỏ (12 chuỗi) để kiểm tra chạy được end-to-end, không tái lập số liệu. Nghiên cứu chính chạy toàn bộ trên Kaggle (`notebooks/06a–06d`).

**30. Làm sao tái lập kết quả?**
Cài `requirements.txt` đúng phiên bản ghim; tải 5 CSV từ Kaggle; chạy bốn notebook `06a–06d` trên Kaggle (clone đúng nhánh/tag), tải kết quả; chạy `run_stacking.py` rồi các script thu thập/phân tích (mục 7.3). Mỗi run lưu `environment.json` (phiên bản, git SHA) và `data_sha256.json` (hash CSV). Lưu ý: XGBoost (pilot) phải đúng 3.4.1 vì 3.2.0 cho kết quả khác một chút.

**31. ETS(A,N,A) nghĩa là gì?**
Sai số cộng (A), không có xu hướng (N), mùa vụ cộng (A) — chu kỳ 7 ngày. 52,4% chuỗi chọn cấu trúc này; 35,8% chọn ETS(A,N,N) (không mùa).

**32. Vì sao nói 68,4% ngày bán 0 mà pilot lại 70,1%?**
Mẫu pilot 1.200 chuỗi lệch nhẹ về chuỗi gián đoạn (43,3% chuỗi có > 80% ngày 0, so 39,1% toàn bộ). Vì thế chạy toàn bộ dữ liệu cho ước lượng đại diện hơn.

**33. Làm sao biết bốn mô hình thắng Seasonal Naive không phải do may mắn?**
KTC bootstrap của chênh lệch với ETS không gần 0 (ví dụ Seasonal Naive − ETS = +0,239, KTC [0,236; 0,242]); mức 22–24% cải thiện nhất quán ở cả Validation và Test và ở mọi nhóm nhu cầu/ngành hàng/bang. Tuy vậy vẫn chỉ một cửa sổ thời gian.

**34. Nếu là doanh nghiệp, chọn mô hình nào?**
Không có "tốt nhất tuyệt đối": phụ thuộc mục tiêu và chi phí. Cần dự báo từng chuỗi, tiết kiệm tính toán: Croston SBA (10 giây, sai số gần ETS) hoặc ETS nếu có tài nguyên. Cần nhất quán ở cấp tổng hợp: LightGBM/stacking. Cân bằng: stacking (nhưng cần đủ ba mô hình). Chưa đo chi phí tồn kho nên không kết luận lợi ích kinh doanh.

**35. Đóng góp của từng thành viên?** (theo tài liệu bàn giao; kiểm tra lại với nhóm)
Tuấn Hải: Seasonal Naive, điều phối, W10/P07. Tuấn Vũ: ETS, Croston, Croston SBA. Thủy Tiên: LightGBM, tuning Optuna (pilot), bỏ giá/sự kiện (pilot). Thanh Tiền: XGBoost/CatBoost (pilot), tóm tắt bài báo M5; gợi ý WRMSSE và phân tích theo nhóm. Thùy Trang: Linear Regression (pilot), stacking, ghép báo cáo. Xem `report/handoff/*.md`.

---

## 14. Bẫy dễ nói sai (đọc trước khi lên bảo vệ)
- ❌ "Stacking tốt nhất." → ✅ "Stacking tốt nhất **trên Test**, nhưng chênh lệch với ETS nhỏ và đổi chiều trên Validation."
- ❌ "LightGBM thua ETS." → ✅ "Thua ở cấp **đáy** (macro-RMSSE) nhưng **thắng** ETS ở WRMSSE (0,722 vs 0,760 Test; 0,591 vs 0,724 Validation)."
- ❌ "Croston SBA kém." → ✅ "Tốt ở cấp đáy và rẻ nhất, **kém nhất ở cấp tổng hợp**."
- ❌ "ETS chạy 35.121 giây." → ✅ "Tổng CPU 35.121 giây, thực ≈ 8.794 giây (≈ 2,5 giờ) nhờ 4 tiến trình."
- ❌ "Nhanh hơn 3.350 lần." → ✅ "Croston SBA ít hơn ETS khoảng 3.350 lần **tổng CPU** (35.121 / 10,5), không phải thời gian thực."
- ❌ "Giá/lịch không quan trọng với LightGBM." → ✅ "Importance (gain) của giá nhỏ; không phải kết luận nhân quả."
- ❌ "Test hoàn toàn khách quan." → ✅ "Khóa cấu hình và chạy một lần; nhưng nhãn Test của 1.200 chuỗi pilot đã được xem trước."
- ❌ "So với bảng xếp hạng M5, nhóm đứng hạng…" → ✅ Cùng cửa sổ chấm công khai và thước đo, nhưng không phải bài nộp thi; không xếp hạng.
- ❌ "KTC bootstrap chứng minh stacking hơn ETS." → ✅ KTC chỉ phản ánh biến thiên giữa chuỗi trong một cửa sổ.
- ❌ "R² âm nghĩa là mô hình hỏng." → ✅ Nghĩa là kém hơn lấy trung bình nhãn; phổ biến với chuỗi thưa 28 ngày.
- ❌ Gọi `d_1914` là "ngày đầu tháng 5". → ✅ 25/04/2016.
- Nhớ: ngành hàng = `cat_id` (3), nhóm hàng = `dept_id` (7).
- Nhớ: **RMSSE macro** (cấp đáy, 30.490) khác **WRMSSE** (12 cấp, 42.840, trọng số doanh thu).

---

## 15. Bảng số cần thuộc

**Dữ liệu:** 30.490 chuỗi; 42.840 chuỗi / 12 cấp; 3.049 sản phẩm; 10 cửa hàng; 3 bang; 3 ngành hàng; 7 nhóm hàng; 1.941 ngày; d_1–1885 Train, d_1886–1913 Val, d_1914–1941 Test; horizon 28.
**Gián đoạn:** 68,4% ngày bán 0; 39,1% chuỗi > 80% ngày 0 (11.907); 21,4% (6.530) ≤ 0,5; 39,5% (12.053) trong 0,5–0,8; 23.511 chuỗi có 0 đầu (trung vị 452 ngày).
**Pilot:** 1.200 chuỗi (10 × 3 × 40, seed 42), 11 mô hình, 3,9% số chuỗi, nhãn Test đã xem.
**Đặc trưng:** lag 28/35/42; rolling 7/28 sau shift 28; 21 đặc trưng; bỏ `item_id`; ≈ 55,8 triệu dòng (Val) / 56,65 triệu (Test).
**LightGBM:** 100 cây, lr 0,1, 31 lá, seed 42, 4 luồng, không tuning.
**Stacking:** Ridge; OOF mốc 1829 & 1857; 1.707.440 dòng; alpha 0,1; β: ETS 0,543, LightGBM 0,371, SBA 0,138, b₀ −0,005.
**Croston SBA:** α = 0,1; hệ số 0,95.
**Bootstrap:** 2.000 lần, seed 42.
**WRMSSE kiểm tra:** hoàn hảo 0; Seasonal Naive 0,870; toàn 0 → 5,447.

| | Test RMSSE | Val RMSSE | Test WRMSSE | Val WRMSSE | Test MAE |
|---|---:|---:|---:|---:|---:|
| Seasonal Naive | 0,997 | 0,976 | 0,870 | 0,923 | 1,244 |
| ETS | 0,758 | 0,737 | 0,760 | 0,724 | 1,040 |
| Croston SBA | 0,769 | 0,759 | 1,095 | 1,121 | 1,075 |
| LightGBM | 0,775 | 0,757 | 0,722 | 0,591 | 1,079 |
| Stacking | 0,757 | 0,738 | 0,668 | 0,715 | 1,050 |

**Cải thiện so Seasonal Naive (RMSSE, Test):** ETS −23,9%; SBA −22,9%; LightGBM −22,3%; Stacking −24,1%.
**Thời gian Test:** SN 0,7 s; SBA 10,5 s; LightGBM 228 s; ETS 35.121 s CPU (≈ 8.794 s thực); Stacking 12,1 s (+ cơ sở ≈ 35.372 s).
**Dự báo âm (ETS):** 16.474 / 853.720 (1,9%). **R² không xác định:** 812 (Test), 1.587 (Val).
**ETS cấu trúc:** A,N,A 52,4%; A,N,N 35,8%; A,Ad,A 7,6%; A,Ad,N 3,6%. α trung bình 0,066.
**LightGBM gain:** rolling_mean_28 68,5%; rolling_mean_7 18,2%; lag_28 3,9%; sell_price 1,9%.
**M5:** đội thắng WRMSSE 0,520; ES_bu 0,671; cải thiện 22,4%; cấp đáy 3,4%; 415/5.507 đội (7,5%) vượt ES_bu.
**Môi trường:** Kaggle CPU 4 lõi, ≈ 30 GB RAM, Python 3.13; commit `c8ba5c8`; hash manifest `56a6267c…`.

---

## 16. Việc còn tồn đọng trước khi nộp (theo `report/handoff/VIEC_CUA_NHOM.md`, cập nhật 18/11/2026)
- **Điền họ tên đầy đủ và MSSV** vào báo cáo (đang ghi "chờ đối chiếu danh sách chính thức"); trên Overleaf còn các chỗ điền tên/MSSV/email.
- **W10: đọc chéo báo cáo** và đối chiếu số liệu với `results/full/tables/` (hạn kế hoạch 23/11); sửa bằng PR, bảng sinh lại bằng `scripts/build_report.py`.
- **W11: nộp báo cáo** (hạn kế hoạch 24/11) chỉ sau W10.
- **P07/P08: tập dượt** có bấm giờ (23/11) và một người đóng vai thầy hỏi (24/11), dùng mục 13 làm bộ câu hỏi.
- **Đọc lại toàn văn các nguồn tham khảo** chưa đọc lại: [2] Hyndman & Khandakar 2008; [3] Croston 1972; [4] Syntetos & Boylan 2005; [5] Friedman 2001; [7] Chen & Guestrin 2016; [8] Prokhorenkova 2018; và nên đọc [1], [6], [9] (hiện chỉ đối chiếu qua tóm tắt/abstract).
- **Thời lượng seminar** chưa xác minh: hỏi giảng viên để cắt slide cho vừa giờ.
- **Slide** mới kiểm cấu trúc (17 slide, đủ ghi chú), **chưa xem hình ảnh**: từng người xem phần của mình.
- **Phân công lại** vì XGBoost/CatBoost/Linear Regression đã bị bỏ khỏi nghiên cứu chính (gợi ý: Thanh Tiền WRMSSE và phân tích theo nhóm; Thùy Trang stacking). Chỉ là gợi ý để nhóm chỉnh.

---

## 17. Tài liệu tham khảo (trong báo cáo)
1. Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2022). M5 accuracy competition: Results, findings, and conclusions. *International Journal of Forecasting*, 38(4), 1346–1364. doi:10.1016/j.ijforecast.2021.11.013
2. Hyndman, R. J., & Khandakar, Y. (2008). Automatic time series forecasting: the forecast package for R. *J. Statistical Software*, 27(3).
3. Croston, J. D. (1972). Forecasting and stock control for intermittent demands. *Operational Research Quarterly*, 23(3), 289–303.
4. Syntetos, A. A., & Boylan, J. E. (2005). The accuracy of intermittent demand estimates. *Int. J. Forecasting*, 21(2), 303–314.
5. Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. *Annals of Statistics*, 29(5), 1189–1232.
6. Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *NeurIPS 30*.
7. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD 2016*.
8. Prokhorenkova, L., et al. (2018). CatBoost: unbiased boosting with categorical features. *NeurIPS 31*.
9. Bandara, K., Hewamalage, H., Godahewa, R., & Gamakumara, P. (2022). A fast and scalable ensemble of global models with long memory and data partitioning for the M5 forecasting competition. *Int. J. Forecasting*, 38(4), 1400–1404. doi:10.1016/j.ijforecast.2021.11.004

Nguồn mã và dữ liệu: [Cuộc thi M5 Forecasting – Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy); [M5 methods (mã đội thắng)](https://github.com/Mcompetitions/M5-methods).

---

## Phụ lục A. Từ điển thuật ngữ nhanh

| Thuật ngữ | Nghĩa |
|---|---|
| Chuỗi (series) | Doanh số theo ngày của một sản phẩm tại một cửa hàng |
| Intermittent demand | Nhu cầu gián đoạn, nhiều ngày bán 0 |
| Horizon | Số ngày dự báo (28) |
| Origin | Ngày cuối cùng có dữ liệu dùng để dự báo (T) |
| Refit | Huấn luyện lại trên dữ liệu mới hơn trước khi chạy Test |
| Leakage (rò rỉ) | Mô hình vô tình dùng thông tin từ tương lai |
| Local / Global | Một mô hình mỗi chuỗi / một mô hình cho mọi chuỗi |
| Macro / Pooled | Trung bình các chuỗi / gộp mọi điểm rồi tính |
| OOF (out-of-fold) | Dự báo "ngoài mẫu" dùng huấn luyện bộ gộp |
| Backtest | Dự báo thử trong quá khứ từ một mốc giả định |
| Stacking | Kết hợp dự báo bằng một mô hình học trọng số (meta-learner) |
| Ridge | Hồi quy tuyến tính có phạt L2 |
| AICc | Tiêu chí chọn mô hình có hiệu chỉnh mẫu nhỏ |
| α, β, γ, φ (ETS) | Hệ số làm mượt mức, xu hướng, mùa vụ; hệ số tắt dần |
| z, p (Croston) | Lượng bán khi có bán; khoảng cách giữa hai lần bán |
| SBA | Syntetos–Boylan Approximation, hệ số (1 − α/2) khử chệch |
| GBDT | Cây quyết định tăng cường gradient |
| Gain (importance) | Mức giảm hàm mất mát nhờ đặc trưng trong các lần chia cây |
| Bootstrap | Lấy mẫu lại có hoàn lại để ước lượng độ bất định |
| KTC 95% | Khoảng tin cậy 95% |
| SNAP | Chương trình tem thực phẩm theo bang |
| `run_id` | Mã nhận diện lần chạy (split, thời điểm UTC, SHA mã) |
| `config-freeze` | Git tag khóa cấu hình trước khi chạy Test |
| Public leaderboard (M5) | Bảng xếp hạng công khai, chấm d_1914–d_1941 |
| WRMSSE | RMSSE có trọng số doanh thu, trung bình 12 cấp |

---

## Phụ lục B. Nơi tìm bằng chứng cho từng khẳng định

| Cần chứng minh | Mở file |
|---|---|
| Bảng Test tổng thể | `results/full/tables/test/test_overall.csv` |
| Bảng Validation tổng thể | `results/full/tables/validation/validation_overall.csv` |
| WRMSSE 12 cấp | `results/full/tables/{validation,test}/*_wrmsse.csv`, `scripts/wrmsse.py` |
| KTC bootstrap | `results/full/tables/test/test_bootstrap_ci.csv`, `test_bootstrap_pairs.csv` |
| Theo nhóm nhu cầu, ngành hàng, bang | `test_by_demand_class.csv`, `test_by_cat_id.csv`, `test_by_state.csv` |
| Độ bền phân nhóm | `test_by_demand_class_after_first_sale.csv` |
| Thời gian | `test_runtime_tradeoff.csv`, `timings.json` |
| Cấu trúc ETS | `results/full/runs/val-…/diag_ets.csv` |
| Importance LightGBM | `diag_lightgbm_importance.csv` |
| Hệ số stacking | `diag_stacking_coefficients.csv` |
| Phiên bản/môi trường | `environment.json`, `data_sha256.json` |
| Cấu hình đã khóa | `configs/freeze.yaml` (khối `full_data_v3`) |
| Chống rò rỉ | `tests/test_no_leakage.py`, `src/m5/features.py` |
| Định nghĩa độ đo | `src/m5/metrics.py`, `src/m5/evaluate.py`, `scripts/wrmsse.py` |
| Kết quả pilot | `results/runs/*/metrics_summary.csv`, Bảng P1 |
| Tóm tắt bài báo M5 | `paper/summary.md` |
| Bản thảo báo cáo | `report/draft_report.md` |
| Giải thích từng mô hình | `report/handoff/*.md` (một số số liệu còn thuộc pilot 7 mô hình) |
