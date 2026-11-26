# ÔN TẬP ĐỒ ÁN M5 TỪ SỐ 0 (bản dễ hiểu)

> **Dành cho ai?** Người chưa biết gì về dự báo, học máy hay thống kê, nhưng phải hiểu đồ án để báo cáo và trả lời câu hỏi.
>
> **Cách dùng.** File này giải thích bằng ví dụ đời thường và số nhỏ, tự tính được. Mỗi thuật ngữ khó đều có
> "**Nói đơn giản**" (cách hiểu) và "**Cách nhớ**". Khi cần số liệu chi tiết, tra file đầy đủ
> [`KIEN_THUC_DO_AN.md`](KIEN_THUC_DO_AN.md) (cùng thư mục). File này **không thay** file đó, mà là bước đi vào dễ hơn.
>
> **Thứ tự đọc.** Phần 1 → 2 → 3 → 4 (nền tảng) → 5 → 6 → 7 (đồ án) → 8 (tự kiểm tra). Đọc đến đâu, tự trả lời
> các ô "Dừng lại, tự hỏi" đến đó.

---

# PHẦN 1. BỨC TRANH LỚN TRONG 5 PHÚT

## 1.1. Đồ án này làm gì? (kể bằng chuyện đời thường)

Hãy tưởng tượng bạn là quản lý một siêu thị. Mỗi sáng bạn phải quyết định **nhập bao nhiêu hàng** cho 4 tuần tới:
- Nhập **ít** quá → khách đến mà hết hàng → mất doanh thu.
- Nhập **nhiều** quá → hàng ế, tồn kho → tốn tiền, hỏng hàng.

Muốn nhập đúng, bạn cần **dự đoán** mỗi ngày mỗi món bán được bao nhiêu. Đó là **bài toán dự báo**.

Walmart (chuỗi siêu thị Mỹ) có **30.490** "cặp" như vậy (mỗi cặp = một món hàng ở một cửa hàng). Cuộc thi **M5** trên Kaggle
đưa dữ liệu thật của Walmart và hỏi: *"Ai dự đoán chính xác nhất doanh số 28 ngày tới?"*

**Đồ án của nhóm** không đi thi lấy giải. Nhóm làm một việc khác: **thử 5 cách dự đoán khác nhau trên cùng một bộ dữ
liệu, cùng một đề bài, cùng một thước chấm điểm**, rồi trả lời: *cách nào chính xác hơn? cách nào nhanh/rẻ hơn? vì sao?*

Giống như cho 5 bạn cùng làm một đề thi rồi so điểm và so thời gian làm bài.

## 1.2. Năm "thí sinh" (nhớ bằng một câu)

| # | Tên | Một câu dễ nhớ |
|---|---|---|
| 1 | **Seasonal Naive** | "Tuần này giống hệt tuần trước." (cách ngây thơ nhất, làm **mốc so sánh**) |
| 2 | **ETS** | "Lấy trung bình nhưng nhớ nhiều về gần đây, quên dần chuyện xa; có tính thêm chu kỳ trong tuần." |
| 3 | **Croston SBA** | "Món này **bao lâu bán một lần** và **mỗi lần bán mấy cái**?" (chuyên cho hàng bán lâu lâu mới có người mua) |
| 4 | **LightGBM** | "Một **hội đồng các cây hỏi–đáp** học từ dữ liệu của **cả 30.490 món cùng lúc**." (học máy) |
| 5 | **Stacking lai** | "Họp ba chuyên gia (ETS, Croston SBA, LightGBM) và **tin mỗi người theo một mức** đã học được." |

## 1.3. Kết quả trong một đoạn

- **Cả bốn cách "thông minh" đều giỏi hơn cách ngây thơ khoảng 22–24%.**
- Với hàng **bán rất thưa**, mấy cách "gọn nhẹ" (ETS, Croston SBA) **thắng** học máy LightGBM.
- **Chi phí** chênh nhau khủng khiếp: Croston SBA chạy ~10 giây, ETS chạy ~2,5 giờ, mà độ chính xác chỉ hơn chút xíu.
- Khi **cộng dồn lên cấp lớn** (cả cửa hàng, cả bang, cả công ty), thứ hạng **đảo lộn**: LightGBM tốt, Croston SBA **tệ nhất**.
- Cách gộp (Stacking) đứng nhất trên tập thi cuối, nhưng hơn ETS **rất ít** và **không chắc bền**.

> **Dừng lại, tự hỏi:** Nếu ai đó hỏi "đồ án làm gì?", bạn nói được trong 30 giây không? Thử nói to:
> *"Nhóm so sánh 5 cách dự báo doanh số 28 ngày của toàn bộ 30.490 món–cửa hàng Walmart, xem cách nào chính xác hơn, nhanh hơn
> và vì sao."*

---

# PHẦN 2. KIẾN THỨC NỀN (không biết gì cũng đọc được)

## 2.1. Chuỗi thời gian và dự báo

**Chuỗi thời gian** = một dãy số ghi theo thời gian. Ví dụ doanh số hàng ngày của "kem đánh răng X ở cửa hàng A":

```
Thứ Hai  Ba  Tư  Năm  Sáu  Bảy  CN   Hai  Ba ...
   2     0   1    3    0    1   4     2    0 ...
```

**Dự báo** = đoán các số **chưa xảy ra** dựa trên các số **đã xảy ra**.

| Thuật ngữ | Nói đơn giản | Trong đồ án |
|---|---|---|
| **Horizon** (tầm dự báo) | Dự đoán xa bao nhiêu ngày | 28 ngày (4 tuần) |
| **Origin** (mốc dự báo) | "Hôm nay" — ngày cuối cùng mình đã biết số liệu | Validation: ngày d_1885; Test: d_1913 |
| **Dự báo điểm** | Mỗi ngày đoán **một con số** | Đồ án làm cái này |

*Cách nhớ:* origin là **"cột mốc đứng nhìn về tương lai"**; horizon là **"nhìn xa bao nhiêu"**.

## 2.2. Chuỗi thời gian có những "tính cách" gì?

1. **Mùa vụ (chu kỳ):** cuối tuần bán nhiều hơn giữa tuần, lặp lại mỗi 7 ngày. (Dữ liệu M5: thứ Bảy, Chủ nhật cao hơn khoảng 20%.)
2. **Xu hướng:** nhìn dài hạn thì tăng hay giảm.
3. **Nhiễu:** dao động ngẫu nhiên không giải thích được (hôm nay bất ngờ có đoàn khách mua).
4. **Nhu cầu gián đoạn (intermittent demand)** ← *đặc điểm quan trọng nhất của đồ án*.

### Nhu cầu gián đoạn là gì?
> **Nói đơn giản:** món hàng này **hầu như ngày nào cũng không ai mua**, thỉnh thoảng mới có người mua 1–2 cái.

Ví dụ một chuỗi: `0,0,0,2,0,0,0,0,1,0,0,3,0,0,...`

Số liệu thật: **68,4%** số ngày trong dữ liệu có doanh số **bằng 0**. **39,1%** số chuỗi có hơn 80% ngày bằng 0.

**Vì sao khó?** Nếu đoán "ngày mai bằng hôm nay" thì sai liên tục (hôm nay 0, ngày mai bỗng có người mua). Nếu lấy trung bình thì ra số lẻ như 0,3 cái — đúng "trung bình" nhưng chưa bao giờ đúng một ngày cụ thể. Nên cần phương pháp riêng (Croston).

*Cách nhớ:* hình dung tiệm **bán đồ sưu tầm hiếm**: cả buổi vắng khách, đột nhiên một người mua vài món.

## 2.3. Đo sai số: "dự đoán trật bao nhiêu?"

Muốn biết cách nào giỏi hơn, phải **đo độ trật**. Làm quen bằng ví dụ 4 ngày:

| Ngày | Thực tế | Dự đoán | Sai lệch (thực − dự) | Trị tuyệt đối | Bình phương |
|---|---:|---:|---:|---:|---:|
| 1 | 0 | 1 | −1 | 1 | 1 |
| 2 | 2 | 1 | +1 | 1 | 1 |
| 3 | 1 | 1 | 0 | 0 | 0 |
| 4 | 3 | 1 | +2 | 2 | 4 |

### MAE — "trung bình trật bao nhiêu cái"
MAE = trung bình các trị tuyệt đối = (1+1+0+2)/4 = **1,0**.
> **Nói đơn giản:** mỗi ngày trật trung bình **1 sản phẩm**. Dễ hiểu nhất, cùng đơn vị với doanh số.

### RMSE — "giống MAE nhưng ghét lỗi to"
RMSE = căn bậc hai của trung bình các bình phương = √((1+1+0+4)/4) = √1,5 ≈ **1,22**.
> **Nói đơn giản:** bình phương làm lỗi **lớn bị phạt nặng hơn nhiều** (lỗi 2 bị tính 4, lỗi 1 chỉ tính 1). Nên RMSE
> lớn hơn MAE khi có vài lần trật to.
> *Cách nhớ:* MAE = "thầy giáo dễ tính", RMSE = "thầy giáo ghét sai sót lớn".

### Vấn đề: so sánh công bằng giữa các món
Món A bán 100 cái/ngày, trật 5 cái là **rất giỏi**. Món B bán 1 cái/ngày, trật 5 cái là **rất tệ**. Nếu cộng MAE thẳng
sẽ không công bằng. Cần một thước **chuẩn hóa theo từng món** → đó là **RMSSE**.

### RMSSE — thước chính của đồ án
> **Nói đơn giản:** so sai số của mình với sai số của **"người chỉ biết nói mai bằng hôm nay"** trên lịch sử của chính món đó.
> - RMSSE **= 1**: ngang người ngây thơ đó.
> - RMSSE **< 1**: **giỏi hơn** người ngây thơ. Càng nhỏ càng tốt.
> - RMSSE **> 1**: dở hơn người ngây thơ.

Tính thế nào (ví dụ nhỏ):
1. Lịch sử huấn luyện của món: `0, 0, 3, 0, 2, 1`. **Bỏ các số 0 ở đầu** (lúc đó món chưa bán) → `3, 0, 2, 1`.
2. "Người ngây thơ" mỗi ngày sai bao nhiêu? Hiệu giữa hai ngày liền: `0−3=−3`, `2−0=2`, `1−2=−1`. Bình phương: `9, 4, 1`. Trung bình = **4,667**.
3. Giả sử sai số bình phương trung bình của mình trên 28 ngày dự báo là **1,2**.
4. RMSSE = √(1,2 / 4,667) ≈ **0,507**. Nhỏ hơn 1 → giỏi hơn người ngây thơ.

Sau đó lấy **trung bình RMSSE của cả 30.490 món** → gọi là **macro-RMSSE** ("macro" = trung bình các món, món nào cũng một phiếu).

**Vì sao không dùng MAPE (sai số %)?** Vì MAPE chia cho doanh số thật; 68% số ngày doanh số = 0 → **chia cho 0**, vô nghĩa.

### R² — ít quan trọng, nhưng sẽ bị hỏi
> **Nói đơn giản:** "mô hình tốt hơn việc đoán **luôn bằng trung bình** của 28 ngày thật bao nhiêu?"
> - R² = 1: hoàn hảo; R² = 0: ngang việc đoán bằng trung bình; **R² âm: kém hơn việc đoán bằng trung bình**.

Trong đồ án R² **đều âm** (ví dụ −0,18 đến −1,37). **Không phải lỗi**: món bán thưa, trong 28 ngày hầu như toàn 0, trung bình của chính 28 ngày thật là mốc quá "dễ" để thắng.
Có **812 món** (Test) mà 28 ngày thật toàn bằng nhau (toàn 0) → mẫu số bằng 0 → R² **không tính được**; nhóm ghi "không xác định" và **không** tự bịa số thay vào.

### Macro và Pooled (hay nhầm)
- **Macro:** tính điểm từng món rồi **lấy trung bình** → món bán ít hay nhiều đều có "một phiếu bầu".
- **Pooled:** dồn **mọi ngày của mọi món** vào một đống rồi tính một lần → món bán nhiều **áp đảo**.

> **Dừng lại, tự hỏi:** Vì sao dùng RMSSE thay vì MAE để xếp hạng? *(Đáp: RMSSE chuẩn hóa theo từng món nên công bằng giữa món bán nhiều và bán ít; MAE thì không.)*

## 2.4. Chia dữ liệu: Train / Validation / Test (ví dụ ôn thi)

Tưởng tượng bạn học thi:

| Giai đoạn | Ví dụ ôn thi | Trong đồ án |
|---|---|---|
| **Train** (huấn luyện) | Học bài, làm bài tập | Ngày d_1 → d_1885: mô hình "học" từ đây |
| **Validation** (kiểm tra) | **Thi thử**: xem mình đến đâu, chỉnh cách học | d_1886 → d_1913 (28 ngày) |
| **Test** (thi thật) | **Kỳ thi thật**, chỉ thi **một lần** | d_1914 → d_1941 (28 ngày) |

Quy tắc vàng: **không được xem đề thi thật khi còn đang chỉnh cách học.** Nếu xem, điểm thi sẽ "đẹp giả".

- **config-freeze** = "**niêm phong cách làm bài trước khi mở đáp án**". Nhóm khóa cấu hình bằng một "tem" (git tag) rồi mới chạy Test, và **chạy đúng một lần**. Mã còn **chặn** không cho chạy Test nếu chưa niêm phong.
- **Refit trên d_1–d_1913 cho Test:** thi thật thì được học thêm cả giai đoạn thi thử (cập nhật thông tin mới nhất như doanh nghiệp thực tế).

**Chia theo thời gian, không xáo ngẫu nhiên.** Dự báo là nhìn về tương lai; nếu xáo trộn, mô hình sẽ "thấy" tương lai khi học.

### Rò rỉ dữ liệu (data leakage) — "lộ đề thi"
> **Nói đơn giản:** mô hình **vô tình biết** thông tin từ tương lai khi đang dự đoán tương lai. Điểm sẽ đẹp giả, ngoài đời chạy tệ.

Cách nhóm phòng:
1. Nhãn (doanh số thật) từ ngày dự báo trở đi bị **che** (đặt rỗng) trước khi tạo đặc trưng.
2. Mọi đặc trưng từ doanh số đều **lùi ít nhất 28 ngày** (giải thích dưới đây).
3. Có **bài kiểm thử tự động**: cố tình làm hỏng nhãn tương lai rồi kiểm tra kết quả **không đổi**.

**Vì sao phải lùi ít nhất 28 ngày?** Mô hình đoán **một lần** cả 28 ngày từ cùng một "hôm nay". Ngày xa nhất (ngày thứ 28) chỉ có thông tin đến "hôm nay". Nếu dùng "doanh số hôm qua" cho ngày thứ 28, thì "hôm qua" của ngày đó nằm **trong tương lai** của mình → lộ đề. Lùi 28 ngày thì luôn an toàn.

## 2.5. Mô hình, tham số, quá khớp

- **Mô hình:** một "công thức" biến dữ liệu quá khứ thành dự đoán.
- **Huấn luyện (fit):** cho công thức "học" — tự chỉnh các con số bên trong sao cho khớp dữ liệu quá khứ.
- **Tham số / siêu tham số:** các "núm vặn". Siêu tham số do người đặt (ví dụ số cây), tham số do máy tự học.
- **Quá khớp (overfitting):** học vẹt dữ liệu cũ đến mức gặp dữ liệu mới thì sai. *Giống học thuộc đáp án đề cũ mà không hiểu bài.* Chia Train/Test là để **phát hiện** chuyện này.

## 2.6. Local và Global — khái niệm rất hay bị hỏi

| | **Local (cục bộ)** | **Global (toàn cục)** |
|---|---|---|
| Ý tưởng | **Mỗi món một mô hình riêng**, chỉ nhìn lịch sử món đó | **Một mô hình chung** học từ **mọi món** |
| Ví dụ đời thường | Mỗi học sinh có **một gia sư riêng**, chỉ biết học sinh đó | **Một thầy dạy cả lớp**, thấy bài làm của cả lớp |
| Trong đồ án | Seasonal Naive, ETS, Croston SBA (30.490 mô hình) | LightGBM (1 mô hình cho cả 30.490 món) |
| Thông tin | Chỉ doanh số của món đó | Thêm giá, lịch, ngày lễ, SNAP, nhóm hàng… |

**Hệ quả quan trọng:** khi LightGBM thắng hay thua ETS, ta không biết là do *thuật toán* hay do *được biết thêm nhiều thông tin*. Đây là một giới hạn phải nói thẳng.

---

# PHẦN 3. DỮ LIỆU CỦA ĐỒ ÁN

## 3.1. Dữ liệu là gì?

Năm file từ Kaggle (không lưu trên Git vì quá nặng):

| File | Chứa gì | Ví dụ đời thường |
|---|---|---|
| `sales_train_evaluation.csv` | Doanh số mỗi món × mỗi cửa hàng × mỗi ngày (1.941 ngày) | Sổ ghi bán hàng |
| `calendar.csv` | Ngày, thứ, tháng, sự kiện/lễ, cờ SNAP | Lịch treo tường có đánh dấu ngày lễ |
| `sell_prices.csv` | Giá từng món theo tuần | Bảng giá |
| `sales_train_validation.csv` | Như file đầu nhưng chỉ đến d_1913 | Bản cũ của sổ bán hàng |
| `sample_submission.csv` | Mẫu định dạng nộp bài thi | Phiếu trả lời trống |

Nhóm dùng `sales_train_evaluation.csv` (đủ ngày nhất), không ghép với `…validation…` vì chúng **trùng** nhau ở phần đầu.

## 3.2. Các "tầng" của dữ liệu (phân cấp)

```
Walmart (1 công ty)
 ├─ 3 bang: California (CA), Texas (TX), Wisconsin (WI)
 │    └─ 10 cửa hàng (CA có 4, TX có 3, WI có 3)
 ├─ 3 ngành hàng:  FOODS (thực phẩm), HOBBIES (sở thích), HOUSEHOLD (gia dụng)
 │    └─ 7 nhóm hàng (FOODS_1,2,3; HOBBIES_1,2; HOUSEHOLD_1,2)
 └─ 3.049 sản phẩm
```

- **Một chuỗi** = một sản phẩm tại một cửa hàng. 3.049 × 10 = **30.490 chuỗi** (gọi là **cấp đáy**).
- Mã chuỗi ví dụ: `HOBBIES_1_001_CA_1` = sản phẩm 001 thuộc nhóm HOBBIES_1, ở cửa hàng CA_1.
- Có thể **cộng dồn** để được chuỗi cấp lớn hơn: tổng cả bang, tổng cả cửa hàng, tổng cả công ty… M5 định nghĩa **12 cấp** với tổng **42.840 chuỗi**.

*Cách nhớ:* ngành hàng = **3** (FOODS/HOBBIES/HOUSEHOLD, chữ "cat" = category); nhóm hàng = **7** (chữ "dept" = department).

## 3.3. Các khái niệm phụ trong dữ liệu

- **SNAP:** chương trình tem phiếu thực phẩm của chính phủ Mỹ. Một số ngày trong bang, người dùng tem được mua đồ → doanh số thực phẩm đổi. Dữ liệu có cờ 0/1.
- **Giá thiếu:** sản phẩm chưa lên kệ nên chưa có giá. Nhóm **giữ trống** + thêm cờ "giá bị thiếu", **không điền bừa**.
- **Giả định giá tương lai:** khi dự báo, nhóm dùng **giá của đúng ngày cần dự báo** (dữ liệu M5 có sẵn). Ngoài đời chỉ làm được nếu **giá được lên kế hoạch trước**. Phải nêu rõ đây là **giả định**.
- **Chuỗi bắt đầu muộn:** 23.511 / 30.490 chuỗi có một đoạn toàn 0 ở đầu (sản phẩm chưa bán) — trung vị 452 ngày. Vì vậy khi chuẩn hóa RMSSE nhóm **bỏ phần 0 ở đầu**.

## 3.4. Ba nhóm nhu cầu (để trả lời "mô hình nào hợp loại hàng nào")

Với mỗi chuỗi, tính **tỷ lệ ngày bán bằng 0** trên Train rồi chia:

| Nhóm | Ý nghĩa dễ hiểu | Số chuỗi |
|---|---|---:|
| `low_zero` (≤ 50% ngày là 0) | Hàng **bán đều**, ít khi trống | 6.530 (21,4%) |
| `mid_zero` (50–80%) | Hàng **bán vừa vừa** | 12.053 (39,5%) |
| `high_zero` (> 80%) | Hàng **bán rất thưa** | 11.907 (39,1%) |

---

# PHẦN 4. QUY TRÌNH ĐỒ ÁN: TỪ ĐẦU ĐẾN CUỐI (kể theo thứ tự)

**Bước 1 — Thử nhỏ (Pilot).** Nhóm chọn **1.200 chuỗi** (đại diện đủ cửa hàng và ngành hàng, 10 cửa hàng × 3 ngành × 40 chuỗi, cố định "hạt giống" 42 để ai chạy cũng ra cùng mẫu) và chạy **11 mô hình** thử.
*Ví dụ đời thường:* thử nấu một mẻ nhỏ để chọn công thức trước khi nấu cả nồi lớn.

**Bước 2 — Chọn 5 mô hình.** Dựa trên pilot, giữ mỗi **cách tiếp cận** một đại diện:
- Giữ: Seasonal Naive (mốc), ETS, Croston SBA, LightGBM, Stacking.
- Bỏ vì *trùng vai trò mà không tốt hơn*: XGBoost và CatBoost (ngang LightGBM nhưng chậm; CatBoost ước tính cần ~14 giờ CPU trên toàn bộ dữ liệu), Croston cổ điển (kém bản SBA), Linear Regression (kém nhất nhóm học máy).
- Bỏ vì *không giữ được lợi thế trên Test*: LightGBM tinh chỉnh bằng Optuna (hơn ở Validation nhưng không hơn ở Test).

**Bước 3 — Khóa cấu hình** (config-freeze, "niêm phong").

**Bước 4 — Chạy toàn bộ 30.490 chuỗi** trên Kaggle (máy chủ miễn phí, 4 lõi CPU, ~30 GB RAM): 4 notebook chạy song song: Validation, Test, và hai lần "tập dượt" cho Stacking.

**Bước 5 — Chấm điểm** bằng MAE, RMSE, RMSSE, R², WRMSSE, thời gian chạy; thêm khoảng tin cậy bootstrap; phân tích theo nhóm hàng, ngành, bang, thứ trong tuần, ngày dự báo.

**Bước 6 — Viết báo cáo, slide, hỏi đáp** (báo cáo cuối viết bằng LaTeX trên Overleaf).

### Vì sao chạy toàn bộ dữ liệu thay vì mẫu 1.200?
1. Mẫu pilot **lệch** về hàng thưa (43,3% chuỗi > 80% ngày 0 so với 39,1% thật) nên không đại diện hoàn hảo.
2. Muốn tính **WRMSSE 12 cấp** thì cần **đủ cả 30.490 chuỗi đáy** để cộng dồn lên.
3. Kết quả: thứ tự bốn mô hình **giống pilot** → chứng tỏ chọn mô hình từ mẫu nhỏ là hợp lý.

### Một điều phải trung thực thừa nhận
Nhãn Test của 1.200 chuỗi pilot (3,9% số chuỗi) đã được **xem** từ trước khi chọn 5 mô hình. Nên Test toàn bộ **không còn hoàn toàn "mù"**. Nhóm ghi rõ điều này trong phần giới hạn.
*Đây là kiểu câu nên **tự nói trước** — nói trước thể hiện trung thực, bị hỏi mới nói thì trông như che giấu.*

---

# PHẦN 5. NĂM MÔ HÌNH, GIẢI THÍCH TỪNG CÁI

## 5.1. Seasonal Naive — "tuần này như tuần trước"

**Cách làm:** lấy 7 ngày cuối cùng đã biết, **lặp lại** 4 lần cho 28 ngày.

Ví dụ: tuần cuối là `2, 0, 1, 3, 0, 1, 4` → dự báo 28 ngày = `2,0,1,3,0,1,4 | 2,0,1,3,0,1,4 | …` (4 lần).

- Không có gì để học, chạy dưới 1 giây.
- **Vai trò:** làm **mốc so sánh**. Mô hình nào không thắng nổi nó thì vô dụng.
- **Điểm yếu:** sao chép luôn cả **nhiễu** và các **số 0 ngẫu nhiên** của tuần cuối.
- **Vì sao RMSSE của nó ≈ 1 (0,997)?** Vì thước RMSSE cũng so với "người ngây thơ"; mà Seasonal Naive **chính là một kiểu ngây thơ** nên điểm xấp xỉ 1. Đừng hiểu nhầm là bất thường.

## 5.2. ETS — "trung bình có trí nhớ phai dần"

**ETS** viết tắt: **E**rror (kiểu sai số), **T**rend (xu hướng), **S**easonal (mùa vụ). Là họ mô hình **làm mượt mũ**.

### Làm mượt mũ đơn giản nhất
Công thức: **mức mới = α × (số hôm nay) + (1 − α) × (mức cũ)**.

Ví dụ: mức cũ = 2,0; hôm nay bán 0; α = 0,1 →
mức mới = 0,1 × 0 + 0,9 × 2,0 = **1,8**.

> **Nói đơn giản:** α là **"mức tin vào số mới nhất"**.
> - α **lớn** (gần 1): hôm nay ra sao tin ngay vậy → nhạy, dễ bị nhiễu kéo đi.
> - α **nhỏ** (gần 0): chủ yếu tin vào quá khứ dài → **mượt, ổn định**.
>
> *Cách nhớ:* α giống **"độ hay thay đổi ý kiến"** của một người. Dữ liệu thưa nhiều nhiễu thì nên **bình tĩnh** (α nhỏ).

Kết quả thực tế: α trung bình chỉ **0,066** (rất bình tĩnh).

### Đọc ký hiệu ETS(A,N,A)
Ba chữ cái = (Sai số, Xu hướng, Mùa vụ). **A** = cộng (additive), **N** = không có, **Ad** = cộng nhưng tắt dần (damped).
- **ETS(A,N,N):** chỉ làm mượt, không xu hướng, không mùa. Dự báo là một đường **phẳng**.
- **ETS(A,N,A):** thêm **chu kỳ 7 ngày**. Dự báo có "răng cưa" theo thứ.
- **ETS(A,Ad,A):** thêm xu hướng đang yếu dần.

Trong đồ án: **52,4%** chuỗi chọn (A,N,A); **35,8%** chọn (A,N,N); phần còn lại ~12% chọn các kiểu khác.

### "Auto" nghĩa là gì?
Máy **tự thử nhiều kiểu ETS cho từng chuỗi** và chọn cái tốt nhất theo tiêu chí **AICc**.
> **AICc nói đơn giản:** một điểm "chấm" cân bằng giữa **khớp dữ liệu tốt** và **không quá phức tạp** (phạt mô hình rườm rà). Điểm thấp = tốt.
> *Giống chọn món ăn: vừa ngon vừa không quá cầu kỳ.*

- **Chi phí:** thử nhiều kiểu cho **30.490 chuỗi** → tổng ~35.121 giây CPU; chạy 4 tiến trình song song thì thực tế ~8.794 giây ≈ **2,5 giờ**. Là mô hình **đắt nhất**.
- **Dự báo có thể âm** (do công thức cộng). Doanh số không âm nên nhóm **cắt về 0** (16.474 điểm, 1,9%).

## 5.3. Croston SBA — "bao lâu bán một lần × mỗi lần bán mấy cái"

### Vấn đề của hàng bán thưa
Chuỗi `0,0,0,2,0,0,0,1,0,0,0,0,3`: làm mượt thông thường bị các số 0 "kéo" liên tục.

### Ý tưởng Croston (1972)
Chỉ **cập nhật khi có người mua**, và theo dõi hai thứ riêng:
- **z** = mỗi lần có người mua thì **mua bao nhiêu** (trung bình mượt).
- **p** = **cách bao nhiêu ngày** thì có một lần mua (trung bình mượt).

Dự báo mỗi ngày = **z / p** (số cái mỗi ngày tính trung bình).

Ví dụ: trung bình cứ **4 ngày** có người mua một lần (p = 4), mỗi lần **2 cái** (z = 2) → dự báo **0,5 cái/ngày**.

### "SBA" là gì?
Có người (Syntetos & Boylan, 2005) chứng minh cách tính z/p **hơi cao quá** (chệch lên). Họ đề xuất **nhân thêm hệ số (1 − α/2)**.
Với α = 0,1 hệ số = **0,95**. Ví dụ trên: 0,5 × 0,95 = **0,475**.
> *SBA = Croston + "chỉnh về mức khiêm tốn hơn một chút cho đỡ lố".*

### Điểm mạnh
- Cực **nhanh** (~10 giây cho 30.490 chuỗi).
- Ở hàng bán thưa, độ chính xác **gần ngang ETS**.

### Điểm yếu — rất quan trọng
Dự báo của nó là **một đường phẳng**: ngày nào cũng cùng một số, **không có "cuối tuần bán nhiều hơn"**.
- Ở cấp từng món: chỉ hơi thiệt (nhất là hàng bán đều có chu kỳ tuần rõ).
- Ở cấp **tổng hợp** (cả cửa hàng, cả bang…): đây là thảm họa. Giải thích ở mục 6.3.

## 5.4. LightGBM — "hội đồng các cây hỏi–đáp học từ mọi món"

### Cây quyết định là gì?
Một chuỗi câu hỏi có/không dẫn đến một con số dự đoán. Ví dụ:

```
Doanh số trung bình 28 ngày qua có > 2 không?
 ├─ Có → Hôm nay là cuối tuần không?
 │        ├─ Có → dự đoán 4,0
 │        └─ Không → dự đoán 2,5
 └─ Không → dự đoán 0,3
```

Một cây đơn lẻ **khá dở**. Sức mạnh nằm ở việc **ghép rất nhiều cây**.

### "Tăng cường gradient" (Gradient Boosting) — sửa lỗi dần dần
> **Nói đơn giản:** cây thứ nhất đoán thô. Cây thứ hai **chỉ học phần cây thứ nhất còn trật**. Cây thứ ba học phần cây 1+2 còn trật…
> Cộng tất cả lại, mỗi cây đóng góp một chút.

Ví dụ: điểm thật = 9. Cây 1 đoán 7 → còn thiếu 2. Cây 2 học "thiếu 2" và đoán thêm +1,5 (nhân hệ số bước nhỏ) → tổng 8,5 → còn thiếu 0,5. Cây 3 học "thiếu 0,5"… Dần dần tiệm cận 9.

- Công thức: **F_mới = F_cũ + η × cây_mới** (η = "bước nhỏ" = learning rate, ở đây **0,1**).
- Cấu hình nhóm dùng: **mặc định của thư viện**, tức **100 cây**, mỗi cây tối đa **31 lá**, bước nhỏ 0,1. **Không tinh chỉnh** (không tuning).
- "LightGBM" = phiên bản **nhẹ và nhanh** của kỹ thuật này (Light = nhẹ, GBM = Gradient Boosting Machine).

### Mô hình global — học từ 56 triệu dòng
Mỗi dòng = "một món, một ngày" kèm 21 đặc trưng. Có 30.490 món × ~1.858 ngày ≈ **56,6 triệu dòng**. Một mô hình duy nhất học từ tất cả.

### Các "đặc trưng" (thông tin đưa cho mô hình)
Cách nhớ theo nhóm:
- **Quá khứ của chính món:** doanh số 28/35/42 ngày trước; trung bình & độ biến động của 7 và 28 ngày (đã lùi 28 ngày).
- **Lịch:** thứ, tháng, năm, ngày lễ/sự kiện, SNAP.
- **Giá:** giá tuần, cờ "giá thiếu".
- **Danh tính:** nhóm hàng, ngành hàng, cửa hàng, bang. *(Không dùng mã sản phẩm vì có hơn 3.000 giá trị → dễ học vẹt và nặng máy.)*

### Mô hình "nhìn" cái gì nhiều nhất?
Độ quan trọng (gain) cho thấy **68,5%** dồn vào **trung bình trượt 28 ngày**, **18,2%** vào **trung bình trượt 7 ngày**. Giá chỉ ~1,9%.
> **Nói đơn giản:** LightGBM chủ yếu đang dự báo "**mức bán gần đây điều chỉnh theo lịch**".
> **Lưu ý khi nói:** đây là mức **đóng góp vào dự báo**, **không phải bằng chứng giá không ảnh hưởng doanh số**.

### Điểm mạnh/yếu
- Mạnh: học được mức và chu kỳ chung → **cực tốt ở cấp tổng hợp**.
- Yếu: ở từng chuỗi **thưa**, kém ETS (ETS làm mượt riêng từng chuỗi tốt hơn).
- Chi phí: ~228 giây huấn luyện + ~46 giây dựng bảng đặc trưng.

## 5.5. Stacking lai — "họp ba chuyên gia, tin mỗi người một mức"

**Ý tưởng:** thay vì chọn một mô hình, lấy dự báo của cả ba (ETS, Croston SBA, LightGBM) rồi **trộn có trọng số**:

> dự báo cuối = 0,543 × (ETS) + 0,138 × (Croston SBA) + 0,371 × (LightGBM) − 0,005, rồi cắt về 0 nếu âm.

Ví dụ: ETS đoán 1,0; Croston SBA đoán 0,8; LightGBM đoán 1,2 →
0,5429×1,0 + 0,1377×0,8 + 0,3713×1,2 − 0,005 ≈ **1,094**.

### Trọng số do đâu mà có?
Phải **học** được — bằng một mô hình nhỏ gọi là **Ridge** (hồi quy tuyến tính có "phạt" để các hệ số không phình quá lớn; hữu ích khi hai chuyên gia ý kiến giống nhau như ETS và SBA).

### Làm sao không "lộ đề"?
Đây là câu hỏi rất hay bị hỏi. Cách làm:
1. **Tập dượt trong Train:** chọn hai "hôm nay giả" là d_1829 và d_1857. Ở mỗi mốc, cho ba chuyên gia **học đến mốc đó** rồi dự đoán 28 ngày kế tiếp (đáp án thật còn nằm trong Train, nên ta biết ai đoán đúng ai sai).
   Hai lần tập dượt → 2 × 30.490 × 28 = **1.707.440** cặp (dự đoán, đáp án).
2. Từ các cặp này, Ridge **học "nên tin ai bao nhiêu"**.
3. Bộ trọng số được **khóa** rồi áp cho Validation và Test. **Không dùng đáp án của Validation/Test để tính trọng số.**

*Ví dụ đời thường:* trước kỳ thi, bạn làm hai đề thi thử cũ, thấy mình giỏi toán hơn văn nên phân bổ giờ ôn theo đó; bạn **không** nhìn đề thi thật.

### Điều thú vị
- ETS được tin nhất (**54%**) vì chính xác ở cấp đáy.
- LightGBM được tin 37% vì mang **chu kỳ tuần và mức chung** mà hai mô hình cục bộ thiếu.
- Croston SBA được tin ít nhất (14%) vì ý kiến của nó **trùng ETS** nhiều.

### Hạn chế
Chỉ có **hai** lần tập dượt → chưa biết trọng số có ổn định theo thời gian không. Bộ trộn tối ưu theo sai số ở cấp đáy, **không** tối ưu WRMSSE.

---

# PHẦN 6. WRMSSE VÀ CÂU CHUYỆN "ĐẢO THỨ HẠNG"

## 6.1. WRMSSE bằng ví dụ

M5 không chỉ chấm ở cấp "từng món–từng cửa hàng", mà chấm ở **12 góc nhìn**:

| Cấp | Góc nhìn | Số chuỗi |
|---:|---|---:|
| 1 | Cả công ty | 1 |
| 2 | Từng bang | 3 |
| 3 | Từng cửa hàng | 10 |
| 4 | Từng ngành hàng | 3 |
| 5 | Từng nhóm hàng | 7 |
| 6 | Bang × ngành hàng | 9 |
| 7 | Bang × nhóm hàng | 21 |
| 8 | Cửa hàng × ngành hàng | 30 |
| 9 | Cửa hàng × nhóm hàng | 70 |
| 10 | Từng sản phẩm (gộp mọi cửa hàng) | 3.049 |
| 11 | Sản phẩm × bang | 9.147 |
| 12 | Sản phẩm × cửa hàng (cấp đáy) | 30.490 |

Cách tính, nói bằng lời:
1. **Cộng dự báo** của các món để có dự báo cho từng "góc nhìn" (ví dụ cộng mọi món của cửa hàng CA_1 ra dự báo cho cửa hàng CA_1).
2. Với mỗi chuỗi ở mỗi cấp, tính RMSSE như ở phần 2.3.
3. **Trọng số = doanh thu** (số lượng × giá) của 28 ngày cuối trước "hôm nay" → món bán ra nhiều tiền quan trọng hơn.
4. WRMSSE của một cấp = tổng (trọng số × RMSSE); **điểm cuối = trung bình 12 cấp**.

> **Nói đơn giản:** WRMSSE trả lời "**nhìn từ mọi tầng của công ty, dự báo của bạn có tốt không, ưu tiên món đem lại nhiều tiền**".
> Khác với macro-RMSSE chỉ nhìn **tầng đáy** và coi mọi món bằng nhau.

Kiểm tra cài đặt của nhóm (để chứng minh code đúng): dự báo hoàn hảo → 0; Seasonal Naive trên Test → 0,870; đoán toàn 0 → 5,447.

## 6.2. Hai thước, hai bảng xếp hạng

| (Test) | macro-RMSSE (nhìn tầng đáy) | WRMSSE (nhìn 12 tầng) |
|---|---:|---:|
| Seasonal Naive | 0,997 | 0,870 |
| ETS | 0,758 | 0,760 |
| Croston SBA | 0,769 | **1,095** (tệ nhất) |
| LightGBM | 0,775 (kém nhất trong 4 mô hình "thông minh") | 0,722 |
| **Stacking** | **0,757** | **0,668** |

*(Số nhỏ hơn = tốt hơn.)*

## 6.3. Vì sao Croston SBA tệ ở cấp tổng hợp? (câu hỏi "ăn điểm")

**Ẩn dụ:** 100 người cùng hát nhưng mỗi người chỉ biết hát **một nốt đều đều, không lên xuống**. Dù mỗi người hát "gần đúng cao độ trung bình", **ghép 100 người lại vẫn là một nốt đều** — mất hẳn giai điệu.

Thực tế: doanh số **cả cửa hàng** có hình dạng rõ rệt — cuối tuần cao, giữa tuần thấp. Croston SBA dự báo **phẳng cho từng món**; cộng lại cũng **phẳng** → sai hệt nhau ở mọi món và **sai cộng dồn**, vì đây là sai **có hệ thống** (cùng một hướng). Ở cấp đáy sai số ngẫu nhiên từng món còn **che mất** lỗi này; lên cấp tổng thì lộ ra rõ.

**LightGBM thì ngược lại:** nó học **hình dạng chung** (mức bán và chu kỳ tuần) từ mọi món, nên cộng dồn lên vẫn có "giai điệu" đúng.
Số liệu: cấp 1 (cả công ty), RMSSE có trọng số: Croston SBA **1,186**, LightGBM **0,610**, Stacking **0,505** (Test).

*Cách nhớ một câu:* **"Phẳng thì ghép lại vẫn phẳng."**

---

# PHẦN 7. ĐỌC KẾT QUẢ THẾ NÀO CHO ĐÚNG

## 7.1. Bảng chính (Test, 28 ngày d_1914–d_1941)

| Mô hình | MAE | RMSE | RMSSE | WRMSSE | Thời gian |
|---|---:|---:|---:|---:|---|
| Seasonal Naive | 1,244 | 1,752 | 0,997 | 0,870 | 0,7 giây |
| ETS | 1,040 | 1,360 | 0,758 | 0,760 | ~35.121 giây CPU (~2,5 giờ thực) |
| Croston SBA | 1,075 | 1,393 | 0,769 | 1,095 | ~10,5 giây |
| LightGBM | 1,079 | 1,386 | 0,775 | 0,722 | ~228 giây |
| Stacking | 1,050 | 1,351 | 0,757 | 0,668 | 12,1 giây (+ ba mô hình cơ sở) |

Đọc: "MAE 1,040 nghĩa là trung bình mỗi ngày mỗi món dự đoán lệch khoảng 1 sản phẩm."
"RMSSE 0,758 nghĩa là sai số bằng ~76% sai số của cách 'mai bằng hôm nay'; tốt hơn cách ngây thơ khoảng 24%."

## 7.2. Năm thông điệp (Q1 → Q5)

**Q1: Hơn cách ngây thơ bao nhiêu?** Cả bốn mô hình giảm macro-RMSSE khoảng **22–24%** (ETS −23,9%, SBA −22,9%, LightGBM −22,3%, Stacking −24,1%). Nhưng chỉ khoảng **60–65%** số chuỗi thắng cách ngây thơ → cải thiện đến từ **giảm sai số trung bình**, không phải thắng mọi món.

**Q2: Hợp loại hàng nào?**
- Hàng **bán rất thưa** (> 80% ngày 0): ETS (0,795), Stacking (0,796), Croston SBA (0,801) sát nhau, **đều tốt hơn rõ LightGBM (0,820)**.
- Hàng **bán đều** (≤ 50% ngày 0): Stacking (0,705) và ETS (0,708) dẫn; **Croston SBA kém nhất (0,730)** vì đường phẳng bỏ qua chu kỳ tuần, mà hàng bán đều có chu kỳ tuần rõ.
- Ngành HOBBIES sai số chuẩn hóa thấp nhất, FOODS cao nhất; bang TX thấp nhất — chỉ mô tả, không giải thích nhân quả.

**Q3: Đánh đổi chi phí?** Croston SBA **10,5 giây** cho 0,769; ETS **35.121 giây** cho 0,758. Chi phí gấp ~**3.350 lần** (tính tổng CPU) mà sai số chỉ hơn 0,011. Với hàng bán thưa, nên **thử Croston SBA trước**. Stacking cần đủ ba mô hình nên tổng chi phí còn lớn hơn ETS.

**Q4: Cơ chế?** ETS: làm mượt rất mạnh (α nhỏ). Croston SBA: cập nhật theo lần có người mua. LightGBM: dựa vào mức bán trượt 28 ngày và 7 ngày.

**Q5: Ở các cấp tổng hợp?** Thứ hạng đảo: LightGBM tốt, Croston SBA tệ nhất, Stacking tốt nhất.

## 7.3. "Mô hình nào tốt nhất?" — câu trả lời an toàn

> "Trên **Test**, Stacking thấp nhất ở cả hai thước (0,757 và 0,668). Nhưng hơn ETS ở cấp đáy chỉ 0,0015, và trên **Validation** thì **ETS nhỉnh hơn**, LightGBM có WRMSSE thấp nhất. Nên kết luận **vững** chỉ là: bốn mô hình đều hơn rõ cách ngây thơ, và nhóm {ETS, Stacking} hơn nhóm {Croston SBA, LightGBM} khoảng 0,01–0,02 ở cả hai tập."

**Vì sao không tuyên bố "Stacking thắng chắc"?** Vì Test chỉ là **một khoảng 28 ngày**. Giống một trận đấu: thắng một trận chưa chắc mạnh hơn nếu đấu lại trận khác thì đổi.

## 7.4. Khoảng tin cậy bootstrap — hiểu thế nào?

> **Nói đơn giản:** nhóm "bốc lại" ngẫu nhiên 30.490 món (bốc xong bỏ lại, bốc tiếp, có thể trùng) **2.000 lần**, mỗi lần tính lại điểm. 2.000 điểm đó cho thấy điểm **dao động** quanh mức nào. Khoảng 95% giữa nó là **khoảng tin cậy**.

Điểm cần nhớ: khoảng này chỉ cho biết **dao động giữa các món trong cùng 28 ngày đó**. Nó **không** cho biết điểm đổi thế nào khi sang khoảng thời gian khác. Vì vậy khoảng hẹp **không** nghĩa là chắc bền.

## 7.5. Sai số theo thứ trong tuần, theo ngày dự báo
- **Thứ Bảy, Chủ nhật** mọi mô hình sai nhiều nhất: **vì ngày bán nhiều, sai số tuyệt đối tự nhiên lớn**.
- Sai số **tăng từ tuần 1 đến tuần 3, giảm ở tuần 4** (ETS: 0,969 → 1,049 → 1,093 → 1,047). Dự báo xa hơn thường khó hơn, nhưng không đơn điệu.
- ETS có sai số tuyệt đối thấp nhất ở 22 trong 28 ngày dự báo.

---

# PHẦN 8. GIỚI HẠN — NÓI THẲNG, ĐỪNG ĐỢI BỊ HỎI

Giới hạn nào cũng nên được nói như một **"chúng tôi biết điều này"**, không phải "lỗi bị bắt":

1. **Chỉ một cửa sổ Test 28 ngày** → thứ hạng gần nhau có thể đổi. *(Hướng khắc phục: kiểm tra nhiều cửa sổ — rolling-origin.)*
2. **Hai nhóm mô hình nhận thông tin khác nhau** (local chỉ có doanh số; LightGBM có thêm giá/lịch…) → so sánh không hoàn toàn "công bằng thuật toán".
3. **Giả định biết giá ngày dự báo** từ trước.
4. **Test không còn hoàn toàn mù** (1.200 chuỗi pilot đã bị xem nhãn).
5. **Khoảng tin cậy chỉ phản ánh dao động giữa các món**, thực tế còn rộng hơn vì các món cùng cửa hàng/cùng ngày không độc lập.
6. **Thời gian chạy không so thẳng được** giữa local và global (đo khác phạm vi) và Stacking chạy trên máy khác.
7. **Stacking** chỉ có 2 lần tập dượt, tối ưu theo cấp đáy.
8. **Chưa đo chi phí tồn kho** → chưa kết luận "doanh nghiệp lợi bao nhiêu".
9. **LightGBM dùng mặc định**, chưa tuning, chưa dùng hàm mất mát Tweedie như đội thắng M5.
10. **Báo cáo/slide do AI soạn từ kết quả thật, chưa được các thành viên review hết.**

---

# PHẦN 9. TỰ KIỂM TRA (che đáp án, trả lời miệng)

## 9.1. Thẻ ghi nhớ nhanh (câu hỏi → đáp án một dòng)

1. Đồ án làm gì? → So sánh 5 cách dự báo doanh số 28 ngày của 30.490 chuỗi Walmart.
2. 30.490 là gì? → 3.049 sản phẩm × 10 cửa hàng.
3. Horizon là gì? → Số ngày dự báo (28).
4. Origin là gì? → Ngày cuối cùng đã biết dữ liệu, "hôm nay".
5. Tỷ lệ ngày bán 0? → 68,4%.
6. Train/Val/Test chia thế nào? → d_1–1885 / d_1886–1913 / d_1914–1941.
7. Vì sao chia theo thời gian? → Dự báo nhìn về tương lai; xáo trộn sẽ làm rò rỉ.
8. Rò rỉ là gì? → Mô hình biết trước thông tin tương lai, điểm đẹp giả.
9. Vì sao lag ≥ 28? → Dự đoán trực tiếp cả 28 ngày từ một mốc, ngày thứ 28 chỉ chắc chắn biết đến mốc đó.
10. config-freeze? → "Niêm phong" cấu hình trước khi chạy Test; Test chạy một lần.
11. Seasonal Naive làm gì? → Lặp lại 7 ngày cuối.
12. α nhỏ nghĩa là? → Làm mượt mạnh, ít tin vào số mới nhất.
13. ETS(A,N,A)? → Sai số cộng, không xu hướng, có mùa vụ cộng.
14. AICc? → Điểm cân bằng độ khớp và độ phức tạp, thấp = tốt.
15. Croston chia thành gì? → z (mỗi lần bán bao nhiêu) và p (cách mấy ngày một lần); dự báo z/p.
16. SBA thêm gì? → Nhân (1 − α/2) = 0,95 để bớt chệch lên.
17. Boosting là gì? → Nhiều cây nhỏ nối tiếp, cây sau sửa lỗi cây trước.
18. LightGBM nhìn gì nhiều nhất? → Trung bình trượt 28 ngày (68,5%) và 7 ngày (18,2%).
19. Stacking dùng gì để trộn? → Hồi quy Ridge trên dự báo tập dượt trong Train.
20. Trọng số Stacking? → ETS 0,543; LightGBM 0,371; SBA 0,138.
21. RMSSE < 1 nghĩa là? → Tốt hơn cách "mai bằng hôm nay" trên Train.
22. Vì sao không dùng MAPE? → Chia cho 0 khi doanh số 0.
23. R² âm nghĩa là? → Kém hơn việc đoán bằng trung bình của chính 28 ngày thật; phổ biến với hàng thưa.
24. Macro khác Pooled? → Trung bình các món / gộp mọi điểm rồi tính.
25. WRMSSE là gì? → RMSSE có trọng số doanh thu, trung bình 12 cấp tổng hợp.
26. Mô hình thắng ở macro-RMSSE Test? → Stacking 0,757 (ETS 0,758).
27. Mô hình thắng ở WRMSSE Test? → Stacking 0,668.
28. Vì sao Croston SBA tệ ở cấp tổng? → Dự báo phẳng, ghép lại vẫn phẳng, mất chu kỳ tuần.
29. Vì sao LightGBM tốt ở cấp tổng? → Học chung mức và chu kỳ tuần của mọi chuỗi.
30. ETS chạy bao lâu? → ~35.121 giây CPU, ~2,5 giờ thực.
31. Croston SBA nhanh hơn ETS bao nhiêu? → ~3.350 lần (tổng CPU).
32. Local vs Global? → Mỗi chuỗi một mô hình / một mô hình cho mọi chuỗi.
33. Pilot là gì? → Chạy thử 1.200 chuỗi với 11 mô hình để chọn 5.
34. Vì sao chạy toàn bộ dữ liệu? → Bỏ sai số lấy mẫu và tính được WRMSSE 12 cấp.
35. Giới hạn lớn nhất? → Chỉ một cửa sổ Test 28 ngày.

## 9.2. Câu trả lời mẫu 30 giây cho 6 câu hay gặp

**"Vì sao chọn 5 mô hình?"**
"Pilot 1.200 chuỗi chạy 11 mô hình. Chúng em giữ **mỗi cách tiếp cận một đại diện**: mốc ngây thơ, làm mượt mũ, chuyên cho hàng thưa, học máy toàn cục, và kết hợp. XGBoost và CatBoost ngang LightGBM nhưng chậm hơn, CatBoost ước tính khoảng 14 giờ CPU; Croston cổ điển kém bản SBA; Linear Regression kém nhất trong nhóm học máy; LightGBM tinh chỉnh không giữ được lợi thế trên Test."

**"Làm sao biết không rò rỉ dữ liệu?"**
"Ba lớp: đặc trưng lấy từ doanh số luôn lùi ít nhất 28 ngày; nhãn sau mốc dự báo bị che trước khi tạo đặc trưng; và có bài kiểm thử tự động làm hỏng nhãn tương lai rồi xác nhận kết quả không đổi. Với Stacking, trọng số chỉ học từ dự báo tập dượt trong Train, không dùng nhãn Validation hay Test."

**"Mô hình nào tốt nhất?"**
(Xem mục 7.3.)

**"RMSSE và WRMSSE khác gì?"**
"RMSSE chấm từng chuỗi ở cấp đáy, chuẩn hóa theo sai số của cách ngây thơ rồi lấy trung bình. WRMSSE là thước của cuộc thi M5: cộng dự báo lên 12 cấp, có trọng số doanh thu, lấy trung bình 12 cấp. Hai thước cho thứ hạng khác nhau: Croston SBA tốt ở cấp đáy nhưng WRMSSE tệ nhất."

**"Stacking có đáng để dùng không?"**
"Về độ chính xác thì tốt nhất trên Test, đặc biệt ở cấp tổng hợp. Nhưng nó cần chạy đủ ba mô hình nên tổng chi phí còn lớn hơn ETS, và lợi thế ở cấp đáy rất nhỏ. Tùy mục tiêu của doanh nghiệp: cần tiết kiệm thì Croston SBA; cần chính xác nhất quán ở nhiều cấp thì Stacking hoặc LightGBM."

**"Đồ án đóng góp gì?"**
"Không phải thuật toán mới. Đóng góp là một phép so sánh sạch năm cách tiếp cận trên toàn bộ 30.490 chuỗi với cùng horizon và độ đo, phân tích theo mức gián đoạn và theo 12 cấp WRMSSE, kèm chi phí chạy. Phát hiện chính: ở hàng bán rất thưa, mô hình cục bộ thắng LightGBM; ở cấp tổng hợp thì ngược lại."

## 9.3. Câu hỏi "bẫy" và cách tránh

- **"Chứng minh Stacking hơn ETS đi?"** → Đừng nói "chứng minh". Nói: khoảng tin cậy không chứa 0 *trong cửa sổ này*, nhưng trên Validation chiều ngược lại, nên không coi là bền.
- **"LightGBM kém hả?"** → Kém ở **cấp đáy** nhưng **tốt hơn ETS ở WRMSSE** (0,722 vs 0,760 Test; 0,591 vs 0,724 Validation).
- **"ETS mất 35.121 giây à?"** → Đó là **tổng CPU**; thực tế ~2,5 giờ nhờ chạy 4 tiến trình song song.
- **"Giá không ảnh hưởng à?"** → Đừng khẳng định. Giá chỉ đóng góp ~1,9% vào **dự báo** của LightGBM; đó không phải kết luận nhân quả.
- **"Điểm của nhóm xếp hạng mấy trong M5?"** → Không xếp hạng. Cùng thước đo và cùng cửa sổ công khai, nhưng **không phải bài nộp thi**; đội thắng đạt 0,520.
- **"Test có thật sự khách quan không?"** → Cấu hình khóa trước, chạy một lần; nhưng nhãn Test của 1.200 chuỗi pilot đã xem trước khi chọn 5 mô hình — nhóm đã ghi trong giới hạn.

---

# PHẦN 10. 10 CON SỐ PHẢI THUỘC

1. **30.490** chuỗi (3.049 sản phẩm × 10 cửa hàng); **42.840** chuỗi / **12 cấp**.
2. **28** ngày dự báo.
3. **68,4%** ngày bán 0; **39,1%** chuỗi có > 80% ngày 0.
4. Train **d_1–1885**, Validation **d_1886–1913**, Test **d_1914–1941**.
5. Pilot **1.200** chuỗi, **11** mô hình → giữ **5**.
6. Test macro-RMSSE: Seasonal Naive **0,997**, ETS **0,758**, Croston SBA **0,769**, LightGBM **0,775**, Stacking **0,757**.
7. Test WRMSSE: **0,870 / 0,760 / 1,095 / 0,722 / 0,668** (cùng thứ tự).
8. Cải thiện so với cách ngây thơ: **22–24%**.
9. Thời gian: Croston SBA **~10 giây**, LightGBM **~4 phút**, ETS **~2,5 giờ**.
10. Trọng số Stacking: ETS **0,543**, LightGBM **0,371**, SBA **0,138**.

**Mẹo nhớ thứ tự thí sinh:** *"Naive – ETS – Croston – LightGBM – Stacking"* = "Ngây thơ → Mượt → Thưa → Máy học → Hội đồng". Mỗi mô hình sau là một cách nghĩ khác.

---

# PHẦN 11. TỪ ĐIỂN THUẬT NGỮ (A → Z, ngôn ngữ đời thường)

| Thuật ngữ | Giải thích tự nhiên | Cách nhớ |
|---|---|---|
| **AICc** | Điểm chọn mô hình: cân bằng "khớp tốt" và "không rườm rà", thấp = tốt | Chọn món vừa ngon vừa không cầu kỳ |
| **α (alpha)** | Mức tin vào số mới nhất khi làm mượt | Độ hay đổi ý |
| **Backtest / tập dượt** | Dự đoán thử trong quá khứ ở một "hôm nay giả" để biết mô hình đúng sai | Thi thử |
| **Bootstrap** | Bốc lại ngẫu nhiên có hoàn lại nhiều lần để xem điểm dao động | Rút thăm lại 2.000 lần |
| **Boosting** | Nhiều cây nối tiếp, cây sau sửa lỗi cây trước | Chữa bài từng vòng |
| **Cắt về 0 (clip)** | Dự đoán âm thì đổi thành 0 vì doanh số không thể âm | Không ai bán −2 cái |
| **Chuỗi (series)** | Doanh số theo ngày của một sản phẩm ở một cửa hàng | Một cột sổ bán hàng |
| **Cấp (level)** | Một "góc nhìn" cộng dồn (cả công ty, từng bang…) | Zoom in/zoom out |
| **Croston** | Dự báo hàng thưa bằng z (bán bao nhiêu) và p (cách mấy ngày) | Bao lâu bán 1 lần × mỗi lần mấy cái |
| **Dự báo điểm** | Mỗi ngày một con số | Một con số, không khoảng |
| **Đặc trưng (feature)** | Thông tin đưa vào mô hình | Câu hỏi giáo viên hỏi |
| **ETS** | Làm mượt mũ: Error–Trend–Seasonal | Trung bình có trí nhớ phai dần |
| **Gain (importance)** | Mức đóng góp của một đặc trưng vào việc giảm sai số | Ai "công" nhiều nhất |
| **Global / Local** | Một mô hình chung / mỗi chuỗi một mô hình | Thầy cả lớp / gia sư riêng |
| **Horizon** | Số ngày dự báo | Nhìn xa bao nhiêu |
| **Intermittent demand** | Nhu cầu gián đoạn, nhiều ngày bán 0 | Tiệm đồ hiếm |
| **Khoảng tin cậy 95%** | Khoảng mà điểm "thật" nhiều khả năng nằm trong | Biên độ dao động |
| **Lag (độ trễ)** | Doanh số của k ngày trước | Nhìn lại k ngày |
| **LightGBM** | Phiên bản nhẹ, nhanh của boosting | Hội đồng cây nhanh |
| **MAE** | Trung bình trật bao nhiêu cái | Thầy dễ tính |
| **MAPE** | Sai số theo %, hỏng khi doanh số = 0 | Chia cho 0 |
| **Macro / Pooled** | Trung bình các chuỗi / gộp mọi điểm | Mỗi món một phiếu / ai bán nhiều thì nói to |
| **Mùa vụ (seasonality)** | Chu kỳ lặp, ở đây là 7 ngày | Cuối tuần đông khách |
| **Naive** | Ngây thơ: lấy quá khứ gần nhất làm dự đoán | "Mai như hôm nay" |
| **Nhiễu (noise)** | Dao động ngẫu nhiên không giải thích được | Đột nhiên có đoàn khách |
| **OOF (out-of-fold)** | Dự đoán "ngoài mẫu" dùng để huấn luyện bộ trộn | Bài thi thử có đáp án |
| **Origin** | Ngày cuối cùng đã biết, "hôm nay" | Cột mốc đứng nhìn tương lai |
| **Overfitting (quá khớp)** | Học vẹt dữ liệu cũ, gặp mới thì sai | Thuộc đáp án đề cũ |
| **Pilot** | Thử nhỏ để chọn cách làm | Nấu thử một mẻ |
| **R²** | So mô hình với việc đoán bằng trung bình; âm = kém hơn | Có hơn "đoán đại số trung bình" không |
| **Refit** | Học lại trên dữ liệu mới hơn trước khi chạy Test | Ôn thêm trước thi thật |
| **Ridge** | Hồi quy tuyến tính có phạt để hệ số không phình to | Bộ trộn khiêm tốn |
| **RMSE** | Giống MAE nhưng phạt nặng lỗi lớn | Thầy ghét lỗi to |
| **RMSSE** | RMSE chuẩn hóa theo "cách ngây thơ"; < 1 là giỏi hơn | Điểm so với bạn ngây thơ |
| **Rò rỉ (leakage)** | Mô hình biết trước tương lai | Lộ đề |
| **SBA** | Croston nhân (1 − α/2) cho bớt chệch | Hạ nhiệt cho đỡ lố |
| **SNAP** | Chương trình tem thực phẩm Mỹ, ảnh hưởng doanh số thực phẩm | Ngày được mua bằng tem |
| **Stacking** | Trộn dự báo của nhiều mô hình bằng trọng số học được | Họp hội đồng chuyên gia |
| **Test** | Tập thi thật, chạy một lần | Kỳ thi thật |
| **Train** | Tập học | Bài học |
| **Tweedie** | Hàm mất mát hợp dữ liệu nhiều số 0 (đội thắng M5 dùng) | Thước đo riêng cho hàng thưa |
| **Validation** | Tập thi thử để chỉnh cách làm | Thi thử |
| **WRMSSE** | RMSSE có trọng số doanh thu, trung bình 12 cấp | Chấm từ 12 góc nhìn, ưu tiên món nhiều tiền |
| **z, p** | Trong Croston: lượng bán mỗi lần / khoảng cách giữa hai lần bán | "Bao nhiêu" và "bao lâu" |

---

# PHẦN 12. LỘ TRÌNH ÔN TẬP GỢI Ý

**Buổi 1 (≈ 1,5 giờ): Hiểu nền tảng.** Đọc Phần 1 → 3. Tự tính tay lại ví dụ MAE/RMSE/RMSSE (mục 2.3). Chép lại 5 thí sinh bằng lời của mình.

**Buổi 2 (≈ 1,5 giờ): Hiểu từng mô hình và kết quả.** Đọc Phần 4 → 7. Với mỗi mô hình, tự nói được: *ý tưởng – ví dụ số – điểm mạnh – điểm yếu – vì sao ra kết quả đó*. Học kỹ mục 6.3 ("phẳng thì ghép lại vẫn phẳng").

**Buổi 3 (≈ 1 giờ): Tự kiểm tra.** Che đáp án mục 9.1 và trả lời miệng. Tập trả lời 6 câu mẫu mục 9.2 **thành lời, có bấm giờ**. Đọc kỹ mục 7.3 và Phần 8 (giới hạn). Học thuộc Phần 10.

**Trước giờ bảo vệ:** đọc lại Phần 10 và mục 9.3 (câu bẫy). Nếu thật sự không biết câu nào, nói thật: *"Phần này nhóm chưa đo/chưa kiểm chứng, hiện chúng em chỉ có kết quả …"* — thành thật được đánh giá tốt hơn đoán bừa.

*(Thời lượng seminar chính thức nhóm chưa xác minh với giảng viên; nhớ hỏi để cắt phần nói cho vừa giờ.)*

---

# PHẦN 13. KHI CẦN CHI TIẾT HƠN

| Muốn biết | Mở |
|---|---|
| Bảng số liệu đầy đủ, bootstrap, WRMSSE, theo nhóm | [`KIEN_THUC_DO_AN.md`](KIEN_THUC_DO_AN.md) mục 8 |
| 35 câu hỏi đáp chi tiết | [`KIEN_THUC_DO_AN.md`](KIEN_THUC_DO_AN.md) mục 13 |
| Việc còn tồn đọng, giới hạn đầy đủ | [`KIEN_THUC_DO_AN.md`](KIEN_THUC_DO_AN.md) mục 11, 16 |
| Bản thảo báo cáo gốc | `report/draft_report.md` |
| Hỏi đáp chính thức của nhóm | `report/qa.md` |
| Tóm tắt bài báo M5 | `paper/summary.md` |
| Bảng gốc từng kết quả | `results/full/tables/{validation,test}/` |
