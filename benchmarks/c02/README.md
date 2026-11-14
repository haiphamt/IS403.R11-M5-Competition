# C02 — Hướng dẫn chạy Kaggle/Colab và bàn giao

Thanh Tiền · IS403.R11 · Bàn giao nội bộ · Soạn/đối chiếu nguồn ngày 02/11/2026.

Tài liệu theo task được gom trong thư mục này: [hồ sơ kiểm chứng và kết quả](VALIDATION.md), dependencies tại `requirements.txt`. Mã nguồn, notebook và test nằm trong các thư mục chung của repo. Nội dung bàn giao được ghi trong mô tả PR và Jira; hồ sơ cũ chỉ giữ ở `local/c02/`.

[HTML có code và kết quả thực nghiệm](../../results/c02/C02_20261113T093749_327948+0700/c02_benchmark_xgb_cat.html) được xuất trực tiếp từ notebook đã chạy, run `C02_20261113T093749_327948+0700`. Tải tệp HTML và mở bằng trình duyệt để xem đầy đủ. Phần chạy của C02 yêu cầu 10 chuỗi; đánh giá Test hoặc mẫu 1.200 chuỗi thuộc các task sau.

## 1. C02 cần hoàn thành điều gì?

Theo Task_Tracker, C02 là **viết code XGBoost và CatBoost, chạy thử trên 10 chuỗi và ghi thời gian**; Thủy Tiên xác nhận. Đây là thử tính khả thi trước khi C05 chốt cỡ mẫu, chưa phải M05/M06 trên mẫu chính thức. C02 không yêu cầu tuning, chạy Test, tạo bài nộp dự đoán Kaggle hay tái lập lời giải vô địch.

Đầu ra chính: mã nguồn có thể chạy lại + bảng thời gian **đo thật trên Kaggle/Colab**. Notebook còn tính MAE/RMSE/RMSSE/R² để kiểm tra pipeline. Các điểm số này chưa đại diện 30.490 chuỗi hay mẫu D02. Mẫu 10 chuỗi được chọn bằng seed 42, mỗi cửa hàng một chuỗi, luân phiên đủ ba category trên bộ M5 đầy đủ; không khẳng định mẫu đại diện thống kê.

Hai mô hình là **global model**: mỗi thuật toán học một mô hình chung từ dữ liệu của 10 chuỗi. Không fit 10 mô hình độc lập. Vì thế không lấy thời gian fit chia 10 rồi nhân 1.200 làm dự báo chắc chắn cho C05.

## 2. Các tệp bạn sử dụng

| Tệp | Vai trò |
| --- | --- |
| `../../notebooks/c02_benchmark_xgb_cat.ipynb` | Import lên Kaggle; tự chứa toàn bộ code, giải thích, ô cấu hình và bảng kết quả. Không cần clone repo hoặc upload `.py` cùng notebook. |
| `../../src/c02_benchmark.py` | Mã nguồn Python để bàn giao/review hoặc chạy bằng dòng lệnh. Cùng logic với notebook. |
| `VALIDATION.md` | Kết quả thực nghiệm đã đọc, kiểm tra mã và giới hạn bằng chứng. |
| `../../tests/test_c02_benchmark.py` | Kiểm tra số học, leakage, dữ liệu thiếu, categorical và fit/predict trên dữ liệu tổng hợp có gắn nhãn rõ. Không phải số đo M5. |

## 3. Chuẩn bị trên Kaggle

1. Đăng nhập [Kaggle M5 Forecasting – Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy), chấp nhận điều khoản nếu tài khoản được yêu cầu. Không đưa token/API key vào notebook.
2. Mở trang [Kaggle Code](https://www.kaggle.com/code), tạo notebook Python. Trong menu File/Import Notebook, chọn file `c02_benchmark_xgb_cat.ipynb` trên máy. Tên menu có thể khác theo giao diện; thao tác cần làm là **import notebook hiện có**, không dán toàn bộ JSON vào code cell.
3. Thêm dữ liệu cuộc thi vào Input bằng **Add Input/Add Data → Competitions → M5 Forecasting – Accuracy**. C02 thực sự sử dụng `sales_train_validation.csv`, `calendar.csv`, `sell_prices.csv`. Bộ cuộc thi đầy đủ có thêm `sales_train_evaluation.csv` và `sample_submission.csv`; notebook C02 không đọc nhãn Test.
4. Đặt accelerator **None/CPU**. Hai mô hình đã chỉ định CPU trong code; bật GPU không làm code tự chuyển sang GPU. Giữ `THREADS=2` để số đo dễ so sánh; ghi cấu hình thực tế từ output, không sao chép thông số của người khác.
5. Chạy từ trên xuống. Ô đầu kiểm tra thư viện. Kaggle thường có các thư viện nhưng notebook sẽ kiểm tra thật. Nếu thiếu, bật Internet rồi chạy một ô riêng:

   ```python
   %pip install "numpy>=1.24" "pandas>=2.0" "scikit-learn>=1.2" "xgboost>=2.0" "catboost>=1.2" "psutil>=5.9"
   ```

   Nếu vừa thay thư viện đã import, restart session rồi Run All. Không để lệnh nâng cấp thư viện chạy lại giữa các lần đo. Cần so sánh nhiều notebook thì dùng cùng phiên bản; `requirements_observed.txt` ghi phiên bản thực tế sau lần chạy.

Notebook không yêu cầu Internet để lấy code; Internet chỉ cần nếu phải cài thư viện. Dữ liệu đã thêm vào Input vẫn phải sẵn có trong session.

### Chạy trên Colab — tải CSV từ máy (cấu hình hiện hành)

1. Mở https://colab.research.google.com/ → File → Upload notebook, chọn `notebooks/c02_benchmark_xgb_cat.ipynb`. Chọn runtime Python/CPU.
2. Chạy mục 1: lệnh `%pip install` cài thư viện vào runtime hiện tại, sau đó in phiên bản. Nếu Colab yêu cầu khởi động lại do thư viện đã nạp bị thay đổi, khởi động lại rồi chạy từ mục 1. Mỗi runtime mới đều cần chạy ô này; output lưu từ phiên cũ không chứng minh gói còn được cài.
3. Chạy ô mục 2. Khi xuất hiện nút chọn tệp, mở thư mục `D:/MonHoc_UIT/IS403.R11 (PTDLKD)/Đồ án/data/raw/m5_drive` trên máy và chọn đúng ba CSV `sales_train_validation.csv`, `calendar.csv`, `sell_prices.csv`. Đợi tải xong; notebook dùng chúng tại `/content/m5`, không cần mount Drive. File có sẵn trong session sẽ được dùng lại.
4. Chạy từng mục: mục 3 đọc dữ liệu và hiện 10 chuỗi; mục 4 tạo đặc trưng, kiểm tra rò rỉ và chia tập; mục 5 định nghĩa độ đo; mục 6 định nghĩa mã hóa/mô hình; mục 7 ghi môi trường; mục 8 huấn luyện và dự báo; mục 9 tổng hợp kết quả. Kiểm tra 18.850 dòng Train, 280 Validation, leakage PASS, bảng trials 6 dòng, summary 2 dòng và metric hữu hạn. Giữ 10 chuỗi/seed 42/3 repeats/2 threads.
5. Lưu notebook có output vào Drive hoặc tải notebook đã chạy về máy. Các tệp kết quả ở `/content/c02_results/C02_<timestamp>`; mở biểu tượng thư mục bên trái Colab để tải các tệp cần bàn giao. `/content` là bộ nhớ phiên chạy, cần lưu kết quả trước khi phiên bị xóa. Đối chiếu cùng run_id và gửi Thủy Tiên xác nhận.

Notebook hiện cấu hình riêng cho Colab. Phần Kaggle ở guide mô tả cách chạy thay thế, cần thay ô upload bằng đường dẫn Input khi sử dụng Kaggle.

## 4. Ô cấu hình nên để thế nào?

**Nguồn Drive thay thế (đã kiểm tra 04/11/2026):** thư mục
[Mcompetitions Dataset](https://drive.google.com/drive/folders/1wxz-TAfVE7uKGCjh405eCb2Q_pG3kAm9)
có đủ ba CSV dùng cho C02. Bản này có 30.490 dòng sales, 1.969 dòng lịch và
6.841.121 dòng giá, nhưng sales thiếu `id` và lịch thiếu `d` so với Kaggle.
Notebook bản 04/11 nhận trực tiếp CSV gốc: ghép `item_id_store_id_validation`
và tính khóa ngày từ 2011-01-29 trong bộ nhớ. Không sửa, đổi tên hay ghi đè CSV.
Metadata `inputs.input_adaptations` ghi rõ việc bổ sung khóa. Nếu upload ba
CSV từ Drive làm Input, dùng notebook bản mới trong `notebooks/`; bản đang mở
trên Kaggle có thể vẫn là code cũ và cần import lại. Giữ dataset upload ở chế
độ Private. Nguồn gốc và kiểm tra đầu vào thật nằm trong `local/c02/C02_DRIVE_INPUT_CHECK.json` (hồ sơ cũ chỉ giữ local).
Đây chưa phải số đo huấn luyện, chưa thay thế S06 đọc đủ năm file theo README nhóm.

```python
DATA_DIR = None           # tự tìm đúng bộ CSV dưới /kaggle/input
OUTPUT_DIR = "/kaggle/working/c02_results"
N_SERIES = 10
SEED = 42
REPEATS = 3
THREADS = 2
```

Nếu tự tìm gặp nhiều bộ dữ liệu, notebook sẽ dừng và liệt kê các đường dẫn. Copy đúng đường dẫn từ Input rồi đặt `DATA_DIR`, ví dụ `/kaggle/input/m5-forecasting-accuracy`. Không bắt buộc đường dẫn phải đúng tên ví dụ này; Kaggle có thể mount sâu hơn. Thư mục phải chứa ba CSV kể trên, không phải file ZIP chưa giải nén.

Ba lần lặp giúp thấy dao động; mỗi lần tạo mô hình mới, cùng seed, chạy đủ số vòng mặc định của phiên bản thư viện. Thứ tự XGBoost/CatBoost đổi luân phiên. Không bỏ lần đầu, không chọn lần nhanh nhất. Bảng tổng hợp dùng median và có min/max. Không tuning hoặc dùng Validation để early stopping.

**Dừng và kiểm tra nếu:** số chuỗi khác 10; Train khác 18.850 dòng; Validation khác 280 dòng; kiểm tra leakage không PASS; metric có NaN/inf; hoặc số vòng huấn luyện/config không hiện trong metadata. Các dòng d_1–d_1885 vẫn được giữ, kể cả đầu chuỗi chưa đủ lag/rolling; NaN ở những feature này được xử lý native.

## 5. Các bước code đang làm và lý do

### Đọc dữ liệu và lấy mẫu

Đọc metadata trước để chọn chuỗi bằng seed, không nhìn doanh số tương lai. Quét sales theo chunk và chỉ giữ 10 dòng wide đủ lịch sử; giá cũng được lọc theo cặp store–item. Cách này giảm RAM lưu bảng, nhưng vẫn cần quét CSV gốc nên thời gian đọc không thể ngoại suy tuyến tính theo số chuỗi. Không ghép chồng hai file sales có lịch sử trùng nhau.

### Chuyển chuỗi thành bảng supervised learning

Mỗi dòng là `(chuỗi, ngày)`, nhãn là số lượng bán ngày đó. Đầu vào: item/dept/category/store/state, lịch, hai cặp event, SNAP, giá và `price_missing`, ba lag và bốn rolling feature. Cột `sales` và `day_num` không nằm trong danh sách feature. Không dùng target encoding tự tính trên toàn bộ dữ liệu.

Với ngày đích t:

- `lag_28 = y[t−28]`, tương tự lag 35 và 42.
- `rolling_mean_7 = mean(y[t−34], …, y[t−28])`.
- `rolling_mean_28 = mean(y[t−55], …, y[t−28])`.
- Rolling std dùng cùng cửa sổ, **ddof=1** (độ lệch chuẩn mẫu của pandas).

Ở ngày Validation cuối t=1913, dữ liệu bán mới nhất được dùng là `y[1885]`, vẫn thuộc Train. Dự báo cả 28 ngày bằng một lần predict, không đưa dự báo trở lại làm lag, không cập nhật bằng nhãn Validation.

Notebook còn thay đổi toàn bộ nhãn Validation rồi tạo lại đặc trưng: ma trận feature phải giữ nguyên. Nó đối chiếu riêng công thức lag/rolling tại đầu và cuối horizon cho từng chuỗi. Join calendar/price kiểm tra khóa many-to-one, tránh nhân đôi dòng.

### Giá và thông tin ngoại sinh

Giữ NaN cho giá thiếu và thêm `price_missing`; không điền ngược bằng giá tuần tương lai, không biến giá thiếu thành giá 0. **Giả định thí nghiệm:** giá theo tuần của ngày dự báo trong M5 được coi là đầu vào ngoại sinh biết trước. Đây là điều kiện của bài toán thực nghiệm, không phải bằng chứng hệ thống bán lẻ thực tế luôn biết giá 28 ngày tới. Kiểm tra perturbation chỉ chứng minh không dùng nhãn Validation, không tự chứng minh mọi biến ngoại sinh đã sẵn có tại thời điểm dự báo.

### XGBoost và CatBoost

Boosting cộng đóng góp của nhiều cây: `f_M(x) = b + Σ η_m T_m(x)`. Cây sau được học để cải thiện hàm mất mát của tổ hợp hiện tại. XGBoost dùng thông tin đạo hàm bậc nhất/bậc hai và regularization để xây cây; `hist` là cách tìm split bằng histogram. CatBoost mặc định dùng cây đối xứng và có cơ chế xử lý categorical bằng thống kê/biến đổi nội bộ; không đồng nhất mọi lần chạy CatBoost với `boosting_type=Ordered`. Giá trị thực tế nằm trong `resolved_parameters`.

Code giữ mặc định thư viện cho số cây, depth, learning rate… và chỉ đặt các lựa chọn vận hành: objective/loss hồi quy, CPU, seed, số thread, `hist`/categorical của XGBoost, tắt ghi log file của CatBoost. Số thread áp dụng cả fit và predict. Không gọi các tham số tự chọn 100 cây/500 cây và learning rate thủ công là “mặc định”. Giá trị mặc định phụ thuộc phiên bản, vì vậy metadata lưu cả constructor và tham số **đã resolve sau fit**, kể cả số vòng/số cây thực tế. So sánh này là chi phí của hai cấu hình mặc định, không phải benchmark hai thuật toán với số cây bằng nhau. Chưa khẳng định đây là cấu hình nhóm đã freeze cho M05/M06.

XGBoost: fit từ điển pandas category **chỉ trên Train**, áp cùng dictionary cho Validation. Category chưa thấy được chuyển thành missing, có đếm trong metadata. Không ép category riêng ở mỗi tập gây lệch mã. CatBoost: truyền chuỗi và khai báo đủ `cat_features`, để mô hình xử lý category nội bộ. Hai mô hình dùng cùng dữ liệu và feature gốc, cách biểu diễn categorical có khác theo API. Dự báo âm được clip về 0 cho cả hai; không làm tròn số.

### Cách hiểu độ đo

Đặt e = y − ŷ, n là số dòng Validation:

- `MAE = mean(|e|)`: sai số tuyệt đối trung bình, đơn vị sản phẩm/ngày, càng thấp càng tốt.
- `RMSE = sqrt(mean(e²))`: nhấn mạnh sai số lớn, cùng đơn vị với y, càng thấp càng tốt.
- `R² = 1 − Σe²/Σ(y−ȳ)²`: tính **pooled** trên tất cả chuỗi/ngày; có thể âm. Chênh lệch quy mô giữa chuỗi ảnh hưởng cách đọc. Nếu toàn bộ nhãn hằng: hoàn hảo trả 1, còn lại trả 0, quy ước ghi rõ trong metadata.
- `RMSSE_i = sqrt(MSE_Validation_i / mean((y_t−y_(t−1))²)_Train_i)`: mẫu số bỏ các số 0 đầu chuỗi trước lần bán đầu tiên, chỉ dùng Train. `RMSSE_mean` là trung bình không trọng số theo chuỗi.

Chuỗi toàn 0, chỉ có một quan sát sau lần bán đầu, hoặc lịch sử hoạt động hằng làm scale không xác định/bằng 0. C02 dùng **scale=1, có cờ fallback** để trả điểm hữu hạn theo yêu cầu nhóm. Đây là quy ước bổ sung cần thống nhất với chủ D05 trước thực nghiệm chính thức, không được gọi là RMSSE chính thức cho các chuỗi suy biến. Metadata lưu scale/cờ từng chuỗi. Không dùng `RMSSE < 1` để khẳng định thắng Seasonal Naive 28 ngày nếu chưa chạy cùng baseline trên cùng Validation. `RMSSE_mean` ở đây không phải **WRMSSE 12 cấp có trọng số doanh thu của cuộc thi M5**.

## 6. Đọc bảng thời gian cho đúng

| Cột | Gồm những gì? |
| --- | --- |
| `encoding_s` | Copy feature và chuyển kiểu categorical cho từng mô hình |
| `fit_s` | Lời gọi fit, gồm chuyển đổi dữ liệu nội bộ của thư viện |
| `predict_s` | Predict + clip về không âm; không gồm score |
| `score_s` | Tính các metric và scale từng chuỗi |
| `model_total_s` | Toàn khối encode → tạo model → fit → predict → score; có overhead nhỏ |
| `load_s` | Tìm sample/đọc và lọc CSV; không gồm import thư viện |
| `features_s` | Tạo feature dùng chung một lần |
| `audit_s` | Kiểm tra feature/leakage, có tạo lại feature một lần |
| `pipeline_wall_s_all_repeats` | Toàn pipeline, gồm phần dùng chung và mọi lần lặp; không gồm ghi file và lấy thông tin môi trường |
| `measured_work_s_all_repeats` (notebook từng bước) | Tổng load + features + audit + thời gian khối ba lần đo hai mô hình; không gồm lúc dừng giữa các ô. Khác thời gian toàn phiên notebook |

`load_s`, `features_s`, `audit_s` nằm trong metadata và báo cáo riêng; không cộng chúng lặp lại vào mỗi model. Median các cột thành phần không nhất thiết cộng đúng bằng median tổng. Không so trực tiếp toàn pipeline C02 với thời gian một hàm predict ở C04 hoặc thời gian fit riêng từng chuỗi ở ETS.

Mặc định chưa đo peak RAM; chỉ ghi RAM tổng/khả dụng đầu phiên. Không kết luận không thiếu RAM ở 1.200 chuỗi dựa vào thông tin này. Nếu cần quyết định C05, tiếp tục benchmark 50–100 chuỗi trên cùng máy và cấu hình; giữ kết quả mỗi cỡ mẫu riêng. Không tự giảm mẫu xuống 300, không tự chạy Test.

## 7. Lưu kết quả và bàn giao Thủy Tiên

Sau khi Run All thành công, kết quả nằm trong `/kaggle/working/c02_results/C02_<timestamp>` (Kaggle) hoặc `OUTPUT_DIR` bạn đã đặt (Colab). Notebook hiển thị bảng và run_id; xem `result["output_dir"]` nếu cần đường dẫn đầy đủ, rồi mở thư mục output để tải từng tệp:

- `runtime_summary.csv`: bảng median/min/max runtime và metric, số chuỗi/số lần lặp.
- `runtime_trials.csv`: từng lần đo, không làm tròn hoặc đổi số thành chuỗi “giây”.
- `selected_series.csv`: ID và metadata của đúng các chuỗi đã chạy.
- `run_metadata.json`: run_id, môi trường, versions, cấu hình, SHA256 code, checks, scale RMSSE từng chuỗi.
- `requirements_observed.txt`: phiên bản đã dùng, để tái lập.
- `run_report.md`: báo cáo tạo từ kết quả thật, đủ phạm vi và hạn chế.

Chọn **Save Version → Save & Run All** rồi chờ phiên chạy hoàn tất. Đây là phiên chạy mới từ đầu, nên hãy dùng các tệp và notebook output của **cùng một phiên/run_id** để bàn giao. Xem hướng dẫn chính thức [Kaggle Notebooks](https://www.kaggle.com/docs/notebooks). Không cần bấm Submit lên leaderboard.

Bàn giao nội bộ: mã `.py`, liên kết notebook đã chạy, bảng CSV, metadata và báo cáo. Notebook nguồn `.ipynb` giữ để làm việc; không dùng nó thay HTML nếu chuyển sang quy định bản nộp bài tập của môn học. Không tạo ZIP.

Nếu cần bản HTML để đọc đầy đủ code + output: tải **notebook đã thực thi và còn output** từ phiên Kaggle đã lưu. Trên máy có Jupyter chạy:

```powershell
python -m jupyter nbconvert --to html --embed-images "c02_benchmark_xgb_cat.executed.ipynb" --output "C02-XGBoost-CatBoost"
```

Tên file đầu vào phải sửa đúng file bạn đã tải; mở kiểm tra có bảng 10 chuỗi, môi trường và kết quả. Lệnh này chuyển trực tiếp notebook thật theo [nbconvert](https://nbconvert.readthedocs.io/en/latest/usage.html), không dựng HTML thay cho output. Không xuất bản notebook nguồn chưa chạy rồi gọi đó là kết quả thực nghiệm.

Mẫu lời bàn giao sau khi có kết quả thật:

> C02 đã chạy trên Kaggle, run_id: …, 10 chuỗi, seed 42, CPU … threads, … lần lặp. Em gửi code, notebook đã lưu cùng bảng runtime/metadata. Train d_1–1885, Validation d_1886–1913; không chạy Test. Các kiểm tra leakage PASS; RMSSE có … chuỗi scale fallback. Nhờ Thủy Tiên kiểm tra cấu hình, phạm vi đo và xác nhận.

Không điền dấu “…” bằng số dự đoán. Khi chưa chạy trên Kaggle, trạng thái phù hợp là **code đã chuẩn bị, chờ chạy và xác nhận**, không tự đánh dấu C02 Xong. S05/S06 vẫn phải có bằng chứng riêng; C02 chỉ đọc ba CSV và không thay thế kiểm tra đọc đầy đủ năm CSV. Dùng bộ đọc chung ở mục 7.3 của [README nhóm](https://github.com/haiphamt/IS403.R11-M5-Competition/blob/main/README.md) trong bước S06 riêng.

## 8. Xử lý lỗi thường gặp

| Hiện tượng | Cách xử lý |
| --- | --- |
| Found 0 M5 directories / thiếu CSV | Add Input đúng cuộc thi; mở cây Input để kiểm tra đã có CSV, đặt DATA_DIR đúng thư mục |
| Found nhiều directory | Chỉ định DATA_DIR, không để code tự chọn một bộ tùy ý |
| Không import được xgboost/catboost/psutil | Cài ở ô riêng trước imports; restart và chạy lại từ đầu |
| `PackageNotFoundError: ... catboost` khi chạy mục 8 | Runtime đang dùng chưa có thông tin bản cài CatBoost. Chạy ô cài thư viện mục 1, rồi các mục 2–9 theo thứ tự; không dựa vào output cũ còn lưu trong notebook |
| Sai dtype/category | Dùng lại nguyên ô encoding, không tự astype category riêng cho hai tập |
| Duplicate price/calendar keys | Dừng, kiểm tra nguồn file/merge; không drop_duplicates tùy tiện để che lỗi |
| Merge không ra giá | Xác nhận các file cùng bộ M5, đúng khóa store/item/wm_yr_wk; giá thiếu trước khi bán là hợp lệ |
| CatBoost chạy lâu hơn dự kiến | Chờ log từng lần lặp; ghi số đo thật. Không thay số cây giữa chừng để so với run cũ |
| Session gián đoạn | Chạy lại từ đầu, dùng thư mục run mới; không dùng thư mục dở dang làm bằng chứng |

## 9. Nguồn kỹ thuật

- [Tracker nhóm](https://docs.google.com/spreadsheets/d/1Xr9J2CT5m4a8CbctRfB7m3GtCFCTrhPDsmmK6W62gxo/edit?gid=1176052156): C02, phụ thuộc S05/S06, deadline, approver.
- [features.yaml nhóm](https://github.com/haiphamt/IS403.R11-M5-Competition/blob/main/configs/features.yaml), [data.yaml nhóm](https://github.com/haiphamt/IS403.R11-M5-Competition/blob/main/configs/data.yaml): feature/split; đối chiếu 02/11/2026, SHA được ghi trong báo cáo đánh giá.
- [XGBoost categorical](https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html): category dtype, hist, mapping nhất quán.
- [CatBoostRegressor](https://catboost.ai/docs/en/concepts/python-reference_catboostregressor), [fit/cat_features](https://catboost.ai/docs/en/concepts/python-reference_catboostregressor_fit), [numeric missing values](https://catboost.ai/docs/en/concepts/algorithm-missing-values-processing): API/categorical/NaN.
- Bài báo M5 Accuracy tại `../../references/M5.pdf` và [kho Mcompetitions/M5-methods](https://github.com/Mcompetitions/M5-methods): bối cảnh RMSSE/WRMSSE, không phải kết quả C02 của nhóm.
