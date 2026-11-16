# C02 — Ghi nhận kiểm tra kỹ thuật

Ngày cập nhật: **13/11/2026**, Asia/Bangkok. Tài liệu ghi riêng QA mã nguồn, benchmark M5 thật và các giới hạn của mỗi loại bằng chứng.

## Trạng thái mới nhất — lần chạy notebook sau khi sửa

Notebook Colab được chạy xong với run ID `C02_20261113T093749_327948+0700`: 10 chuỗi, Train 18.850 dòng, Validation 280 dòng; đủ 6 lượt fit/predict, không có lỗi và các kiểm tra rò rỉ cùng lag/rolling đều PASS. Bảng tổng hợp cho MAE: XGBoost 0.552113, CatBoost 0.542314; median fit: XGBoost 0.894220 giây, CatBoost 40.806018 giây. Đây là kết quả trên mẫu thử 10 chuỗi của C02, không đại diện toàn bộ dữ liệu M5.

Notebook được mở qua URL Colab nhưng metadata lần chạy báo `platform=Kaggle`; chưa xác minh vì sao nhãn này xuất hiện. Cần kiểm tra nhãn nền tảng trước khi dùng như bằng chứng xác định nơi chạy. Thủy Tiên chưa xác nhận kết quả. Toàn bộ test suite chưa được chạy lại sau khi thêm regression test cho thông báo thiếu thư viện; các kết quả test cũ bên dưới chỉ áp dụng cho phiên bản/fixture ghi tại thời điểm đó.

Ngày 14/11/2026 đã xuất [HTML đầy đủ code và output](../../results/c02/C02_20261113T093749_327948+0700/c02_benchmark_xgb_cat.html) trực tiếp từ notebook đã thực thi. Phần chạy theo yêu cầu C02 (code hai mô hình, 10 chuỗi, bảng thời gian) đã có bằng chứng; Test/mẫu 1.200 chuỗi không phải điều kiện hoàn tất phần chạy của C02. Việc người xác nhận duyệt là bước nghiệm thu riêng.

## QA lịch sử ngày 02/11/2026

| Kiểm tra | Kết quả |
| --- | --- |
| Parse/compile Python module và tất cả code cell | PASS |
| Notebook nbformat 4.5, 21 cell (10 code + 11 Markdown) | PASS |
| Notebook nguồn không có output hoặc execution count giả | PASS |
| 10 unit/integration tests | PASS, chạy bằng unittest; dữ liệu tổng hợp |
| Chạy tuần tự đủ 10 code cell của notebook trong IPython | PASS, 0 lỗi; dữ liệu tổng hợp |
| Hai estimator XGBoost/CatBoost fit và predict thật | PASS, 10 chuỗi fixture; 18.850 dòng Train, 280 dòng Validation |
| Feature/label perturbation, lag/rolling ở biên, NaN, category | PASS |
| Pipeline ghi đủ 6 tệp và đọc ngược CSV/JSON để kiểm tra | PASS |
| SHA256 trong notebook/metadata khớp script | PASS |
| Xuất HTML trực tiếp từ notebook QA đã thực thi bằng nbconvert | PASS; chỉ lưu trong tmp, không dùng làm bằng chứng M5 |
| CLI --help | PASS |
| Benchmark M5 thật, 10 chuỗi, run `C02_20261113T093749_327948+0700` | **PASS; 6 lượt, audit PASS; nhãn platform còn cần xác minh** |
| Benchmark 1.200 chuỗi / peak RAM / Test | **KHÔNG THỰC HIỆN** |

Mười test gồm: lấy mẫu deterministic/đủ cửa hàng-category/không phụ thuộc thứ tự dòng; giữ NaN giá và lag; perturbation và công thức biên; duplicate price key bị chặn; mapping category mới; RMSSE tính tay/bỏ zero đầu; metric hữu hạn cho chuỗi suy biến; đường dẫn sai không fallback âm thầm; chunk loader không đọc Test; native category/NaN fit/predict ở cả hai estimator.

Đã sửa và kiểm lại việc ghi `missing=np.nan` của XGBoost: metadata JSON biểu diễn mặc định tham số này bằng chuỗi `NaN`, không ghi số JSON không hợp lệ. Đây là tham số missing của mô hình, không phải điểm số bị NaN. Số thread áp dụng cả fit và predict, bao gồm `thread_count` truyền rõ vào CatBoost.predict.

## Môi trường QA local ngày 02/11/2026

Windows 11, Python **3.13.15**, CPU; mô hình dùng 2 thread. Package:

| Package | Version |
| --- | --- |
| numpy | 2.5.4 |
| pandas | 3.1.0 |
| scikit-learn | 1.9.1 |
| xgboost | 3.4.1 |
| catboost | 1.2.10 |
| psutil | 7.2.2 |

Đây là môi trường local dùng kiểm tra, **không phải cấu hình Kaggle của Thanh Tiền**. Notebook ghi lại versions và CPU/RAM mới tại mỗi run; không chép cấu hình local vào kết quả Kaggle. Tệp benchmarks/c02/requirements.txt ghi API tối thiểu để cài đặt; chưa khẳng định đã thử mọi tổ hợp phiên bản được cho phép.

SHA256 của script tại lần kiểm tra ngày 02/11 (bản lịch sử, trước sửa bộ đọc Drive):

```text
39cb35c9cdf5e3e95a34da2a2d4f0e8cb049042653feb0306e2246de9c79d292
```

## Phương pháp và giới hạn của kiểm tra notebook

Thử kernel Jupyter riêng trên máy này không hoàn tất bước kết nối trong thời gian kiểm tra và đã được dừng. QA cuối dùng **IPython chạy thật tuần tự từng cell**, bắt stdout/stderr và rich display thực tế, ghi notebook đã thực thi rồi dùng HTMLExporter của nbconvert xuất trực tiếp. Không giả lập output, không thay mô hình bằng stub. Đây là kiểm tra logic/thứ tự cell và export; chưa xác minh vòng đời kernel hay giao diện Kaggle.

Để QA không bị hiểu thành kết quả M5, bản sao notebook kiểm tra có banner **SYNTHETIC INTEGRATION QA ONLY — NOT M5 RESULTS**, chạy một lần lặp trên fixture và lưu dữ liệu/output trong thư mục tạm. Notebook nguồn bàn giao vẫn giữ **3 lần lặp**, không có output, không chứa fixture và không tự fallback sang dữ liệu tổng hợp.

Các CSV tổng hợp, notebook/HTML QA, bản sao cũ và môi trường ảo đã được xóa theo yêu cầu dọn tệp tạm ngày 02/11/2026. Kết quả kiểm tra được ghi lại trong tài liệu này; mã kiểm thử vẫn được giữ để chạy lại. Các file QA tạm không thuộc hồ sơ thực nghiệm C02 gửi người xác nhận.

## Lệnh chạy lại kiểm thử

Từ thư mục môn học, dùng Python đã cài dependencies:

```powershell
python -m unittest discover -s "Đồ án/tests" -v
```

Notebook được chỉnh trực tiếp để có thể chạy và học theo từng bước. Script build notebook đã được gỡ bỏ; nếu sửa mã nguồn và notebook, cần kiểm tra hai bản đồng bộ trước khi bàn giao.

## Việc còn lại sau lần chạy mới nhất

Kiểm tra lại thông tin nền tảng (Colab hay Kaggle) từ runtime và lưu notebook cùng output, bảng, metadata thuộc run ID trên. Sau đó gửi link notebook và các tệp kết quả cho Thủy Tiên để xác nhận. Mẫu chỉ có 10 chuỗi, không dùng để kết luận về toàn bộ 30.490 chuỗi; chưa chạy Test hoặc mẫu 1.200 chuỗi.

## Bổ sung ngày 04/11/2026 — đọc CSV Mcompetitions Drive

Đã tải nguyên bản ba CSV từ Dataset Drive được lưu trong S04_sources.md vào
`data/raw/m5_drive/`. Sales có 30.490 dòng/1.918 cột (thiếu `id`); calendar có
1.969 dòng/13 cột (thiếu `d`); prices có 6.841.121 dòng/4 cột.
Đây là khác biệt schema với bản Kaggle, không phải thiếu giá trị doanh số.
Bộ đọc mới nhận cả hai schema, bổ sung khóa trong bộ nhớ và ghi metadata;
không thay đổi CSV trên đĩa. Notebook nguồn được build lại cùng mã nguồn mới.

11 kiểm tra không huấn luyện PASS, gồm đối chiếu cùng dữ liệu ở hai schema
cho ra cùng mẫu/đặc trưng, CSV không đổi, mã ngày dựa trên ngày thực thay vì
thứ tự dòng, và các kiểm tra leakage/metric/encoding trước đó.
Lần này không chạy lại kiểm tra fit/predict của hai estimator; kết quả fit/predict
ngày 02/11 ở trên là bằng chứng lịch sử của phiên bản trước.

Đã chạy bộ đọc, tạo feature và audit trên **CSV M5 Drive thật**: 10 chuỗi,
18.850 dòng Train, 280 dòng Validation, các kiểm tra leakage PASS; hash cả
ba CSV không đổi trước/sau. Bằng chứng lưu ở `local/c02/C02_DRIVE_INPUT_CHECK.json` (chỉ giữ local).
Môi trường kiểm tra đầu vào: NumPy 2.5.4, pandas 3.1.0, Windows/Python 3.13.
Không huấn luyện mô hình, không ghi số đo benchmark Kaggle, không chạy Test.

## Rà soát hồ sơ khi clone repo ngày 04/11/2026

- SHA256 script và metadata notebook hiện tại khớp nhau: `bbd4154a42722791d7991999aa1db01a293120884e82920e36b4569f3cb24abc`. Các code cell compile được, không có output hoặc execution count điền sẵn.
- Ba CSV gốc vẫn khớp toàn bộ hash trong `local/c02/C02_DRIVE_INPUT_CHECK.json` (chỉ giữ local).
- Tuy nhiên, `source_sha256` trong JSON bằng chứng là `c10939face296bce93aac262fb205df1c3a546eb1158f94edca35d37f54e4b6c`, khác mã hiện tại. Vì vậy JSON là bằng chứng của phiên bản trước; không coi các kiểm tra feature trong đó là bằng chứng mới cho mã hiện tại. Không sửa hash lịch sử để làm chúng khớp giả.
- Lần sắp xếp này không đổi script/logic notebook, không chạy lại huấn luyện hoặc kiểm thử mô hình. Cần chạy lại kiểm tra đầu vào/leakage và benchmark bằng notebook hiện tại trên Colab/Kaggle trước khi nộp C02.


## 13/11/2026 — Kiểm thử lại bản chuẩn bị cho người dùng tự chạy cloud

- Bản nguồn SHA256: `55c673e76dd1f27cec2829c2ea78d384bcdfb23d548f2c2f88682bed70fe7843`. Notebook được build lại từ đúng bản này, schema nbformat hợp lệ, 10 code cell compile được, chưa có output/execution count M5.
- **PASS: 13/13 unittest**, 31,808 giây cho toàn suite. Đây là thời gian chạy test tổng hợp, không phải runtime benchmark M5. Hai estimator thực sự fit/predict trên fixture có categorical và numeric NaN; 280 dự báo mỗi model đều hữu hạn. Các test khác kiểm tra selection/seed, Train/Validation, lag/rolling, perturbation chống leakage, metric thủ công, loader Drive/Kaggle, khóa trùng và nhận diện Colab.
- Môi trường local tạm: Python 3.13.15; numpy 2.5.4, pandas 3.1.1, scikit-learn 1.9.1, xgboost 3.4.1, catboost 1.2.10, psutil 7.2.2. Lệnh tái lập sau khi cài requirements: `python -B -m unittest discover -s tests -p test_c02_benchmark.py -v`.
- Sửa fill categorical qua object trước fillna để tránh cảnh báo dtype pandas; metadata nhận diện Colab; notebook chọn thư mục output theo nền tảng. Guide bổ sung đủ 5 bước Colab.
- Theo yêu cầu mới nhất, người dùng tự chạy cloud; không chạy benchmark M5 local hoặc cloud trong đợt này. Chưa có bảng thời gian M5 thật; Thủy Tiên chưa xác nhận. Môi trường tạm được dọn sau kiểm thử.


### Chuẩn hóa nội dung notebook ngày 13/11/2026

Theo yêu cầu người dùng, rút gọn Markdown và output cho bản chạy nộp: giữ mục tiêu, split, phương pháp và bảng kết quả; bỏ thông tin điều phối task, bàn giao, hướng dẫn xuất/tải file, lời nhắc lặp và khối JSON tham số dài. Thêm `show_summary` để tránh in lặp bảng ở notebook; CLI vẫn mặc định hiện bảng. Trạng thái `completed` trong metadata chỉ mô tả lần chạy đã kết thúc.

SHA256 nguồn mới: `feb9926834985c5831e54f746107cd7c4537e20a49994d722693a70bed2d41d3`. Kiểm tra tĩnh PASS: mọi cell compile được, các hàm/lớp nhúng khớp AST của script, hash khớp, notebook nguồn không có output/execution count. Các phép tính, cấu hình mô hình, chọn mẫu và audit giữ nguyên; không chạy lại model cho thay đổi trình bày này. Bộ 13 tests ở trên áp dụng bản trước khi thêm tùy chọn hiển thị bảng.


### C02 — cấu hình Colab tải tệp từ máy, 13/11/2026

Theo lựa chọn người dùng, ô cấu hình notebook dùng `google.colab.files.upload(target_dir=DATA_DIR)` để chọn ba CSV sẵn trong `data/raw/m5_drive/` trên máy, đưa vào `/content/m5`. Có đủ tệp thì dùng lại; thiếu sau upload thì báo lỗi trước benchmark. Output nằm tại `/content/c02_results`. Không mount Drive, không tải dữ liệu mới hoặc chạy model trong đợt sửa này. Guide đồng bộ.

Kiểm tra bằng stub API upload (không phải tương tác Colab thật): PASS luồng tải, đường dẫn đích, dùng lại file không gọi upload, báo lỗi thiếu file; compile các cell và source hash PASS. Fixture tạm tự xóa. Người dùng sẽ tự import notebook và CSV, chạy cloud và lưu output.


### C02 notebook — bỏ câu hướng dẫn nạp dữ liệu, sửa lỗi CatBoost, 13/11/2026

Bỏ đoạn Markdown mô tả thao tác tải CSV khỏi nội dung notebook nộp; mã cell vẫn mở hộp chọn ba CSV từ máy khi `/content/m5` còn thiếu dữ liệu. Sửa cell thư viện: cài các package thiếu (bao gồm CatBoost) trên Colab, thay vì dừng bằng `RuntimeError` như ảnh người dùng cung cấp. Kiểm tra compile toàn bộ code cell, xác nhận câu hướng dẫn không còn, chưa có output giả: PASS. Không chạy benchmark M5 trong lượt này.

### 13/11/2026 — Lỗi CatBoost và notebook thực hiện từng bước

- Đọc trực tiếp notebook Drive `1tqTUeT65H1R0DRFpnUPqLs0yAENapWXD`: lỗi tại `environment()` là `PackageNotFoundError: No package metadata was found for catboost`, trước bước fit. Ô thư viện lưu output từ lần chạy trước; output cũ không phản ánh các gói hiện có trong runtime.
- Theo yêu cầu làm tuần tự, ô 1 dùng `%pip install` trực tiếp, không có bộ tự dò/cài hoặc hàm setup. Mục 3 thực sự đọc dữ liệu/hiện mẫu; mục 4 tạo đặc trưng, kiểm tra và chia tập; mục 7 ghi môi trường; mục 8 chạy vòng fit/predict; mục 9 tổng hợp kết quả. Chú thích/docstring mã nguồn và notebook bằng tiếng Việt.
- Mã nguồn mới SHA256 `6dc6e03110a8806bcf84af5582fe226368b17a0d2e65cc7f5fbe980885c6b36f`. Kiểm tra tĩnh PASS: cú pháp 9 ô mã (chuyển `%pip` sang lời gọi IPython khi kiểm tra), hàm nhúng khớp AST nguồn, phép tính đặc trưng/độ đo/mô hình và toàn bộ vòng fit/predict khớp bản cũ. Kiểm tra mô phỏng thiếu CatBoost và nhiều gói PASS: dừng với tên gói và hướng dẫn chạy mục 1. Có thêm regression test tương ứng trong suite hiện có; không chạy lại toàn suite vì Python local hiện không có các thư viện mô hình.
- Notebook cộng thời gian xử lý từng bước vào `measured_work_s_all_repeats`; không tính thời gian dừng giữa các ô. Cột fit/predict giữ nguyên cách đo. Script dòng lệnh vẫn đo pipeline liên tục bằng `pipeline_wall_s_all_repeats`.
- Đã cập nhật file Drive, fetch đọc lại khớp notebook local: SHA256 `8532bfc323b19db80ace3fac94feafeaaa21737e26efcd2bdbed647b7ed2294c`, 20 ô. Lúc cập nhật chưa có output; lần chạy mới hơn được ghi ở mục đầu tài liệu.
- Trong lúc sửa, bản cloud cũ vừa được người dùng chạy xong: `C02_20261113T092551_321135+0700`, 10 chuỗi, 18.850 Train, 280 Validation, đủ 6 lượt fit/predict, audit PASS. Giữ nguyên bản đã thực thi tại `local/c02/C02_20261113T092551_321135+0700.ipynb`, SHA256 `e2e0aedc31ec43d57ae8e332d286f7c6bf1c59a373aa66fd66aa466d34e5af1b`. Kết quả này thuộc mã cũ `feb992...`, không gắn sang notebook sửa mới. Output báo nhãn môi trường “Kaggle” trong file mở qua Colab; chưa xác minh nguyên nhân nhãn này, không tự dùng làm bằng chứng nền tảng Kaggle riêng.

### 13/11/2026 — Kết quả chạy sau khi sửa

- Người dùng chạy notebook đã sửa, run ID `C02_20261113T093749_327948+0700`; mục 8 hoàn tất đủ ba lần cho mỗi mô hình, mục 9 hiện đủ bảng tổng hợp, trials và 10 chuỗi. Không có traceback.
- Median fit: XGBoost **0.894220 s** (min 0.452500, max 2.365630); CatBoost **40.806018 s** (min 39.797000, max 47.812040). Median MAE: XGBoost **0.552113**, CatBoost **0.542314**; median RMSE lần lượt **0.891002**, **0.848253**; RMSSE_mean **0.650913**, **0.639600**. Đây là kết quả của đúng mẫu 10 chuỗi, không phải kết luận tổng quát về mô hình.
- Metadata ghi Python 3.13.18, CPU Xeon 2.20GHz, 2 luồng; các thư viện: NumPy 2.4.6, pandas 2.2.3, scikit-learn 1.6.1, XGBoost 3.4.1, CatBoost 1.2.10, psutil 5.9.5. Metadata nhận diện nền tảng là “Kaggle” trong khi notebook được mở từ Colab; mâu thuẫn này cần được kiểm tra trước khi dùng kết quả để xác định nền tảng.
- Chưa xác minh nơi lưu bền vững toàn bộ sáu tệp kết quả được sinh trong `/content/c02_results/C02_20261113T093749_327948+0700`; notebook đang mở có output của lần chạy. Chưa có xác nhận của Thủy Tiên và chưa chạy lại toàn bộ unit test suite sau thay đổi mới nhất.


### Tổ chức thư mục trước bàn giao ngày 13/11/2026

Guide, hồ sơ kiểm chứng và dependencies nằm trong `benchmarks/c02/`. Hồ sơ cũ và notebook bằng chứng trước lần sửa giữ tại `local/c02/`, không thuộc tập commit. Notebook nguồn được sửa trực tiếp; không còn script tạo notebook tự động. Nội dung bàn giao được chuẩn bị cho mô tả PR và bình luận Jira, không commit tệp HANDOFF.

Mã local sau khi sửa đường dẫn có SHA256 `f4d5052be495762a7116b48fb9b3eb032e73d8d693282f085563a0a437b452fa`. Thay đổi chỉ ở đường dẫn tài liệu/cài dependencies và metadata hash; không thay công thức, split, seed, mô hình hoặc cách đo. Nguồn của lần chạy cloud vẫn có SHA256 `6dc6e03110a8806bcf84af5582fe226368b17a0d2e65cc7f5fbe980885c6b36f`; không gắn output cũ vào notebook nguồn mới.

Đã đọc lại notebook Drive sau khi người dùng lưu kết quả, không có traceback. Bản nguyên byte được giữ ở `local/c02/C02_20261113T093749_327948+0700.ipynb`, SHA256 `f73e53ce98aa00e03e59472ce47f11843a798a2dc7843b386b9bc499c4cb30b7`. Bảng kết quả chính được ghi ở phần đầu tài liệu; nội dung bàn giao dùng trong mô tả PR và bình luận Jira. Chưa tìm thấy CSV/JSON gốc riêng trên Drive; không dựng lại chúng từ bảng đã làm tròn rồi coi là file gốc.

### Xuất HTML ngày 14/11/2026

- Nguồn: notebook có output được giữ nguyên byte từ file Drive, hash `f73e53ce98aa00e03e59472ce47f11843a798a2dc7843b386b9bc499c4cb30b7`. Metadata Drive được đối chiếu vẫn cùng phiên bản lưu ngày 13/11.
- Công cụ: nbconvert 7.17.2, `--to html --embed-images`; không dùng `--execute`, không chạy lại model, không dựng HTML thay cho output.
- Tệp: `results/c02/C02_20261113T093749_327948+0700/c02_benchmark_xgb_cat.html`, SHA256 `763329b4d619ea687c36c3a403e8ae4d3dfab046e9ddb9f00ff12d807d1287e1`.
- Kiểm tra: đủ 9 ô code, 5 bảng DataFrame có lần lượt 10/2/6/10/6 dòng, khớp bảng trong notebook gốc; run ID và số đo khớp. Notebook nguồn không có output lỗi và không bị chỉnh sửa khi xuất.
- CSV/JSON riêng trong runtime chưa được lấy. Chúng là bằng chứng bổ trợ để giữ đầy đủ số gốc/metadata, không phải yêu cầu chạy thêm Test hoặc tăng mẫu C02. HTML hiện giữ code và các bảng kết quả cần bàn giao. Nếu runtime vẫn còn thư mục kết quả, tải từng file gốc để lưu bền vững.
