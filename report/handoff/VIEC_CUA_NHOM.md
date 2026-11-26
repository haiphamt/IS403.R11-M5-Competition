# Việc của nhóm sau khi AI bàn giao (chỉ người thật làm được)

Trạng thái lúc bàn giao (15/11/2026): `main` @ `7e2543c`, 7 mô hình có Validation và Test thật. **Cập nhật lần cuối 18/11/2026:** `main` @ `7ab8149` (đã gộp PR #22, #25, #27; phần tùy chọn có kết quả Validation; đã có bản nháp handoff 7 mô hình). **18/11/2026 (chiều): nghiên cứu chính chuyển sang toàn bộ 30.490 chuỗi với 5 mô hình** (Seasonal Naive, ETS, LightGBM, Croston SBA, stacking lai ETS + SBA + LightGBM), chạy trên Kaggle, kết quả ở `results/full/`; báo cáo, slide, Q&A và README đã viết lại; Linear Regression, Croston cổ điển, XGBoost, CatBoost, LightGBM Tuned và bỏ giá/sự kiện chỉ còn trong pilot 1.200 chuỗi. Chủ mô hình của các mô hình bị bỏ (Thanh Tiền với XGBoost/CatBoost, Thùy Trang với Linear Regression) cần thống nhất lại phần việc: gợi ý Thanh Tiền phụ trách WRMSSE và phân tích theo nhóm, Thùy Trang phụ trách stacking lai. **18/11/2026 (sau đó):** Test phần tùy chọn đã chạy, các tag `config-freeze` và `config-freeze-v2` đã gỡ theo yêu cầu của nhóm (xem báo cáo mục 3.10 và giới hạn xiii). Những việc dưới đây AI **không** làm và **không** được đánh dấu Done thay nhóm. Phân công bám cột Owner/Reviewer trong checklist (tài khoản GitHub: Tuấn Hải = `haiphamt`, Tuấn Vũ = `Yuu2006`, Thanh Tiền = `tienlelttt`, Thủy Tiên = `NgKthy`, Thùy Trang = `ThorBietBay001`). Mọi đề xuất chia việc dưới đây là gợi ý của AI để nhóm chỉnh lại nếu cần.

## 1. Review và approve các PR

Tất cả PR #4–#20 đều do tài khoản `NgKthy` tạo và merge; GitHub không cho người tạo PR tự làm reviewer của chính PR, nên phần của Thủy Tiên phải làm theo cách khác (xem cuối mục). Branch protection **không** bắt buộc approval, nên việc này là để có bằng chứng review thật, không phải để mở khóa merge. Cách làm: mở PR, đọc diff và phần "Kiểm tra", chạy lại lệnh nếu cần, rồi bấm Review → Approve (hoặc Request changes kèm bình luận cụ thể).

| Người | PR được gán reviewer | Cần đối chiếu những gì |
|---|---|---|
| Tuấn Hải | #5, #9, #15, #16, #17, #19, #20 (và #4: D02 chưa được gán vì merge trước bước này, nên tự xem) | #5: ETS/Croston chạy theo chuỗi, z/p (M07); #9: README + Q&A (P06); #15: build_report (W09); #16: bảng Test của F03; #17: bản thảo; #19: ghi chú SHA; #20: PR bàn giao |
| Tuấn Vũ | #7, #8, #10, #11, #12, #13, #14, #16, #17, #20 | #7/#13/#14: runner và notebook (D06); #8: script phân tích (F08); #10: EDA (D07); #11: run Validation; #12: cấu hình freeze (T03); #16: run Test (F01, F08) |
| Thanh Tiền | #9, #18, #20 (nên xem thêm #11, #16 phần XGBoost/CatBoost vì chưa được gán) | #18: README (P05); phần importance XGBoost/CatBoost, hình M10 ở #15 |
| Thùy Trang | #15, #17, #20 | #15: khung slide (P01) và hình; #17: bản thảo mà Thùy Trang là người ghép (W09) |
| Thủy Tiên | không gán được trên GitHub (là người tạo PR) | Review theo kế hoạch D04/M03 (#6), M01/M02 (#5), M04–M06, F02/F04/F07 (#11, #16, #8): đọc diff, ghi nhận xét bằng comment trên PR hoặc gửi cho người được gán để họ approve thay; ghi rõ ai review gì vào mục 10 README |

PR #4 và #6 chưa có reviewer: cần người nhóm vào xem (gợi ý: Tuấn Hải xem #4; Thanh Tiền hoặc Tuấn Vũ xem #6, vì D04/M03 là phần của Thùy Trang nên không nhờ Thùy Trang).

*Cập nhật 18/11/2026:* Tuấn Hải đã để nhận xét rà soát (có hỗ trợ tự động) ở #4, #5, #9, #15, #16, #17, #19, #20; Thùy Trang đã review #15, #17, #20, #21; sửa theo các nhận xét này ở PR #28. Các nhận xét của Tuấn Hải nhắc tới PR #29 (Tóm tắt/Kết luận) và #30 (D03, P05, M09) chưa gộp vào `main` tại thời điểm này.

## 2. M09 – mỗi người chạy lại một run của người khác

Quy tắc: chạy theo README mục 11, **chỉ trên Validation hoặc tập con** (Test đã mở, không chạy lại để so điểm), ghi lỗi hoặc chỗ thiếu vào README mục 10 hoặc `report/handoff/`.

| Người chạy lại | Run của | Lệnh gợi ý (từ thư mục repo, đã cài `requirements.txt`) |
|---|---|---|
| Tuấn Hải | ETS và Croston (Tuấn Vũ, M01/M02) | `python scripts/run_experiment.py --split validation --models ets croston --max-series 50 --workers 2` |
| Tuấn Vũ | LightGBM (Thủy Tiên, M04) | `python scripts/run_experiment.py --split validation --models lightgbm --max-series 200` |
| Thanh Tiền | Linear Regression (Thùy Trang, M03) | `python scripts/run_experiment.py --split validation --models linear_regression --max-series 200` |
| Thủy Tiên | XGBoost và CatBoost (Thanh Tiền, M05/M06) | `python scripts/run_experiment.py --split validation --models xgboost catboost --max-series 50` |
| Thùy Trang | Seasonal Naive + gộp/phân tích (Tuấn Hải, D06) | `python scripts/run_experiment.py --split validation --models seasonal_naive --max-series 200` rồi `scripts/collect_runs.py`, `scripts/analyze_results.py` |

*Cập nhật 18/11/2026:* phần của Tuấn Hải (ETS/Croston, 50 chuỗi Validation) đã có biên bản ở PR #30 (chưa gộp): dự báo khớp run `val-20261115T072831Z-20b22f7` (lệch tuyệt đối tối đa 1,67 × 10⁻¹⁶ với ETS, 0 với Croston). Bốn người còn lại chưa ghi nhận.

Cần dữ liệu M5 trong `data/raw/` (hoặc `--data-dir`). Kết quả tập con khác kết quả 1.200 chuỗi; chỉ kiểm tra chạy được và số liệu hợp lý, không dùng thay bảng chính. Muốn chạy đủ 1.200 chuỗi thì dùng notebook `notebooks/02_m5_full_run.ipynb` / `02b_m5_catboost_run.ipynb` trên Kaggle.

## 3. Xem slide bằng PowerPoint

**Bộ slide đã dựng lại ngày 18/11/2026 cho nghiên cứu toàn bộ dữ liệu (5 mô hình), nên các lượt xem trước đây không còn áp dụng.** AI mới kiểm tra cấu trúc `report/slides/m5_nhom6.pptx` (17 slide, không shape nào vượt khung, đủ ghi chú), **chưa xem hình ảnh**. Mỗi người mở file, xem slide của mình (chữ có tràn, bảng có vừa, hình có đọc được không), rồi báo lỗi hoặc sửa và gửi PR. Phân công theo ghi chú người nói trong từng slide:

| Người | Slide |
|---|---|
| Tuấn Hải | 1, 2, 4, 6 (Seasonal Naive), 14, 16 |
| Tuấn Vũ | 6 (ETS, Croston SBA), 8 |
| Thanh Tiền | 3, 4, 11 (WRMSSE), 12, 13 (không còn XGBoost/CatBoost, nên phụ trách phần WRMSSE và phân tích theo nhóm) |
| Thủy Tiên | 7 (LightGBM), 9, 10, 14, 15 |
| Thùy Trang | 5, 7 (stacking lai) |

Thời lượng seminar chính thức **chưa được xác minh**: cần hỏi giảng viên để cắt bớt slide cho vừa giờ.

## 4. Điền họ tên và MSSV trong báo cáo

AI không đoán thông tin cá nhân. Từng người gửi họ tên đầy đủ và MSSV cho Thùy Trang (người ghép báo cáo, W09); Thùy Trang sửa dòng thứ 3 của `report/draft_report.md` (đang ghi "chờ đối chiếu danh sách chính thức") và cả trang bìa khi dựng bản Word.

## 5. Đọc lại toàn văn 6 nguồn "chưa đọc lại toàn văn"

AI chỉ ghi thông tin thư mục từ hiểu biết sẵn có. Đọc bài gốc, đối chiếu tác giả, năm, tạp chí/hội nghị, trang, DOI, rồi sửa mục Tài liệu tham khảo (và chỗ trích dẫn nếu sai):

| Người | Nguồn |
|---|---|
| Tuấn Vũ | [2] Hyndman & Khandakar (2008); [3] Croston (1972); [4] Syntetos & Boylan (2005) |
| Thủy Tiên | [5] Friedman (2001) |
| Thanh Tiền | [7] Chen & Guestrin (2016); [8] Prokhorenkova và cs. (2018) |

Ba nguồn còn lại cũng chỉ được đối chiếu qua tóm tắt/abstract, nên nên đọc luôn: [1] Makridakis và cs. (2022) (Thanh Tiền, vì là người viết S04), [6] Ke và cs. (2017) và [9] Bandara và cs. (2022) (Thủy Tiên).

## 6. W10, W11, P07, P08 (người thật mới làm được, AI không được đánh dấu Done)

- **W10 – đọc chéo báo cáo, đối chiếu số liệu với `results/tables/`** (Tất cả, người phụ trách Tuấn Hải; hạn kế hoạch 23/11). Chia mục gợi ý: Tuấn Hải: Tóm tắt, mục 1, mục 5; Tuấn Vũ: mục 3.5–3.6 và 4.1; Thanh Tiền: mục 2, các hình, bảng theo nhóm; Thủy Tiên: bảng Validation/Test và phần Q1–Q3 của 4.2; Thùy Trang: mục 3.1–3.4, 3.7–3.10, tài liệu tham khảo, định dạng. Sửa văn bản bằng PR; bảng sinh lại bằng `python scripts/build_report.py` (không sửa số tay). Sản phẩm: danh sách lỗi đã sửa.
- **W11 – nộp báo cáo (file Word)**: Thùy Trang (hạn kế hoạch 24/11), chỉ sau W10; người dùng tự định dạng, AI không nộp bài.
- **P07 – tập dượt lần 1, bấm giờ** (Tất cả, Tuấn Hải điều phối; 23/11) và **P08 – tập dượt lần 2, một người đóng vai thầy hỏi** (24/11; gợi ý Thùy Trang đóng vai thầy, dùng `report/qa.md` làm bộ câu hỏi): cả nhóm cùng tham gia; điều chỉnh thời lượng theo giờ giảng viên quy định. Nếu seminar sớm hơn thì kéo P07/P08 lên.
- Ngoài ra còn **F09** (mỗi chủ mô hình viết 3–5 dòng giải thích hành vi mô hình vào `report/handoff/<mô hình>.md`) và **P06** (đọc lại phần hỏi đáp của nhau): đã có bản nháp handoff 7 mô hình trong `report/handoff/` (thêm ở PR #23); mỗi chủ mô hình đọc, sửa và xác nhận file của mình (phần giải thích cũng nằm trong mục 4.2 của báo cáo).

## 7. Cập nhật Jira và Google Sheet

AI không đụng vào Jira hoặc Sheet. Mỗi người tự cập nhật task của mình dựa trên bằng chứng ai cũng xem được: danh sách PR trên GitHub, `results/runs/*/environment.json` (run_id, `git_sha`), `results/pre_rewrite_sha_map.csv` và `configs/freeze.yaml`; chỉ chuyển sang Done khi đã tự đối chiếu, không chuyển chỉ vì AI đã làm. Task chung cho từng người:

| Người | Task riêng cần cập nhật (theo bảng phân công) | Ghi chú trạng thái |
|---|---|---|
| Tuấn Hải | S01, S03, S05, C04, C05, D01, D03, D06, T03, F01, F10, W04, W08, P01, P05 | F10 là smoke, không phải tái lập đầy đủ; W08 (Tóm tắt/Kết luận) đã soạn |
| Tuấn Vũ | C01, D05, M01, M02, M07, M08, F03, F07, W03, P03, X01 | X01 (Croston SBA): *measured on M5* ở Validation (RMSSE 0,7425) và Test (0,7619; tốt hơn Croston 0,7635, kém ETS 0,7520) |
| Thanh Tiền | S04, C02, D02, D07, M05, M06, M10, F04, F08 | |
| Thủy Tiên | C03, M04, T01, F05, W07, P04, X02 | T01 (Optuna) và X02 (bỏ giá/sự kiện): *measured on M5* ở Validation (Tuned 0,7447; bỏ giá/sự kiện 0,7479) và Test (Tuned 0,7700, không hơn Default 0,7680; bỏ giá/sự kiện 0,7662); F05 trên Test chỉ có LightGBM Default |
| Thùy Trang | C06, D04, M03, T02, F02, F06, W02, W09, W11 | T02 và F06 (stacking): *measured on M5* ở Validation (macro-RMSSE 0,7519, không tốt hơn LightGBM Tuned; RMSE gộp thấp nhất) và Test (0,7689, ngang LightGBM; RMSE gộp 1,928 thấp nhất trong các mô hình global, cao hơn ETS 1,913); W11 chưa làm |

Task chung (S02, S06, M09, W01, W05, W10, P06, P07, P08, F09): mỗi người cập nhật phần của mình. Trạng thái cần ghi rõ: *implemented* (mã đã có), *tested on fixtures*, *measured on M5* (có run_id), *AI reviewed*, hoặc *human approved* (chỉ khi người thật đã duyệt).
