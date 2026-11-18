# M09 — Đối chiếu ETS và Croston trên Validation

Ngày 16/11/2026, phần đối chiếu của Tuấn Hải chạy lại hai mô hình M01/M02 của Tuấn Vũ trên 50 chuỗi đầu trong `data/sample_series_ids.csv`. Dùng lịch sử đến `d_1885` và dự báo `d_1886`–`d_1913` trên Windows 11, Python 3.12.14, StatsForecast 2.1.1 và NumPy 2.1.3.

- Run đối chiếu `m09-haiphamt-val-20261116-71ef7ab`, mã nguồn `71ef7abe0bf415a87757d16f5ec678a7218df1df`.
- Run tham chiếu `val-20261115T072831Z-20b22f7`.
- Đủ 2.800 dự báo, mỗi mô hình 1.400. Ghép theo mô hình, mã chuỗi và ngày. Không có lượt dùng mô hình dự phòng. Chỉ số tính lại từ Train và Validation của D03 khớp với tệp `metrics_summary.csv` của lần chạy.

| Mô hình | Macro-MAE | Macro-RMSSE | Chênh lệch dự báo tuyệt đối lớn nhất so với run tham chiếu |
| --- | ---: | ---: | ---: |
| ETS | 1,232587 | 0,711848 | 1,67 × 10⁻¹⁶ |
| Croston | 1,190892 | 0,716015 | 0 |

Lệnh chạy từ bản sao repo có dữ liệu trong `data/raw/`:

```bash
python scripts/run_experiment.py --split validation --models ets croston --max-series 50 --workers 2
```

Nhật ký môi trường ghi thêm `--data-dir`, `--out` và `--run-id` cho đường dẫn dữ liệu và nơi lưu kết quả trên máy. Hai mô hình dùng cùng tập con của lần chạy tham chiếu để kiểm tra khả năng tái lập. Các chỉ số ở đây tính riêng cho 50 chuỗi.
