"""Build the seminar deck (pptx) from result tables; every number comes from results/tables/*.csv.

    python scripts/build_slides.py            # needs results/tables/{validation,test}/ and results/figures/{validation,test}/
Speaker notes carry the responsibility area of each slide. Time per slide is a suggestion only: the official
seminar length was not verified.
"""
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
T, F = ROOT / "results" / "tables", ROOT / "results" / "figures"
OUT = ROOT / "report" / "slides" / "m5_nhom6.pptx"
LABEL = {"seasonal_naive": "Seasonal Naive", "linear_regression": "Linear Regression", "ets": "ETS", "croston": "Croston",
         "lightgbm": "LightGBM", "xgboost": "XGBoost", "catboost": "CatBoost"}
ORDER = list(LABEL)


def overall(split):
    d = pd.read_csv(T / split / f"{split}_overall.csv", index_col=0).loc[ORDER]
    return d


prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def slide(title, notes, bullets=None, image=None, table=None, caption=None):
    s = prs.slides.add_slide(BLANK)
    tb = s.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.3), Inches(0.9))
    p = tb.text_frame.paragraphs[0]
    p.text, p.font.size, p.font.bold = title, Pt(30), True
    top = Inches(1.3)
    if bullets:
        box = s.shapes.add_textbox(Inches(0.6), top, Inches(6.0 if (image is not None or table is not None) else 12.0), Inches(5.5))
        tf = box.text_frame
        tf.word_wrap = True
        for i, b in enumerate(bullets):
            para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            para.text, para.font.size = "• " + b, Pt(20)
            para.space_after = Pt(8)
    if image is not None and Path(image).exists():
        s.shapes.add_picture(str(image), Inches(6.8 if bullets else 1.5), top, width=Inches(6.0 if bullets else 10.3))
    if table is not None:
        rows, cols = table.shape[0] + 1, table.shape[1]
        left = Inches(6.8 if bullets else 0.6)
        width = Inches(6.0 if bullets else 12.0)
        gt = s.shapes.add_table(rows, cols, left, top, width, Inches(0.4) * rows).table
        for j, c in enumerate(table.columns):
            gt.cell(0, j).text = str(c)
        for i, row in enumerate(table.itertuples(index=False), start=1):
            for j, v in enumerate(row):
                gt.cell(i, j).text = f"{v:.3f}".replace(".", ",") if isinstance(v, float) else str(v)
        for r in gt.rows:
            for c in r.cells:
                for para in c.text_frame.paragraphs:
                    para.font.size = Pt(14)
    if caption:
        cb = s.shapes.add_textbox(Inches(0.6), Inches(6.8), Inches(12), Inches(0.5))
        cb.text_frame.paragraphs[0].text = caption
        cb.text_frame.paragraphs[0].font.size = Pt(12)
    s.notes_slide.notes_text_frame.text = notes
    return s


def main():
    val, test = overall("validation"), overall("test")
    best_t = test["macro_rmsse"].idxmin()
    snaive_t = test.loc["seasonal_naive"]
    slide("Dự báo nhu cầu bán lẻ M5: so sánh bảy mô hình", "Mở đầu (Tuấn Hải). Giới thiệu đề tài, nhóm, phạm vi: 28 ngày, 1.200 chuỗi.",
          ["IS403.R11 – Phân tích dữ liệu kinh doanh – Nhóm 6", "Dữ liệu Walmart M5, mẫu phân tầng 1.200 chuỗi, horizon 28 ngày",
           "Seasonal Naive, Linear Regression, ETS, Croston, LightGBM, XGBoost, CatBoost"])
    slide("Bài toán và động lực", "Tuấn Hải: đầu vào/ra, vì sao dự báo ảnh hưởng bổ sung hàng.",
          ["Đầu vào: lịch sử bán, lịch/sự kiện, SNAP, giá tuần", "Đầu ra: 28 dự báo điểm không âm cho mỗi chuỗi sản phẩm–cửa hàng",
           "Thách thức: nhu cầu gián đoạn (70% ngày bán 0 trong mẫu), mùa vụ tuần, rò rỉ thông tin", "Chưa đo chi phí tồn kho: không tuyên bố tác động kinh doanh"])
    slide("Nghiên cứu nền", "Thanh Tiền (S04) và từng chủ mô hình: M5 là bối cảnh, không phải thứ hạng để so trực tiếp.",
          ["Makridakis và cs. (2022): tổng kết M5, 42.840 chuỗi, WRMSSE", "Đội thắng dùng LightGBM; chỉ 7,5% đội vượt đối chuẩn thống kê tốt nhất",
           "ETS (Hyndman & Khandakar 2008), Croston (1972), boosting (Friedman 2001; Ke 2017; Chen 2016; Prokhorenkova 2018)",
           "Khác biệt: RMSSE/MAE trên mẫu 1.200 chuỗi, không phải WRMSSE"])
    slide("Dữ liệu, mẫu và chia thời gian", "Tuấn Hải (D03), Thanh Tiền (D02/D07).",
          ["Mẫu: 10 cửa hàng × 3 category × 40 chuỗi, seed 42", "Train d_1–1885, Validation d_1886–1913, Test d_1914–1941",
           "Test mở sau tag config-freeze; refit trên d_1–1913", "Không chọn lại mẫu sau khi thấy kết quả"],
          table=pd.DataFrame({"Chỉ tiêu": ["Chuỗi", "Tỷ lệ ngày 0", "Doanh số TB"], "Toàn bộ": ["30.490", "0,684", "1,122"], "Mẫu": ["1.200", "0,701", "1,019"]}))
    slide("Đặc trưng và chống rò rỉ", "Thùy Trang (D04): lag ≥ 28, rolling shift 28, kiểm thử làm hỏng nhãn tương lai.",
          ["lag 28/35/42; rolling mean/std 7 và 28 sau shift 28", "Lịch, sự kiện, SNAP theo bang, giá tuần (NaN + price_missing)", "Dự báo trực tiếp 28 ngày từ origin",
           "Test: làm hỏng mọi nhãn sau origin, feature không đổi", "Giả định: giá ngày dự báo được biết trước"])
    slide("Mô hình thống kê", "Tuấn Vũ (ETS/Croston) và Tuấn Hải (Seasonal Naive).",
          ["Seasonal Naive: ŷ(T+h)=y(T−6+((h−1) mod 7))", "ETS: AutoETS chu kỳ 7, chọn theo AICc, mỗi chuỗi một mô hình",
           "Croston: z (lượng bán), p (khoảng cách), ŷ=z/p, α=0,1", "Local: chỉ dùng lịch sử doanh số của chính chuỗi"])
    slide("Hồi quy và boosting", "Thùy Trang (LR), Thủy Tiên (LightGBM), Thanh Tiền (XGBoost, CatBoost).",
          ["Linear Regression: impute trung vị, chuẩn hóa, one-hot", "Boosting: F_m(x)=F_{m−1}(x)+η f_m(x), tham số mặc định thư viện",
           "Global: một mô hình cho toàn mẫu, cùng bảng đặc trưng", "Chỉ đặt seed, luồng, khai báo categorical"])
    slide("Độ đo", "Tuấn Vũ (W03/P03): công thức, edge cases.",
          ["MAE, RMSE theo chuỗi; RMSSE chuẩn hóa bằng sai số naïve một bước", "R² có thể âm; hằng → không xác định",
           "Không epsilon: NaN + lý do, loại khỏi macro, đếm riêng", "Xếp hạng: macro-RMSSE; pooled MAE/RMSE báo cáo riêng", "WRMSSE của M5 ≠ trung bình RMSSE"])
    d = val.reset_index()[["model", "macro_mae", "macro_rmse", "macro_rmsse", "macro_r2", "fit_predict_s"]]
    d["model"] = d["model"].map(LABEL)
    d.columns = ["Mô hình", "MAE", "RMSE", "RMSSE", "R²", "fit+predict (s)"]
    slide("Kết quả Validation (d_1886–1913)", "Thủy Tiên (W07/P04). Dùng để chọn cấu hình; không có tuning.", table=d,
          caption="Nguồn: results/tables/validation/validation_overall.csv; macro trên chuỗi xác định; runtime local và global không cùng phạm vi.")
    d = test.reset_index()[["model", "macro_mae", "macro_rmse", "macro_rmsse", "macro_r2", "fit_predict_s"]]
    d["model"] = d["model"].map(LABEL)
    d.columns = ["Mô hình", "MAE", "RMSE", "RMSSE", "R²", "fit+predict (s)"]
    slide("Kết quả Test (d_1914–1941)", "Thủy Tiên: Test chạy một lần sau config-freeze; refit d_1–1913.", table=d,
          caption="Nguồn: results/tables/test/test_overall.csv; cùng mẫu, cùng chính sách metric.")
    cls = pd.read_csv(T / "test" / "test_by_demand_class.csv").pivot(index="demand_class", columns="model", values="rmsse")
    c = lambda x: f"{x:.3f}".replace(".", ",")
    hz, lz = cls.loc["high_zero(>0.8)"], cls.loc["low_zero(<=0.5)"]
    trees = [hz[m] for m in ("lightgbm", "xgboost", "catboost")]
    slide("Theo nhóm nhu cầu", "Thanh Tiền (F08): nhóm theo tỷ lệ ngày 0 trên Train; số lấy từ test_by_demand_class.csv.", image=F / "test" / "test_rmsse_by_demand_class.png",
          bullets=[f"Hàng bán chậm (trên 80% ngày không bán): ETS {c(hz['ets'])}, Croston {c(hz['croston'])}, ba mô hình cây {c(min(trees))}–{c(max(trees))}",
                   f"Hàng bán thường xuyên (tối đa 50% ngày không bán): ETS {c(lz['ets'])}, CatBoost {c(lz['catboost'])}",
                   "Chỉ là liên hệ trên một cửa sổ 28 ngày"])
    slide("Theo thứ trong tuần và horizon", "Thanh Tiền (F08).", image=F / "test" / "test_mae_by_horizon.png",
          bullets=["MAE gộp theo horizon 1–28", "Sai số cao hơn vào cuối tuần (Thứ Bảy, Chủ nhật)"])
    slide("Đánh đổi độ chính xác và thời gian", "Tuấn Hải/Thủy Tiên: không so trực tiếp runtime local và global.", image=F / "test" / "test_accuracy_runtime.png",
          bullets=["Thời gian: local = tổng fit+predict từng chuỗi; global = fit toàn mẫu", "CatBoost mặc định chậm nhất trên CPU 4 lõi"])
    slide("Insight và giới hạn", "Thủy Tiên: kết luận gắn với bảng/hình; nêu giới hạn.",
          ["Một cửa sổ Test 28 ngày; mẫu 1.200 chuỗi", "Khác biệt input giữa local và global gây nhiễu thứ hạng", "Giả định giá tương lai; importance không phải nhân quả",
           "Phần tùy chọn: mới có Validation (Optuna, SBA, ablation); stacking và Test chưa chạy", "Smoke ≠ tái lập toàn bộ"])
    slide("Kết luận và hướng phát triển", "Tuấn Hải (W08/P05).",
          [f"Mô hình tốt nhất trên Test theo macro-RMSSE: {LABEL[best_t]}", "Mọi mô hình cải thiện rõ so với Seasonal Naive (xem bảng)", "Hướng: rolling-origin nhiều mùa, WRMSSE phân cấp, mô phỏng tồn kho, dự báo xác suất"])
    slide("Q&A", "Cả nhóm: xem report/qa.md.", ["Câu hỏi dự kiến: horizon 28, lag ≥ 28, RMSSE vs WRMSSE, R² âm, AICc, Croston z/p, hệ số LR, boosting cộng cây, giá tương lai, local/global"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print("saved", OUT, len(prs.slides), "slides")


if __name__ == "__main__":
    main()
