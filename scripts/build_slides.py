"""Build the seminar deck (pptx) of the full-data study (all 30,490 series, 5 models) from result tables.

Numbers about results come from results/full/tables/*.csv and the pilot runs (results/runs); typed by hand: literature facts on
slide 3 (Makridakis et al. 2022) and model settings on slides 5-8 (copied from configs/ and src/m5/).

    python scripts/build_slides.py      # needs results/full/tables/{validation,test}/ and results/full/figures/{validation,test}/
Speaker notes carry the responsibility area of each slide. Time per slide is a suggestion only: the official seminar length was not verified.
"""
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt

from build_report import L as LABEL, ORDER, PILOT_RUNS, num  # same labels, model order and number format as the report

ROOT = Path(__file__).resolve().parents[1]
T, F = ROOT / "results" / "full" / "tables", ROOT / "results" / "full" / "figures"
OUT = ROOT / "report" / "slides" / "m5_nhom6.pptx"

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def overall(split):
    return pd.read_csv(T / split / f"{split}_overall.csv", index_col=0).loc[ORDER]


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
            para.text, para.font.size = "• " + b, Pt(18)
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
                # time columns "(s)" use 1 decimal like the report; other floats 3 decimals
                gt.cell(i, j).text = num(f"{v:.1f}" if "(s)" in str(table.columns[j]) else f"{v:.3f}") if isinstance(v, float) else str(v)
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
    c = lambda x: f"{x:.3f}".replace(".", ",")
    sg = lambda x: num(f"{x:+.4f}")
    others = [m for m in ORDER if m != "seasonal_naive"]
    wr = {sp: pd.read_csv(T / sp / f"{sp}_wrmsse.csv").set_index("model") for sp in ("validation", "test")}
    test = overall("test")
    best_t, best_w = test["macro_rmsse"].idxmin(), wr["test"].loc[ORDER, "wrmsse"].idxmin()
    ci = pd.read_csv(T / "test" / "test_bootstrap_ci.csv").set_index("model")
    pilot = pd.concat([pd.read_csv(p / "metrics_summary.csv") for p in sorted(PILOT_RUNS.glob("tes-*"))]).set_index("model")["macro_rmsse"]
    slide("Dự báo nhu cầu bán lẻ M5: năm cách tiếp cận trên toàn bộ dữ liệu", "Mở đầu (Tuấn Hải). Đề tài, nhóm, phạm vi: 30.490 chuỗi, 28 ngày.",
          ["IS403.R11 – Phân tích dữ liệu kinh doanh – Nhóm 6", "Walmart M5: toàn bộ 30.490 chuỗi sản phẩm–cửa hàng, horizon 28 ngày",
           "Seasonal Naive, ETS, Croston SBA, LightGBM, Stacking lai"])
    slide("Bài toán và động lực", "Tuấn Hải: đầu vào/ra, vì sao dự báo ảnh hưởng bổ sung hàng.",
          ["Đầu vào: lịch sử bán, lịch/sự kiện, SNAP, giá tuần", "Đầu ra: 28 dự báo điểm không âm cho mỗi chuỗi sản phẩm–cửa hàng",
           "Thách thức: nhu cầu gián đoạn (39% số chuỗi có trên 80% ngày bán 0), mùa vụ tuần, rò rỉ thông tin",
           "Chưa đo chi phí tồn kho: không tuyên bố tác động kinh doanh"])
    slide("Nghiên cứu nền", "Thanh Tiền (S04): M5 là bối cảnh; Test của nhóm trùng giai đoạn public leaderboard M5.",
          ["Makridakis và cs. (2022): M5, 42.840 chuỗi, 12 cấp, WRMSSE",
           "Đội thắng dùng LightGBM, hơn ES_bu 22,4% trên 12 cấp nhưng chỉ khoảng 3,4% ở cấp sản phẩm–cửa hàng",
           "ETS (Hyndman & Khandakar 2008), Croston (1972) và SBA (Syntetos & Boylan 2005), LightGBM (Ke và cs. 2017)",
           "Kết hợp dự báo là kết luận lặp lại của M5",
           "Đồ án: 5 cách tiếp cận trên cùng 30.490 chuỗi; xếp hạng bằng macro-RMSSE, kèm WRMSSE như M5"])
    keep = [m for m in pilot.sort_values().index if m in LABEL and m != "lightgbm_noprice"]
    slide("Thiết kế hai giai đoạn: pilot rồi toàn bộ dữ liệu", "Tuấn Hải (D03), Thanh Tiền (D02): pilot 11 mô hình trên 1.200 chuỗi để chọn 5.",
          ["Pilot: 7 mô hình lõi + 4 phương án mở rộng trên 1.200 chuỗi phân tầng",
           "Chọn 5 đại diện 5 cách tiếp cận; bỏ mô hình trùng vai trò hoặc không đứng vững trên Test",
           "Chính thức: 30.490 chuỗi; Train d_1–1885, Validation d_1886–1913, Test d_1914–1941",
           "Test chạy một lần sau tag config-freeze (tạo lại cho cấu hình toàn bộ dữ liệu)"],
          table=pd.DataFrame({"Pilot (Test)": [{**LABEL, "stacking": "Stacking cũ (4 mô hình học máy)"}[m] for m in keep],
                              "RMSSE": [float(pilot[m]) for m in keep],
                              "Chọn": ["dựng lại (lai)" if m == "stacking" else "có" if m in ORDER else "" for m in keep]}))
    slide("Đặc trưng và chống rò rỉ", "Thùy Trang (D04): lag ≥ 28, rolling shift 28, kiểm thử làm hỏng nhãn tương lai.",
          ["lag 28/35/42; rolling mean/std 7 và 28 sau shift 28", "Lịch, sự kiện, SNAP theo bang, giá tuần (NaN + price_missing)",
           "Dự báo trực tiếp 28 ngày từ origin; kiểm thử làm hỏng mọi nhãn sau origin, đặc trưng không đổi",
           "Toàn bộ dữ liệu: khoảng 56 triệu dòng huấn luyện, cột phân loại dựng từ mã số để vừa 30 GB RAM"])
    slide("Mô hình cục bộ: Seasonal Naive, ETS, Croston SBA", "Tuấn Hải (Seasonal Naive), Tuấn Vũ (ETS, Croston SBA).",
          ["Seasonal Naive: ŷ(T+h)=y(T−6+((h−1) mod 7)), mốc đối chuẩn", "ETS: AutoETS chu kỳ 7, chọn cấu trúc theo AICc, mỗi chuỗi một mô hình",
           "Croston SBA: z (lượng bán), p (khoảng cách), ŷ=(1−α/2)·z/p, α=0,1", "Chỉ dùng lịch sử doanh số của chính chuỗi"])
    slide("LightGBM và Stacking lai", "Thủy Tiên (LightGBM), Thùy Trang (stacking).",
          ["LightGBM: một mô hình cho 30.490 chuỗi, tham số mặc định, F_m = F_{m−1} + η·f_m",
           "Stacking lai: ŷ = max(0, β0 + β1·ETS + β2·SBA + β3·LightGBM)",
           "β khớp bằng Ridge trên dự báo ngoài mẫu trong Train (mốc 1829, 1857)", "alpha chọn bằng leave-one-origin-out; không chạm Validation/Test"])
    slide("Độ đo", "Tuấn Vũ (W03/P03): công thức, edge cases, WRMSSE.",
          ["MAE, RMSE theo chuỗi; RMSSE chuẩn hóa bằng sai số naïve một bước trên Train", "Xếp hạng chính: macro-RMSSE trên 30.490 chuỗi",
           "WRMSSE: 12 cấp, 42.840 chuỗi, trọng số doanh thu 28 ngày cuối (như M5)", "R² có thể âm; chuỗi nhãn hằng → R² không xác định, không dùng epsilon"])
    for split, title, note in (("validation", "Kết quả Validation (d_1886–1913)", "Thủy Tiên (W07/P04): báo cáo trước khi khóa cấu hình."),
                               ("test", "Kết quả Test (d_1914–1941)", "Thủy Tiên: Test chạy một lần sau tag config-freeze; refit d_1–1913.")):
        o = overall(split)
        d = pd.DataFrame({"Mô hình": [LABEL[m] for m in ORDER], "MAE": o["macro_mae"].values, "RMSE": o["macro_rmse"].values,
                          "RMSSE": o["macro_rmsse"].values, "WRMSSE": wr[split].loc[ORDER, "wrmsse"].values, "fit+predict (s)": o["fit_predict_s"].values})
        slide(title, note, table=d,
              caption=f"Nguồn: results/full/tables/{split}/{split}_overall.csv, {split}_wrmsse.csv; macro trên 30.490 chuỗi; thời gian local/global khác phạm vi; stacking chỉ tính bước khớp bộ gộp.")
    lv = wr["test"].loc[ORDER]
    slide("WRMSSE theo cấp và đối chiếu M5", "Thanh Tiền (F08): Test của nhóm trùng giai đoạn public leaderboard M5.",
          [f"Tốt nhất theo WRMSSE trên Test: {LABEL[best_w]} ({c(lv.loc[best_w, 'wrmsse'])}); theo macro-RMSSE: {LABEL[best_t]}",
           "Cấp tổng (cấp 1): " + ", ".join(f"{LABEL[m]} {c(lv.loc[m, 'level_1'])}" for m in others),
           "Cấp sản phẩm–cửa hàng (cấp 12): " + ", ".join(f"{LABEL[m]} {c(lv.loc[m, 'level_12'])}" for m in others),
           "Chỉ một cửa sổ 28 ngày; không phải bài nộp thi"])
    cls = pd.read_csv(T / "test" / "test_by_demand_class.csv").pivot(index="demand_class", columns="model", values="rmsse")
    hz, lz = cls.loc["high_zero(>0.8)"], cls.loc["low_zero(<=0.5)"]
    slide("Theo nhóm nhu cầu", "Thanh Tiền (F08): nhóm theo tỷ lệ ngày 0 trên Train; số lấy từ test_by_demand_class.csv.",
          image=F / "test" / "test_rmsse_by_demand_class.png",
          bullets=["Hàng bán chậm (trên 80% ngày không bán): " + ", ".join(f"{LABEL[m]} {c(hz[m])}" for m in others),
                   "Hàng bán thường xuyên (tối đa 50% ngày không bán): " + ", ".join(f"{LABEL[m]} {c(lz[m])}" for m in others),
                   "Chỉ là liên hệ trên một cửa sổ 28 ngày"])
    wd = pd.read_csv(T / "test" / "test_mae_by_weekday.csv", index_col=0)[best_t]
    wk = wd.drop(["Saturday", "Sunday"])
    slide("Theo thứ trong tuần và horizon", "Thanh Tiền (F08).", image=F / "test" / "test_mae_by_horizon.png",
          bullets=["MAE gộp theo horizon 1–28 (hình)",
                   f"Cuối tuần sai nhiều hơn: MAE của {LABEL[best_t]} Thứ Bảy {c(wd['Saturday'])}, Chủ nhật {c(wd['Sunday'])}, ngày thường {c(wk.min())}–{c(wk.max())}"])
    slide("Đánh đổi độ chính xác và thời gian", "Tuấn Hải/Thủy Tiên: thời gian local và global khác phạm vi.", image=F / "test" / "test_accuracy_runtime.png",
          bullets=["Thời gian: local = tổng fit+predict từng chuỗi (4 tiến trình); global = fit trên toàn bộ dữ liệu",
                   "Stacking chỉ tốn thêm bước khớp bộ gộp, nhưng cần đủ ba mô hình cơ sở"])
    slide("Insight và giới hạn", "Thủy Tiên: kết luận gắn với bảng/hình; nêu giới hạn.",
          ["Chênh macro-RMSSE Test so với ETS: " + ", ".join(f"{LABEL[m]} {sg(ci.loc[m, 'diff_vs_ets'])}" for m in others if m != "ets"),
           "Trên Validation thứ tự đổi: ETS nhỉnh hơn stacking, LightGBM có WRMSSE thấp nhất → chênh lệch nhỏ không bền vững",
           "Croston SBA tốt ở cấp đáy nhưng kém nhất ở cấp tổng hợp (dự báo phẳng, không có chu kỳ tuần)",
           "Một cửa sổ Test 28 ngày; nhãn Test của 1.200 chuỗi pilot đã được xem trước khi chọn mô hình",
           "Khác biệt input local/global; giả định giá tương lai; importance không phải nhân quả"])
    slide("Kết luận và hướng phát triển", "Tuấn Hải (W08/P05).",
          [f"Tốt nhất trên Test: {LABEL[best_t]} theo macro-RMSSE, {LABEL[best_w]} theo WRMSSE", "Mọi mô hình cải thiện rõ so với Seasonal Naive",
           "Hướng: rolling-origin nhiều mùa, mô phỏng tồn kho, dự báo xác suất"])
    slide("Q&A", "Cả nhóm: xem report/qa.md.",
          ["Câu hỏi dự kiến: vì sao 5 mô hình, pilot, lag ≥ 28, RMSSE và WRMSSE, Croston SBA, stacking lai, giá tương lai, local/global"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print("saved", OUT, len(prs.slides), "slides")


if __name__ == "__main__":
    main()
