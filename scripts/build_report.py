"""Fill the generated tables of report/draft_report.md from result CSV/JSON.

Every block  <!-- TABLE:name -->  ...  <!-- /TABLE -->  is replaced by a markdown table built from the files below.
    python scripts/build_report.py

Main study: all 30,490 series, 5 models (results/full/). Pilot: 1,200-series sample, 7 core + 4 optional models (results/runs,
results/tables), used to choose the 5 models; only the pilot_* tables read it.
"""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "results" / "full"
T, RUNS = FULL / "tables", FULL / "runs"
PILOT_T, PILOT_RUNS = ROOT / "results" / "tables", ROOT / "results" / "runs"
REPORT = ROOT / "report" / "draft_report.md"
L = {"seasonal_naive": "Seasonal Naive", "linear_regression": "Linear Regression", "ets": "ETS", "croston": "Croston",
     "croston_sba": "Croston SBA", "lightgbm": "LightGBM", "xgboost": "XGBoost", "catboost": "CatBoost",
     "lightgbm_tuned": "LightGBM Tuned", "lightgbm_noprice": "LightGBM bỏ giá/sự kiện", "stacking": "Stacking lai"}
ORDER = ["seasonal_naive", "ets", "croston_sba", "lightgbm", "stacking"]
N_POINTS = 30490 * 28


def num(text):
    """Vietnamese decimal comma and a true minus sign (U+2212), matching the report text."""
    text = text.replace(".", ",")
    return "−" + text[1:] if text.startswith("-") else text


def md(df, floatfmt="{:.3f}"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(num(floatfmt.format(v)) if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(lines)


def raw(split):
    return pd.read_csv(T / split / "metrics_summary.csv").set_index("model").loc[ORDER]  # unrounded values


def overall(split):
    d = raw(split).join(pd.read_csv(T / split / f"{split}_overall.csv", index_col=0)[["mae_vs_snaive_pct", "rmsse_vs_snaive_pct"]])
    rt = pd.read_csv(T / split / f"{split}_runtime_tradeoff.csv").set_index("model")
    out = pd.DataFrame({"Mô hình": [L[m] for m in d.index], "MAE (macro)": d["macro_mae"].values, "RMSE (macro)": d["macro_rmse"].values,
                        "RMSSE (macro)": d["macro_rmsse"].values, "R² (macro)": d["macro_r2"].values,
                        "So Seasonal Naive (RMSSE)": [num(f"{v:+.1f}%") for v in d["rmsse_vs_snaive_pct"]],
                        "fit+predict (s)": [f"{float(rt.loc[m, 'fit_predict_s']):.1f}".replace(".", ",") for m in d.index]})
    return md(out)


def wrmsse(split):
    d = pd.read_csv(T / split / f"{split}_wrmsse.csv").set_index("model").loc[ORDER]
    out = pd.DataFrame({"Mô hình": [L[m] for m in d.index], "WRMSSE": d["wrmsse"].values,
                        "Cấp 1 (tổng)": d["level_1"].values, "Cấp 3 (cửa hàng)": d["level_3"].values,
                        "Cấp 9 (cửa hàng × nhóm hàng)": d["level_9"].values, "Cấp 12 (sản phẩm × cửa hàng)": d["level_12"].values})
    return md(out)


def group(split, key, metric="rmsse"):
    d = pd.read_csv(T / split / f"{split}_by_{key}.csv")
    p = d.pivot(index=key, columns="model", values=metric)[ORDER].rename(columns=L)
    p.insert(0, key, p.index)
    return md(p.reset_index(drop=True))


def bootstrap(split):
    d = pd.read_csv(T / split / f"{split}_bootstrap_ci.csv")
    out = pd.DataFrame({"Mô hình": [L[m] for m in d["model"]], "RMSSE (macro)": d["macro_rmsse"].values,
                        "KTC 95%": [f"[{num(f'{a:.3f}')}; {num(f'{b:.3f}')}]" for a, b in zip(d["ci_low"], d["ci_high"])],
                        "Chênh so với ETS": ["—" if m == "ets" else num(f"{v:+.4f}") for m, v in zip(d["model"], d["diff_vs_ets"])],
                        "KTC 95% của chênh lệch": ["—" if m == "ets" else f"[{num(f'{a:+.4f}')}; {num(f'{b:+.4f}')}]"
                                                    for m, a, b in zip(d["model"], d["diff_ci_low"], d["diff_ci_high"])]})
    return md(out)


def bootstrap_pairs():
    rows = []
    for split, label in (("validation", "Validation"), ("test", "Test")):
        for r in pd.read_csv(T / split / f"{split}_bootstrap_pairs.csv").itertuples():
            rows.append({"Tập": label, "Cặp (thứ nhất − thứ hai)": f"{L[r.first]} − {L[r.second]}", "Chênh": num(f"{r.diff:+.4f}"),
                         "KTC 95%": f"[{num(f'{r.ci_low:+.4f}')}; {num(f'{r.ci_high:+.4f}')}]",
                         "Khoảng chứa 0": "có" if r.ci_low <= 0 <= r.ci_high else "không"})
    return md(pd.DataFrame(rows))


def class_after_first_sale(split):
    d = pd.read_csv(T / split / f"{split}_by_demand_class_after_first_sale.csv").set_index("demand_class_after_first_sale")
    out = d[ORDER].rename(columns=L)
    out.insert(0, "Số chuỗi", d["n_series"].astype(int))
    out.insert(0, "Nhóm (sau lần bán đầu tiên)", out.index)
    return md(out.reset_index(drop=True))


def stack_coef():
    rows = []
    for p in sorted(RUNS.glob("*-stack")):
        e = json.loads((p / "environment.json").read_text())
        c = pd.read_csv(p / "diag_stacking_coefficients.csv").set_index("feature")["coefficient"]
        rows.append({"Run": e["run_id"], "alpha": num(f"{e['meta']['alpha']:g}"), **{L.get(f, "hệ số chặn"): float(v) for f, v in c.items()},
                     "Dòng OOF": f"{e['meta']['oof_rows']:,}".replace(",", ".")})
    return md(pd.DataFrame(rows), "{:.4f}")


PILOT_CHOICE = {"seasonal_naive": "chọn (mốc)", "ets": "chọn", "lightgbm": "chọn", "croston_sba": "chọn",
                "stacking": "chọn, dựng lại thành bản lai ETS + SBA + LightGBM"}


def pilot_summary():
    """Test macro-RMSSE of all 11 models of the 1,200-series pilot and the decision taken for the full-data study."""
    t = pd.concat([pd.read_csv(p / "metrics_summary.csv") for p in sorted(PILOT_RUNS.glob("tes-*"))]).set_index("model")
    v = pd.concat([pd.read_csv(p / "metrics_summary.csv") for p in sorted(PILOT_RUNS.glob("val-*"))]).set_index("model")
    names = t["macro_rmsse"].sort_values().index
    label = {**L, "stacking": "Stacking (LR + LightGBM Tuned + XGBoost + CatBoost)"}
    out = pd.DataFrame({"Mô hình (pilot 1.200 chuỗi)": [label[m] for m in names], "RMSSE Validation": v.loc[names, "macro_rmsse"].values,
                        "RMSSE Test": t.loc[names, "macro_rmsse"].values, "Quyết định": [PILOT_CHOICE.get(m, "bỏ") for m in names]})
    return md(out)


def table(name):
    if name == "pilot_summary":
        return pilot_summary()
    if name == "stack_coef":
        return stack_coef()
    if name in ("val_wrmsse", "test_wrmsse"):
        return wrmsse("validation" if name.startswith("val") else "test")
    if name in ("val_bootstrap", "test_bootstrap"):
        return bootstrap(name.split("_")[0].replace("val", "validation"))
    if name == "bootstrap_pairs":
        return bootstrap_pairs()
    if name == "test_class_after_first_sale":
        return class_after_first_sale("test")
    special = ("ets_structure", "importance", "runs")
    split = "test" if name.startswith("test_") else "validation"
    base = name if name in special else name.split("_", 1)[1]
    if base == "overall":
        return overall(split)
    if base in ("demand_class_rmsse", "cat_rmsse", "state_rmsse", "demand_class_mae"):
        key = {"demand_class": "demand_class", "cat": "cat_id", "state": "state"}[base.rsplit("_", 1)[0]]
        return group(split, key, base.rsplit("_", 1)[1])
    if base == "pooled":
        d = raw(split)
        out = pd.DataFrame({"Mô hình": [L[m] for m in d.index], "RMSE macro": d["macro_rmse"].values, "RMSE gộp": d["pooled_rmse"].values,
                            "MAE macro": d["macro_mae"].values, "R² không xác định (số chuỗi)": d["n_r2_undefined"].astype(int).values,
                            "RMSSE không xác định (số chuỗi)": d["n_rmsse_undefined"].astype(int).values})
        return md(out)
    if base == "negraw":
        rows = []
        for p in sorted(RUNS.glob(f"{split[:3]}-*")):
            t = json.loads((p / "timings.json").read_text())
            for m, v in t.items():
                if isinstance(v, dict) and "n_negative_raw_predictions" in v and m in ORDER:
                    rows.append({"Mô hình": L[m], f"Số dự báo âm trước khi cắt về 0 (trên {N_POINTS:,})".replace(",", "."): v["n_negative_raw_predictions"]})
        return md(pd.DataFrame(rows))
    if base == "weekday":
        d = pd.read_csv(T / split / f"{split}_mae_by_weekday.csv")
        return md(d[["weekday", *ORDER]].rename(columns={**L, "weekday": "Thứ"}))
    if base == "horizon":
        d = pd.read_csv(T / split / f"{split}_mae_by_horizon.csv", index_col=0).loc[[1, 7, 14, 21, 28], ORDER].rename(columns=L)
        d.insert(0, "horizon", d.index)
        return md(d.reset_index(drop=True))
    if base == "winrate":
        d = pd.read_csv(T / split / f"{split}_win_rate_vs_snaive.csv", index_col=0).loc[[m for m in ORDER if m != "seasonal_naive"]]
        d.insert(0, "Mô hình", [L[m] for m in d.index])
        d = d.rename(columns={"win_rate_vs_snaive": "Tỷ lệ chuỗi thắng Seasonal Naive", "tie_rate": "Tỷ lệ hòa", "median_mae_diff": "Chênh MAE trung vị"})
        return md(d.reset_index(drop=True))
    if base == "ets_structure":
        run = next(p for p in sorted(RUNS.glob("val-*")) if (p / "diag_ets.csv").exists())
        d = pd.read_csv(run / "diag_ets.csv")
        vc = d["ets_method"].value_counts().rename_axis("Cấu trúc").reset_index(name="Số chuỗi").head(8)
        vc["Tỷ lệ"] = [f"{v / len(d):.1%}".replace(".", ",") for v in vc["Số chuỗi"]]
        vc["Số chuỗi"] = [f"{v:,}".replace(",", ".") for v in vc["Số chuỗi"]]
        st = d[["ets_alpha", "ets_beta", "ets_gamma"]].describe().loc[["count", "mean", "50%", "max"]].round(3).astype(object)
        st.loc["count"] = [f"{int(v):,}".replace(",", ".") for v in st.loc["count"]]
        return md(vc) + "\n\n*α/β/γ (số chuỗi có tham số, trung bình, trung vị, lớn nhất):*\n\n" + md(st.reset_index().rename(columns={"index": "thống kê"}))
    if base == "importance":
        run = next(p for p in sorted(RUNS.glob("val-*")) if (p / "diag_lightgbm_importance.csv").exists())
        d = pd.read_csv(run / "diag_lightgbm_importance.csv")
        d["share"] = d["importance"] / d["importance"].sum()
        d = d.sort_values("share", ascending=False).head(8)
        return md(pd.DataFrame({"Đặc trưng": d["feature"].values, "Tỷ trọng gain": [f"{v:.1%}".replace(".", ",") for v in d["share"]]}))
    if base == "runs":
        rows = []
        for p in sorted(RUNS.glob("*")):
            e = json.loads((p / "environment.json").read_text())
            rows.append({"run_id": e["run_id"], "split": e["split"], "origin": f"d_{e['origin_day']}", "mô hình": ", ".join(L.get(m, m) for m in e["models"]),
                         "SHA mã": e["git_sha"][:7], "CPU": e.get("cpu_count", "máy cục bộ"), "Python": e.get("python", "n/a").split()[0]})
        return md(pd.DataFrame(rows))
    raise KeyError(name)


def main():
    text = REPORT.read_text(encoding="utf-8")
    pat = re.compile(r"<!-- TABLE:(\w+) -->.*?<!-- /TABLE -->", re.S)
    text = pat.sub(lambda m: "<!-- TABLE:" + m.group(1) + " -->\n" + table(m.group(1)) + "\n<!-- /TABLE -->", text)
    REPORT.write_text(text, encoding="utf-8")
    print("tables filled:", len(pat.findall(text)))


if __name__ == "__main__":
    main()
