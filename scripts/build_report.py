"""Fill the generated tables of report/draft_report.md from result CSV/JSON.

Every block  <!-- TABLE:name -->  ...  <!-- /TABLE -->  is replaced by a markdown table built from the files below.
    python scripts/build_report.py
"""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results" / "tables"
RUNS = ROOT / "results" / "runs"
REPORT = ROOT / "report" / "draft_report.md"
L = {"seasonal_naive": "Seasonal Naive", "linear_regression": "Linear Regression", "ets": "ETS", "croston": "Croston",
     "lightgbm": "LightGBM", "xgboost": "XGBoost", "catboost": "CatBoost"}
ORDER = list(L)


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
                        "So Seasonal Naive (MAE)": [num(f"{v:+.1f}%") for v in d["mae_vs_snaive_pct"]],
                        "So Seasonal Naive (RMSSE)": [num(f"{v:+.1f}%") for v in d["rmsse_vs_snaive_pct"]],
                        "fit+predict (s)": [f"{float(rt.loc[m, 'fit_predict_s']):.1f}".replace(".", ",") for m in d.index]})
    return md(out)


def group(split, key, metric="rmsse"):
    d = pd.read_csv(T / split / f"{split}_by_{key}.csv")
    p = d.pivot(index=key, columns="model", values=metric)[ORDER].rename(columns=L)
    p.insert(0, key, p.index)
    return md(p.reset_index(drop=True))


OPT_LABEL = {"croston_sba": "Croston SBA", "lightgbm_noprice": "LightGBM bỏ giá/sự kiện", "lightgbm_tuned": "LightGBM Tuned (Optuna)",
             "stacking": "Stacking Ridge"}
OPT_NOTE = {"croston_sba": "X01", "lightgbm_noprice": "X02", "lightgbm_tuned": "T01; thời gian không gồm 1.295 giây tìm tham số",
            "stacking": "T02; thời gian chỉ gồm khớp bộ gộp, không gồm các mô hình cơ sở"}


def val_optional():
    """Validation scores of the optional models next to their core references (see test_optional for Test)."""
    rows = pd.concat([pd.read_csv(p / "metrics_summary.csv") for p in sorted(RUNS.glob("val-*"))]).set_index("model")
    names = [m for m in ["lightgbm", "lightgbm_tuned", "lightgbm_noprice", "croston", "croston_sba", "stacking"] if m in rows.index]
    d = rows.loc[names]
    lab = {**L, **OPT_LABEL}
    out = pd.DataFrame({"Mô hình": [lab[m] for m in names], "MAE (macro)": d["macro_mae"].values, "RMSE (macro)": d["macro_rmse"].values,
                        "RMSSE (macro)": d["macro_rmsse"].values, "fit+predict (s)": [f"{v:.1f}".replace(".", ",") for v in d["fit_predict_s"]],
                        "Ghi chú": [OPT_NOTE.get(m, "tham chiếu lõi") for m in names]})
    return md(out, "{:.4f}")


def test_optional():
    """Validation and Test scores of the optional models next to the core references they are compared with."""
    summ = lambda g: pd.concat([pd.read_csv(p / "metrics_summary.csv") for p in sorted(RUNS.glob(g))]).set_index("model")
    val, tst = summ("val-*"), summ("tes-*")
    names = ["ets", "lightgbm", "lightgbm_tuned", "lightgbm_noprice", "croston", "croston_sba", "stacking"]
    lab = {**L, **OPT_LABEL}
    out = pd.DataFrame({"Mô hình": [lab[m] for m in names], "RMSSE Validation": val.loc[names, "macro_rmsse"].values,
                        "RMSSE Test": tst.loc[names, "macro_rmsse"].values, "MAE Test (macro)": tst.loc[names, "macro_mae"].values,
                        "RMSE Test (macro)": tst.loc[names, "macro_rmse"].values, "RMSE Test (gộp)": tst.loc[names, "pooled_rmse"].values,
                        "fit+predict Test (s)": [f"{v:.1f}".replace(".", ",") for v in tst.loc[names, "fit_predict_s"]],
                        "Ghi chú": [OPT_NOTE.get(m, "tham chiếu lõi") for m in names]})
    return md(out)


def optional_pairs():
    lab = {**L, **OPT_LABEL}
    d = {sp: pd.read_csv(T / sp / f"{sp}_bootstrap_optional_pairs.csv").set_index(["first", "second"]) for sp in ("validation", "test")}
    rows = []
    for first, second in d["test"].index:  # pair order of OPT_PAIRS, Validation row then Test row
        for sp, label in (("validation", "Validation"), ("test", "Test")):
            r = d[sp].loc[(first, second)]
            rows.append({"Cặp (thứ nhất − thứ hai)": f"{lab[first]} − {lab[second]}", "Tập": label, "Chênh": num(f"{r['diff']:+.4f}"),
                         "KTC 95%": f"[{num(f'{r.ci_low:+.4f}')}; {num(f'{r.ci_high:+.4f}')}]",
                         "Khoảng chứa 0": "có" if r.ci_low <= 0 <= r.ci_high else "không"})
    return md(pd.DataFrame(rows))


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


def table(name):
    if name == "val_optional":
        return val_optional()
    if name == "test_optional":
        return test_optional()
    if name == "optional_pairs":
        return optional_pairs()
    if name in ("val_bootstrap", "test_bootstrap"):
        return bootstrap(name.split("_")[0].replace("val", "validation"))
    if name == "bootstrap_pairs":
        return bootstrap_pairs()
    if name == "test_class_after_first_sale":
        return class_after_first_sale("test")
    # ets_structure, croston_zp, lr_coef and importance come from the first core Validation run that holds the diagnostic file (val-...072831Z)
    special = ("ets_structure", "croston_zp", "lr_coef", "importance", "runs")
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
                if isinstance(v, dict) and "n_negative_raw_predictions" in v and m in L:  # core models only: optional runs have their own table
                    rows.append({"Mô hình": L[m], "Số dự báo âm trước khi cắt về 0 (trên 33.600)": v["n_negative_raw_predictions"]})
        return md(pd.DataFrame(rows))
    if base == "weekday":
        d = pd.read_csv(T / split / f"{split}_mae_by_weekday.csv").rename(columns=L)
        return md(d)
    if base == "horizon":
        d = pd.read_csv(T / split / f"{split}_mae_by_horizon.csv", index_col=0).loc[[1, 7, 14, 21, 28]].rename(columns=L)
        d.insert(0, "horizon", d.index)
        return md(d.reset_index(drop=True))
    if base == "winrate":
        d = pd.read_csv(T / split / f"{split}_win_rate_vs_snaive.csv", index_col=0)
        d.insert(0, "Mô hình", [L[m] for m in d.index])
        return md(d.reset_index(drop=True))
    if base == "ets_structure":
        run = next(p for p in sorted(RUNS.glob("val-*")) if (p / "diag_ets.csv").exists())
        d = pd.read_csv(run / "diag_ets.csv")
        vc = d["ets_method"].value_counts().rename_axis("Cấu trúc").reset_index(name="Số chuỗi")
        vc["Tỷ lệ"] = [f"{v / len(d):.1%}".replace(".", ",") for v in vc["Số chuỗi"]]
        st = d[["ets_alpha", "ets_beta", "ets_gamma"]].describe().loc[["count", "mean", "50%", "max"]].round(3).astype(object)
        st.loc["count"] = [int(v) for v in st.loc["count"]]
        return md(vc) + "\n\n*α/β/γ (số chuỗi có tham số, trung bình, trung vị, lớn nhất):*\n\n" + md(st.reset_index().rename(columns={"index": "thống kê"}))
    if base == "croston_zp":
        run = next(p for p in sorted(RUNS.glob("val-*")) if (p / "diag_croston.csv").exists())
        d = pd.read_csv(run / "diag_croston.csv")
        z = pd.concat([d.sort_values("croston_p").head(2), d.iloc[(d["croston_p"] - d["croston_p"].median()).abs().argsort()[:2]], d.sort_values("croston_p").tail(2)])
        return md(z.rename(columns={"series_id": "Chuỗi", "croston_z": "z", "croston_p": "p", "croston_zp": "ŷ = z/p"}), "{:.3f}")
    if base == "lr_coef":
        run = next(p for p in sorted(RUNS.glob("val-*")) if (p / "diag_linear_regression_coefficients.csv").exists())
        d = pd.read_csv(run / "diag_linear_regression_coefficients.csv")
        d = d.reindex(d["coefficient"].abs().sort_values(ascending=False).index).head(12)
        return md(d.rename(columns={"feature": "Biến", "coefficient": "Hệ số"}), "{:.4f}")
    if base == "importance":
        rows = []
        for name, f in (("LightGBM", "diag_lightgbm_importance.csv"), ("XGBoost", "diag_xgboost_importance.csv"), ("CatBoost", "diag_catboost_importance.csv")):
            run = next(p for p in sorted(RUNS.glob("val-*")) if (p / f).exists())
            d = pd.read_csv(run / f)
            d["share"] = d["importance"] / d["importance"].sum()
            d = d.sort_values("share", ascending=False).head(5)
            rows.append({"Mô hình": name, "Top 5 đặc trưng (tỷ trọng)": "; ".join(f"{a} {b:.1%}".replace(".", ",") for a, b in zip(d["feature"], d["share"]))})
        return md(pd.DataFrame(rows))
    if base == "runs":
        rows = []
        for p in sorted(RUNS.glob("*")):
            e = json.loads((p / "environment.json").read_text())
            rows.append({"run_id": e["run_id"], "split": e["split"], "origin": f"d_{e['origin_day']}", "mô hình": ", ".join({**L, **OPT_LABEL}.get(m, m) for m in e["models"]),
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
