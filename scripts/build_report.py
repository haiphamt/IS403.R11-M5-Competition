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


def md(df, floatfmt="{:.3f}"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(lines)


def overall(split):
    d = pd.read_csv(T / split / f"{split}_overall.csv", index_col=0).loc[ORDER]
    rt = pd.read_csv(T / split / f"{split}_runtime_tradeoff.csv").set_index("model")
    out = pd.DataFrame({"Mô hình": [L[m] for m in d.index], "MAE (macro)": d["macro_mae"].values, "RMSE (macro)": d["macro_rmse"].values,
                        "RMSSE (macro)": d["macro_rmsse"].values, "R² (macro)": d["macro_r2"].values,
                        "So Seasonal Naive (MAE)": [f"{v:+.1f}%" for v in d["mae_vs_snaive_pct"]],
                        "So Seasonal Naive (RMSSE)": [f"{v:+.1f}%" for v in d["rmsse_vs_snaive_pct"]],
                        "fit+predict (s)": [round(float(rt.loc[m, "fit_predict_s"]), 1) for m in d.index]})
    return md(out)


def group(split, key, metric="rmsse"):
    d = pd.read_csv(T / split / f"{split}_by_{key}.csv")
    p = d.pivot(index=key, columns="model", values=metric)[ORDER].rename(columns=L)
    p.insert(0, key, p.index)
    return md(p.reset_index(drop=True))


def table(name):
    split = "test" if name.startswith("test_") else "validation"
    base = name.split("_", 1)[1]
    if base == "overall":
        return overall(split)
    if base in ("demand_class_rmsse", "cat_rmsse", "state_rmsse", "demand_class_mae"):
        key = {"demand_class": "demand_class", "cat": "cat_id", "state": "state"}[base.rsplit("_", 1)[0]]
        return group(split, key, base.rsplit("_", 1)[1])
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
        vc["Tỷ lệ"] = [f"{v / len(d):.1%}" for v in vc["Số chuỗi"]]
        st = d[["ets_alpha", "ets_beta", "ets_gamma"]].describe().loc[["count", "mean", "50%", "max"]].round(3)
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
            rows.append({"Mô hình": name, "Top 5 đặc trưng (tỷ trọng)": "; ".join(f"{a} {b:.1%}" for a, b in zip(d["feature"], d["share"]))})
        return md(pd.DataFrame(rows))
    if base == "runs":
        rows = []
        for p in sorted(RUNS.glob("*")):
            e = json.loads((p / "environment.json").read_text())
            rows.append({"run_id": e["run_id"], "split": e["split"], "origin": f"d_{e['origin_day']}", "mô hình": ", ".join(L.get(m, m) for m in e["models"]),
                         "SHA mã": e["git_sha"][:7], "CPU": e["cpu_count"], "Python": e["python"].split()[0]})
        return md(pd.DataFrame(rows))
    raise KeyError(name)


def main():
    text = REPORT.read_text(encoding="utf-8")
    pat = re.compile(r"(<!-- TABLE:(\w+) -->\n).*?(\n<!-- /TABLE -->)", re.S)
    text = pat.sub(lambda m: m.group(1) + table(m.group(2)) + m.group(3), text)
    REPORT.write_text(text, encoding="utf-8")
    print("tables filled:", len(pat.findall(text)))


if __name__ == "__main__":
    main()
