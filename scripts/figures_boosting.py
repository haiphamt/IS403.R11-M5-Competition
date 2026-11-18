"""M10: (a) toy illustration of additive trees (NOT a learned M5 model), (b) feature importance of the real fitted models.

    python scripts/figures_boosting.py --runs results/runs/val-A results/runs/val-B --out results/figures
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.tree import DecisionTreeRegressor  # noqa: E402


def toy(out):
    rng = np.random.default_rng(0)
    x = np.linspace(0, 6, 120)
    y = np.sin(x) + 0.3 * rng.normal(size=x.size)
    pred, eta, trees = np.zeros_like(y), 0.5, []
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2), sharey=True)
    axes[0].scatter(x, y, s=8)
    axes[0].set_title("Dữ liệu minh họa (không phải M5)")
    for m in range(2):
        t = DecisionTreeRegressor(max_depth=2).fit(x[:, None], y - pred)
        trees.append(t)
        step = eta * t.predict(x[:, None])
        pred = pred + step
        axes[m + 1].plot(x, step, label=f"η·f{m + 1}(x)")
        axes[m + 1].set_title(f"Cây {m + 1} học phần dư")
    axes[3].scatter(x, y, s=8, alpha=0.4)
    axes[3].plot(x, pred, "r", label="F2 = η·f1 + η·f2")
    axes[3].set_title("Tổng hai cây")
    axes[3].legend(fontsize=7)
    fig.suptitle("Sơ đồ minh họa cơ chế cộng cây; khác với cây thật học được trên M5", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "m10_boosting_illustration.png", dpi=140)
    plt.close(fig)


def importance(runs, out):
    files = {"LightGBM": "diag_lightgbm_importance.csv", "XGBoost": "diag_xgboost_importance.csv", "CatBoost": "diag_catboost_importance.csv"}
    found = {name: next((Path(r) / f for r in runs if (Path(r) / f).exists()), None) for name, f in files.items()}
    found = {k: v for k, v in found.items() if v is not None}  # full-data study: LightGBM only
    if not found:
        raise FileNotFoundError(f"không run nào trong {list(runs)} có tệp importance ({', '.join(files.values())})")
    fig, axes = plt.subplots(1, len(found), figsize=(4.7 * len(found), 4), squeeze=False)
    for ax, (name, path) in zip(axes[0], found.items()):
        d = pd.read_csv(path)
        d["share"] = d["importance"] / d["importance"].sum()
        d = d.sort_values("share").tail(10)
        ax.barh(d["feature"], d["share"])
        ax.set_title(f"{name} ({d['type'].iloc[0]})", fontsize=9)
        ax.set_xlabel("Tỷ trọng trong từng mô hình (không so giữa các mô hình)", fontsize=7)
    fig.suptitle("Importance của mô hình đã huấn luyện (mức đóng góp vào dự báo, không phải nhân quả)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "m10_feature_importance.png", dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    toy(a.out)
    importance(a.runs, a.out)
    print("ok")


if __name__ == "__main__":
    main()
