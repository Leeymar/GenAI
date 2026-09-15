# -*- coding: utf-8 -*-
"""
plot_main_figures_clean.py

用途：
1. 读取 core / supplement 回归输出结果
2. 绘制不含图题、不含图注的干净版论文图
3. 输出 3 张图：
   - 技术阶段异质性
   - 非线性阈值效应（Log_RCR）
   - 高影响论文概率差异（Top 10%）

说明：
- 图题和图注请在论文 LaTeX 里写
- 图中只保留坐标轴、点估计、95% 置信区间
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# 0. 路径设置
# =========================================================
CORE_DIR = Path(r"E:\PythonProject\GenAI\data_process\reg_results_core")
SUPP_DIR = Path(r"E:\PythonProject\GenAI\data_process\reg_results_supplement")
FIG_DIR = Path(r"E:\PythonProject\GenAI\data_process\figures_clean")
FIG_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 1. 全局画图风格
# =========================================================
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 11


# =========================================================
# 2. 基础工具函数
# =========================================================
def check_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"缺少文件：{path}")


def save_figure(fig, stem: str):
    png_path = FIG_DIR / f"{stem}.png"
    pdf_path = FIG_DIR / f"{stem}.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"[OK] 已保存：{png_path}")
    print(f"[OK] 已保存：{pdf_path}")


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)


def horizontal_coef_plot(
    plot_df: pd.DataFrame,
    label_col: str,
    value_col: str,
    low_col: str,
    high_col: str,
    xlabel: str,
    out_name: str,
    figsize=(7.2, 4.8),
):
    fig, ax = plt.subplots(figsize=figsize)

    y = np.arange(len(plot_df))
    x = plot_df[value_col].values
    xerr = np.vstack([
        x - plot_df[low_col].values,
        plot_df[high_col].values - x
    ])

    ax.errorbar(
        x=x,
        y=y,
        xerr=xerr,
        fmt="o",
        capsize=3,
        markersize=5,
        linewidth=1.2
    )

    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df[label_col].tolist())
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    style_axis(ax)

    plt.tight_layout()
    save_figure(fig, out_name)
    plt.close(fig)


# =========================================================
# 3. 提取 HSS_Bin 阈值模型系数
# =========================================================
def extract_hss_bin_terms(coef_df: pd.DataFrame) -> pd.DataFrame:
    mapping = [
        ("Low HSS", "低 HSS 团队"),
        ("Medium HSS", "中等 HSS 团队"),
        ("High HSS", "高 HSS 团队"),
        ("HSS-dominant", "HSS 主导团队"),
    ]

    rows = []
    term_series = coef_df["term"].astype(str)

    for eng, zh in mapping:
        sub = coef_df[term_series.str.contains(eng, regex=False)].copy()
        if len(sub) == 0:
            continue
        sub = sub.iloc[0]
        rows.append({
            "组别": zh,
            "coef": sub["coef"],
            "ci_low": sub["ci_low"],
            "ci_high": sub["ci_high"],
        })

    out = pd.DataFrame(rows)
    return out


# =========================================================
# 4. 图1：技术阶段异质性
# 数据来源：stage_slope_tests.csv
# =========================================================
def plot_stage_heterogeneity():
    path = SUPP_DIR / "stage_slope_tests.csv"
    check_file(path)

    df = pd.read_csv(path)

    keep_map = {
        "Phase1_total_slope": "第一阶段",
        "Phase2_total_slope": "第二阶段",
        "Phase3_total_slope": "第三阶段",
    }

    plot_df = df[df["label"].isin(keep_map.keys())].copy()
    plot_df["阶段"] = plot_df["label"].map(keep_map)

    order = ["第一阶段", "第二阶段", "第三阶段"]
    plot_df["阶段"] = pd.Categorical(plot_df["阶段"], categories=order, ordered=True)
    plot_df = plot_df.sort_values("阶段").reset_index(drop=True)

    plot_df = plot_df.rename(columns={
        "estimate": "coef"
    })

    horizontal_coef_plot(
        plot_df=plot_df[["阶段", "coef", "ci_low", "ci_high"]],
        label_col="阶段",
        value_col="coef",
        low_col="ci_low",
        high_col="ci_high",
        xlabel="团队 HSS 浓度对领域标准化引用的边际效应",
        out_name="fig1_stage_heterogeneity_clean",
        figsize=(7.2, 4.4),
    )


# =========================================================
# 5. 图2：非线性阈值效应（Log_RCR）
# 数据来源：S1_LogRCR_Threshold_YearFE_coef.csv
# =========================================================
def plot_threshold_logrcr():
    path = SUPP_DIR / "S1_LogRCR_Threshold_YearFE_coef.csv"
    check_file(path)

    coef_df = pd.read_csv(path)
    plot_df = extract_hss_bin_terms(coef_df)

    if plot_df.empty:
        raise ValueError("未能从 S1_LogRCR_Threshold_YearFE_coef.csv 中识别 HSS_Bin 系数。")

    horizontal_coef_plot(
        plot_df=plot_df,
        label_col="组别",
        value_col="coef",
        low_col="ci_low",
        high_col="ci_high",
        xlabel="相对于纯 STEM 团队的估计系数（Log_RCR）",
        out_name="fig2_threshold_logrcr_clean",
        figsize=(7.4, 4.8),
    )


# =========================================================
# 6. 图3：高影响论文概率差异
# 数据来源：M6_Hit10YearCat_Threshold_LPM_coef.csv
# =========================================================
def plot_threshold_hit():
    path = CORE_DIR / "M6_Hit10YearCat_Threshold_LPM_coef.csv"
    check_file(path)

    coef_df = pd.read_csv(path)
    plot_df = extract_hss_bin_terms(coef_df)

    if plot_df.empty:
        raise ValueError("未能从 M6_Hit10YearCat_Threshold_LPM_coef.csv 中识别 HSS_Bin 系数。")

    horizontal_coef_plot(
        plot_df=plot_df,
        label_col="组别",
        value_col="coef",
        low_col="ci_low",
        high_col="ci_high",
        xlabel="相对于纯 STEM 团队进入高影响组的概率差异",
        out_name="fig3_threshold_hit_clean",
        figsize=(7.4, 4.8),
    )


# =========================================================
# 7. 主程序
# =========================================================
def main():
    print("=" * 80)
    print("开始生成干净版正文图……")
    print("=" * 80)

    plot_stage_heterogeneity()
    plot_threshold_logrcr()
    plot_threshold_hit()

    print("\n全部图形已生成。")
    print(f"输出目录：{FIG_DIR}")


if __name__ == "__main__":
    main()
