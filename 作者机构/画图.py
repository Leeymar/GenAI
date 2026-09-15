import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator, PercentFormatter
from matplotlib import font_manager

# =========================
# 0. 中文字体设置
# =========================
candidate_fonts = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "PingFang SC",
    "Arial Unicode MS",
    "DejaVu Sans"
]

available_fonts = {f.name for f in font_manager.fontManager.ttflist}
selected_font = next((f for f in candidate_fonts if f in available_fonts), "DejaVu Sans")

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 400,
    "font.family": "sans-serif",
    "font.sans-serif": [selected_font],
    "axes.unicode_minus": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.0,
    "axes.labelsize": 12,
    "axes.titlesize": 15,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})

# =========================
# 1. 读取数据
# =========================
file_path = r"E:\PythonProject\GenAI\data_process\derived\genai_regression_enriched_v3_fixed.csv"
df = pd.read_csv(file_path, low_memory=False)

# =========================
# 2. 变量设置
# =========================
x_var = "team_hss_dna_pct"     # HSS知识浓度
y_var = "请改成你的结果变量名"   # 例如：novelty / impact / disruptiveness
y_label = "结果变量（请改成中文名称）"

fig_name = "图2_HSS知识浓度与结果变量关系图"

# =========================
# 3. 数据清洗
# =========================
data = df[[x_var, y_var]].replace([np.inf, -np.inf], np.nan).dropna().copy()
data[x_var] = data[x_var].astype(float).clip(0, 100)
data[y_var] = data[y_var].astype(float)

print(f"用于作图的样本量: {len(data):,}")

# =========================
# 4. 分箱：按分位数分组
#    这样每组样本量更均衡，图更稳
# =========================
n_bins = 20
data["x_bin"] = pd.qcut(data[x_var], q=n_bins, duplicates="drop")

plot_df = (
    data.groupby("x_bin", observed=True)
        .agg(
            x_mean=(x_var, "mean"),
            y_mean=(y_var, "mean"),
            n=(y_var, "size"),
            y_std=(y_var, "std")
        )
        .reset_index(drop=True)
)

plot_df["y_std"] = plot_df["y_std"].fillna(0)
plot_df["se"] = plot_df["y_std"] / np.sqrt(plot_df["n"])
plot_df["ci_low"] = plot_df["y_mean"] - 1.96 * plot_df["se"]
plot_df["ci_high"] = plot_df["y_mean"] + 1.96 * plot_df["se"]

# =========================
# 5. 原始样本的线性拟合
# =========================
coef = np.polyfit(data[x_var], data[y_var], 1)
x_line = np.linspace(0, 100, 200)
y_line = coef[0] * x_line + coef[1]

corr = data[[x_var, y_var]].corr().iloc[0, 1]

# =========================
# 6. 作图
# =========================
fig, ax = plt.subplots(figsize=(8.8, 5.4), facecolor="white")

# 分箱均值 + 95% CI
ax.errorbar(
    plot_df["x_mean"],
    plot_df["y_mean"],
    yerr=[
        plot_df["y_mean"] - plot_df["ci_low"],
        plot_df["ci_high"] - plot_df["y_mean"]
    ],
    fmt="o",
    markersize=5.5,
    linewidth=1.1,
    capsize=3,
    color="#2F6DB5",
    ecolor="#9DBFE8",
    markerfacecolor="#2F6DB5",
    markeredgecolor="white",
    markeredgewidth=0.7,
    label="分箱均值（95%置信区间）"
)

# 线性拟合线
ax.plot(
    x_line,
    y_line,
    color="#F28E2B",
    linewidth=2.2,
    linestyle="--",
    label="线性拟合"
)

# 标题与坐标轴
ax.set_title("HSS知识浓度与结果变量的关系", pad=12)
ax.set_xlabel("团队 HSS 知识浓度（%）")
ax.set_ylabel(y_label)

# X轴刻度
ax.set_xlim(0, 100)
ax.xaxis.set_major_locator(MultipleLocator(10))
ax.xaxis.set_minor_locator(MultipleLocator(5))

# Y轴刻度
ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

# 如果结果变量是 0/1 变量，把下面这行取消注释
# ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

# 网格
ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.25)

# 图例
ax.legend(frameon=False, loc="best")

# 左上角信息框
info_text = (
    f"样本量 N = {len(data):,}\n"
    f"相关系数 = {corr:.3f}"
)
ax.text(
    0.02, 0.95,
    info_text,
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=10,
    bbox=dict(
        boxstyle="round,pad=0.35",
        facecolor="white",
        edgecolor="0.8",
        alpha=0.95
    )
)

plt.tight_layout()

# =========================
# 7. 保存
# =========================
plt.savefig(f"{fig_name}.png", bbox_inches="tight")
plt.savefig(f"{fig_name}.pdf", bbox_inches="tight")
plt.show()
