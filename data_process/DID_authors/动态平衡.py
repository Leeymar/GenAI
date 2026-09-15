import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from linearmodels.panel import PanelOLS
import statsmodels.api as sm
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体，防止图表乱码（Windows常用SimHei，Mac常用Arial Unicode MS）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def run_dynamic_did():
    print("⏳ 正在加载并清洗长面板数据...")
    df = pd.read_csv('Perfect_DID_Panel.csv')

    # 1. 截断极端的相对年份（防止样本太少导致标准误爆炸）
    # 比如我们只关注跨界前 4 年 到 跨界后 4 年的窗口期
    WINDOW_MIN = -4
    WINDOW_MAX = 4

    # 把超出窗口期的时间强行归入两端（尾部归并）
    df['rel_year_clipped'] = df['relative_year'].clip(lower=WINDOW_MIN, upper=WINDOW_MAX)

    # 2. 生成相对年份的虚拟变量（Dummy Variables）
    # 为每一个相对年份生成一列 0/1 变量
    dummies = pd.get_dummies(df['rel_year_clipped'], prefix='T')

    # 3. 生成 DID 核心交互项：is_treated × 年份虚拟变量
    # 这一步算出来的就是实验组在特定年份的专属红利
    interaction_cols = []
    for col in dummies.columns:
        inter_name = f"DID_{col}"
        df[inter_name] = df['is_treated'] * dummies[col]
        interaction_cols.append(inter_name)

    # 4. 【极其关键】设立基准期（Reference Period）
    # 在 DID 中，通常把跨界前的最后一年（T=-1）作为基准期踢出回归方程。
    # 之后所有的系数，都是在跟 T=-1 这一年做对比！
    baseline_col = 'DID_T_-1'
    if baseline_col in interaction_cols:
        interaction_cols.remove(baseline_col)
    else:
        # 如果没有 -1，可能格式是 DID_T_-1.0
        baseline_col = 'DID_T_-1.0'
        interaction_cols.remove(baseline_col)

    # 5. 构建并运行 TWFE 面板回归
    print(f"🚀 正在运行动态平行趋势回归 (基准期设为: 跨界前1年)...")
    df = df.set_index(['author_id', 'year'])

    Y = df['avg_logRCR']
    X = df[interaction_cols]
    X = sm.add_constant(X)

    model = PanelOLS(Y, X, entity_effects=True, time_effects=True)
    results = model.fit(cov_type='clustered', cluster_entity=True)

    print("✅ 回归完成！正在提取系数并绘制图表...")

    # 6. 提取系数和 95% 置信区间
    coefs = results.params.drop('const')
    conf_ints = results.conf_int().drop('const')

    # 将被踢除的基准期 T=-1 重新补回来，系数和误差都设为 0
    coefs[baseline_col] = 0.0
    conf_ints.loc[baseline_col] = [0.0, 0.0]

    # 整理画图数据
    plot_data = pd.DataFrame({
        'coef': coefs,
        'lower_ci': conf_ints['lower'],
        'upper_ci': conf_ints['upper']
    })

    # 从索引名 (如 DID_T_-3.0) 中直接剔除前缀，提取纯数字
    plot_data['time'] = [float(str(idx).replace('DID_T_', '')) for idx in plot_data.index]
    plot_data = plot_data.sort_values('time')

    # 极度安全机制：强行丢弃任何没提取成功的 NaN 垃圾数据
    plot_data = plot_data.dropna(subset=['time'])

    # 7. 绘制经典的 Event Study 平行趋势图
    plt.figure(figsize=(10, 6), dpi=150)

    # 绘制误差棒 (95% 置信区间) 和 点
    plt.errorbar(x=plot_data['time'], y=plot_data['coef'],
                 yerr=[plot_data['coef'] - plot_data['lower_ci'], plot_data['upper_ci'] - plot_data['coef']],
                 fmt='-o', color='crimson', ecolor='salmon', capsize=5, capthick=2, elinewidth=2, markersize=8)

    # 辅助线
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1)  # 0效应水平线
    plt.axvline(x=-0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)  # 干预发生的分界线

    # 美化图表
    plt.title('跨学科合作的动态溢价效应 (Event Study)', fontsize=16, pad=15)
    plt.xlabel('距离首次跨界合作的相对年份 (Relative Year)', fontsize=12)
    plt.ylabel('对平均 logRCR 的净影响 (估计系数)', fontsize=12)
    plt.xticks(plot_data['time'])
    plt.grid(axis='y', linestyle='--', alpha=0.4)

    # 标注基准期
    plt.annotate('跨界基准期 (T=-1)', xy=(-1, 0), xytext=(-2.5, 0.02),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                 fontsize=10)

    plt.tight_layout()
    plt.savefig('Parallel_Trend_Plot.png')
    print("🎉 图表已保存为：Parallel_Trend_Plot.png")

    plt.show()


if __name__ == "__main__":
    run_dynamic_did()