import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


def run_final_arxiv_did_analysis():
    print("🚀 启动【arXiv 平衡面板：大模型时代因果效应】终极分析...\n")

    # 1. 加载平衡后的 arXiv 专属面板
    try:
        df = pd.read_csv('arXiv_Balanced_DID_Panel.csv')
        print(f"📊 成功加载平衡面板，共 {len(df):,} 条 Author-Year 观测记录。")
    except FileNotFoundError:
        print("❌ 找不到 arXiv_Balanced_DID_Panel.csv，请确认路径。")
        return

    # 2. 依然锁定“大模型时代 (2022及之后)”发生的合作队列
    # 计算干预年份 T0
    df['T0_year'] = df['year'] - df['relative_time']
    df_llm = df[df['T0_year'] >= 2022].copy()
    df_llm['year_str'] = df_llm['year'].astype(str)

    print(f"🎯 锁定 2022-2026 跨界队列，样本量：{len(df_llm):,} 行。")

    # ==========================================
    # 3. 跑出最严谨的 DID 基准回归
    # ==========================================
    print("\n⚖️ 正在计算双重差分 (DID) 净效应...")
    # 公式：影响力 ~ 处理组*干预后 + 年份固定效应
    formula = "mean_log_rcr ~ is_treated * Post + C(year_str)"
    model = smf.ols(formula, data=df_llm).fit(cov_type='HC1')

    print("=========================================================")
    print("✅ 终极回归结果 (这是你论文 Section 3.5 的核心数据)：")
    print("=========================================================")
    res_df = pd.DataFrame({'Coefficient': model.params, 'Std.Error': model.bse, 'P-value': model.pvalues})
    print(res_df.loc[['is_treated', 'Post', 'is_treated:Post']])
    print("=========================================================\n")

    # ==========================================
    # 4. 绘制动态事件研究图 (Event Study Plot)
    # ==========================================
    print("🎨 正在绘制动态平行趋势图 (这是你论文最漂亮的一张图)...")

    # 计算均值和标准误 (用于画置信区间)
    # 我们要展示相对时间从 -3 到 +3 的动态变化
    stats = df_llm.groupby(['relative_time', 'is_treated'])['mean_log_rcr'].agg(['mean', 'sem']).unstack()

    plt.figure(figsize=(11, 6), facecolor='white')

    # 处理组：红色圆点
    plt.errorbar(stats.index, stats['mean'][1], yerr=1.96 * stats['sem'][1],
                 fmt='-o', color='firebrick', capsize=5, elinewidth=1.5,
                 label='Treated (Collaborated with HSS)', markersize=8, linewidth=2)

    # 对照组：蓝色方块
    plt.errorbar(stats.index, stats['mean'][0], yerr=1.96 * stats['sem'][0],
                 fmt='--s', color='navy', capsize=5, elinewidth=1.5,
                 label='Control (Pure STEM Twins)', markersize=8, linewidth=2, alpha=0.7)

    # 标出 T0 垂直线
    plt.axvline(x=0, color='black', linestyle=':', linewidth=2, label='First Collaboration ($T_0$)')
    # 标出 0 水平参考线
    plt.axhline(y=stats['mean'][0].iloc[0], color='gray', linestyle='--', alpha=0.3)

    plt.title('Event Study: Causal Impact of HSS Integration on AI Research Impact (arXiv)',
              fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Years Relative to Collaboration (Event Time)', fontsize=12)
    plt.ylabel('Scientific Impact (Mean Log-RCR)', fontsize=12)

    plt.xticks(np.arange(-3, 4, 1))
    plt.legend(fontsize=11, loc='upper left')
    plt.grid(True, axis='y', alpha=0.2)

    # 装饰美化
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.tight_layout()
    plot_fn = 'arXiv_Final_DID_Impact.png'
    plt.savefig(plot_fn, dpi=300)
    print(f"🎉 动态趋势图已完美出炉：{plot_fn}")


run_final_arxiv_did_analysis()