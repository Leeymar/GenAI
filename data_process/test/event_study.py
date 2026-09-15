import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import re

def run_annual_event_study():
    print("🚀 启动【年度 Event Study 动态因果检验】引擎...")

    # ==========================================
    # 1. 加载你的终极数据集
    # ==========================================
    file_path = 'Ultimate_Regression_Base.csv'
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"❌ 找不到文件 {file_path}")
        return

    # 清洗：剔除缺失值
    df = df.dropna(subset=['year', 'team_hss_dna_pct', 'Log_RCR', 'total_authors', 'Prior_Knowledge_Stock'])
    # 限制年份范围
    df = df[(df['year'] >= 2018) & (df['year'] <= 2025)].copy()

    # 确保年份是整型
    df['year'] = df['year'].astype(int)

    # 设定 Event Study 的基准年份
    baseline_year = 2021

    print("📊 正在运行加入硬核控制变量的 OLS 交互项模型 (请稍候)...")

    # ==========================================
    # 2. 构建计量经济学模型 (极度严谨版)
    # ==========================================
    # 🚨 核心修复：使用 '*' 将主效应和交互项完整包含，保证系数是真正的“动态差异”
    formula = (
        "Log_RCR ~ np.log(total_authors) + np.log(Prior_Knowledge_Stock + 1) "
        "+ C(year, Treatment(reference=2021)) * team_hss_dna_pct"
    )

    # 使用稳健标准误
    model = smf.ols(formula, data=df).fit(cov_type='HC1')

    # ==========================================
    # 3. 提取动态系数与置信区间 (正则表达式鲁棒版)
    # ==========================================
    results_dict = {}
    conf_int = model.conf_int(alpha=0.05)

    for idx in model.params.index:
        # 只捕捉包含文科浓度、包含年份、并且是交互项(:) 的特征
        if 'team_hss_dna_pct' in idx and 'year' in idx and ':' in idx:
            # 🚨 核心修复：用正则表达式提取 [T.2018] 里面的数字，无视前后顺序
            match = re.search(r'\[T\.(\d+)(?:\.\d+)?\]', idx)
            if match:
                year_int = int(match.group(1))
                coef = model.params[idx]
                lower_ci = conf_int.loc[idx, 0]
                upper_ci = conf_int.loc[idx, 1]
                results_dict[year_int] = {'coef': coef, 'lower': lower_ci, 'upper': upper_ci}

    # 把基准年手动加进去，它的系数和误差强制为 0
    results_dict[baseline_year] = {'coef': 0.0, 'lower': 0.0, 'upper': 0.0}

    # 转换为 DataFrame 并按年份排序
    res_df = pd.DataFrame.from_dict(results_dict, orient='index').reset_index()
    res_df.rename(columns={'index': 'Year'}, inplace=True)
    res_df = res_df.sort_values('Year')

    # 在终端打印一下提取到的数据，核实数据有没有被吃掉
    print("\n🔍 提取到的历年系数如下：")
    print(res_df.to_string(index=False))

    # ==========================================
    # 4. 绘制顶刊级 Coefficient Plot
    # ==========================================
    print("\n🎨 正在生成年度动态效应图...")
    fig, ax = plt.subplots(figsize=(10, 6))

    yerr_lower = res_df['coef'] - res_df['lower']
    yerr_upper = res_df['upper'] - res_df['coef']

    x = res_df['Year']
    y = res_df['coef']

    # 画线和误差棒
    ax.errorbar(x, y, yerr=[yerr_lower, yerr_upper], fmt='-o', color='#1d3557',
                ecolor='#457b9d', elinewidth=2, capsize=5, capthick=2, markersize=8, linewidth=2,
                label='Marginal Citation Premium of HSS DNA')

    # 画一条贯穿的 y=0 的水平基准线
    ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.7)

    # 画 ChatGPT 发布的垂直断点线 (标在 2022 和 2023 之间)
    chatgpt_loc = 2022.8
    ax.axvline(x=chatgpt_loc, color='#e63946', linestyle='--', linewidth=2, zorder=0)
    ax.text(chatgpt_loc - 0.1, ax.get_ylim()[1] * 0.9, 'ChatGPT\nRelease\n(Late 2022)',
            color='#e63946', fontsize=12, fontweight='bold', ha='right', va='top')

    # 标注基准年
    ax.annotate('Baseline\n(2021)', xy=(2021, 0), xytext=(2021, ax.get_ylim()[0] * 0.4),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                fontsize=11, fontweight='bold', ha='center', color='black')

    # 坐标轴美化
    ax.set_ylabel('Estimated Coefficient of Team HSS DNA', fontsize=13, fontweight='bold')
    ax.set_xlabel('Publication Year', fontsize=13, fontweight='bold')
    ax.set_xticks(x)

    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='upper left', frameon=False, fontsize=12)

    plt.tight_layout()
    output_img = 'Event_Study_Annual_HSS_Premium.png'
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"✅ 绝杀图已保存: {output_img}")
    plt.show()

# 运行代码
run_annual_event_study()