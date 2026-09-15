import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_yoy_growth_rate_all_hss():
    # 1. 加载数据
    df = pd.read_csv('phase3_annual_growth_by_field.csv')
    df.set_index('primary_origin_field', inplace=True)

    # 2. 截取 2022-2025 年完整数据
    years = ['2022', '2023', '2024', '2025']
    df_years = df[years]

    # 3. 核心数学转换：计算逐年增长率 (YoY)
    df_yoy = df_years.pct_change(axis=1) * 100
    plot_years = ['2023', '2024', '2025']
    df_yoy = df_yoy[plot_years]

    # 4. 图表样式初始化，稍微加宽画布以适应更长图例
    plt.figure(figsize=(14, 9))
    sns.set_theme(style="whitegrid", context="talk")

    # 5. 完整纳入 6 大文科阵营
    highlight_hss = [
        'Social Sciences',
        'Psychology',
        'Business, Management and Accounting',
        'Arts and Humanities',
        'Economics, Econometrics and Finance',
        'Decision Sciences'
    ]
    # 理工科基准线
    baseline_stem = [
        'Computer Science',
        'Medicine',
        'Engineering'
    ]

    # 为 6 大文科分配 6 种不同的高亮实心标记
    hss_markers = ['o', 's', '^', 'D', 'v', 'p']

    # 为 3 大理工科分配不同的颜色、线型和灰度标记
    stem_markers = ['X', '*', 'P']
    stem_styles = ['--', ':', '-.']
    stem_colors = ['#555555', '#777777', '#999999']

    # 6. 绘制 6 大文科狂飙线
    for i, field in enumerate(highlight_hss):
        if field in df_yoy.index:
            plt.plot(plot_years, df_yoy.loc[field],
                     marker=hss_markers[i], markersize=10,
                     linewidth=3.5, label=f"{field} (HSS)")

    # 7. 绘制 3 大理工科基准线（视觉彻底分离）
    for i, field in enumerate(baseline_stem):
        if field in df_yoy.index:
            plt.plot(plot_years, df_yoy.loc[field],
                     marker=stem_markers[i], markersize=10,
                     linewidth=2.5, linestyle=stem_styles[i],
                     color=stem_colors[i], alpha=0.85,
                     label=f"{field} (Baseline)")

    # 8. 图表美化与标签
    plt.title('The "ChatGPT Shockwave": Year-over-Year (YoY) Growth Rate (%)',
              fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Year', fontsize=14, fontweight='bold')
    plt.ylabel('YoY Growth Rate (%)', fontsize=14, fontweight='bold')

    # 图例排版优化，放在右侧
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11, frameon=True)
    plt.grid(True, linestyle='--', alpha=0.7)

    # 添加一条 0% 的基准线
    plt.axhline(0, color='black', linewidth=1.5, linestyle='-')

    plt.tight_layout()

    # 9. 保存高清图片
    output_filename = 'phase3_yoy_growth_rate_all_hss.png'
    plt.savefig(output_filename, dpi=600, bbox_inches='tight')
    plt.close()

    print(f"✅ 包含 6 大文科且线条彻底分离的 YoY 高清大图已生成: {output_filename}")


# 立即执行画图！
plot_yoy_growth_rate_all_hss()