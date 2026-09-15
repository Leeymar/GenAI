import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def analyze_hss_growth_and_plot_ultimate():
    print("📈 启动【HSS 人文社科跨界觉醒】终极图表生成引擎 (学术定稿版)...\n")

    file_path = 'MASTER_AUTHOR_PROFILES_12YEARS.csv'

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"❌ 找不到文件 {file_path}")
        return

    # ---------------------------------------------------------
    # 1. 定义 HSS (人文与社会科学) 的 Scopus 学科映射池
    # ---------------------------------------------------------
    hss_fields = [
        'Social Sciences',
        'Arts and Humanities',
        'Psychology',
        'Business, Management and Accounting',
        'Economics, Econometrics and Finance',
        'Decision Sciences'
    ]

    def categorize_broad_field(field):
        if field == 'Computer Science':
            return 'Computer Science (CS)'
        elif field in hss_fields:
            return 'Humanities & Social Sciences (HSS)'
        elif field in ['Unknown', 'Error', 'Ghost']:
            return 'Invalid'
        else:
            return 'Other STEM & Medicine'

    df['Broad_Field'] = df['primary_origin_field'].apply(categorize_broad_field)
    df_valid = df[df['Broad_Field'] != 'Invalid'].copy()

    # ---------------------------------------------------------
    # 2. 计算 2014-2025 的绝对人数与 YoY 增长率
    # ---------------------------------------------------------
    df_target_years = df_valid[(df_valid['T0_year'] >= 2014) & (df_valid['T0_year'] <= 2025)]
    yearly_counts = pd.crosstab(df_target_years['T0_year'], df_target_years['Broad_Field'])
    yoy_growth = yearly_counts.pct_change() * 100

    # ---------------------------------------------------------
    # 3. 绘制顶刊级别折线图 (双 Y 轴设计 + 负值自适应)
    # ---------------------------------------------------------
    plot_years = range(2015, 2026)
    hss_counts_plot = yearly_counts.loc[plot_years, 'Humanities & Social Sciences (HSS)']
    hss_growth_plot = yoy_growth.loc[plot_years, 'Humanities & Social Sciences (HSS)']
    cs_growth_plot = yoy_growth.loc[plot_years, 'Computer Science (CS)']

    fig, ax1 = plt.subplots(figsize=(12, 7))

    # --- 左轴：HSS 绝对入局人数 (标签已学术化) ---
    color_bar = '#E1E5F2'
    ax1.bar(plot_years, hss_counts_plot, color=color_bar, alpha=0.9, label='New HSS Authors')

    # 使用 LaTeX 格式的 $T_0$ 提升数学质感
    ax1.set_xlabel('Year of First Publication ($T_0$)', fontsize=14, fontweight='normal')
    ax1.set_ylabel('Number of New HSS Authors', fontsize=14, color='#5c6bc0', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#5c6bc0')
    ax1.set_xticks(plot_years)

    # --- 右轴：年增长率折线图 (标签已学术化) ---
    ax2 = ax1.twinx()

    color_hss_line = '#D62828'
    ax2.plot(plot_years, hss_growth_plot, color=color_hss_line, marker='o', linewidth=3, markersize=8,
             label='HSS Annual Growth Rate (%)')

    color_cs_line = '#8D99AE'
    ax2.plot(plot_years, cs_growth_plot, color=color_cs_line, marker='s', linestyle='--', linewidth=2, markersize=6,
             label='CS Annual Growth Rate (%)')

    ax2.set_ylabel('Annual Growth Rate (%)', fontsize=14, color='black', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='black')

    # ---------------------------------------------------------
    # 4. 🎯 核心修复：Y 轴自适应与零轴基准线
    # ---------------------------------------------------------
    max_growth = max(hss_growth_plot.max(), cs_growth_plot.max())
    min_growth = min(hss_growth_plot.min(), cs_growth_plot.min())

    y_min = min_growth * 1.2 if min_growth < 0 else 0
    ax2.set_ylim(y_min, max_growth * 1.35)

    ax2.axhline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.5)

    # ---------------------------------------------------------
    # 5. 精准锚定历史节点标注 (措辞已去情绪化)
    # ---------------------------------------------------------
    if 2017 in hss_growth_plot.index:
        val_2017 = hss_growth_plot.loc[2017]
        ax2.annotate(f'Transformer\nIntroduced\n+{val_2017:.1f}%',
                     xy=(2017, val_2017), xytext=(2017 - 0.2, val_2017 * 1.15),
                     arrowprops=dict(facecolor=color_cs_line, shrink=0.05, width=1.5, headwidth=6),
                     fontsize=11, fontweight='bold', color=color_cs_line, ha='center')

    if 2023 in hss_growth_plot.index:
        val_2023 = hss_growth_plot.loc[2023]
        # 去掉了主观的 "Shock"，改为客观事实
        ax2.annotate(f'ChatGPT\nReleased\n+{val_2023:.1f}%',
                     xy=(2023, val_2023), xytext=(2023 - 0.8, val_2023 * 1.15),
                     arrowprops=dict(facecolor=color_hss_line, shrink=0.05, width=2, headwidth=8),
                     fontsize=12, fontweight='bold', color=color_hss_line, ha='center')

    # ---------------------------------------------------------
    # 6. 图注合并居中
    # ---------------------------------------------------------
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc='upper center', bbox_to_anchor=(0.5, -0.12),
               ncol=3, frameon=False, fontsize=12)

    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()

    # 更改了输出文件名，以便和旧版本区分
    output_img = 'HSS_Awakening_Growth_Rate_Academic.png'
    plt.savefig(output_img, dpi=300, bbox_inches='tight')

    print(f"💾 学术版图表已保存为: {output_img}")
    plt.show()


# 引爆数据！
analyze_hss_growth_and_plot_ultimate()