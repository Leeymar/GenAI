import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def analyze_disciplinary_composition():
    print("🌍 启动【学科构成演化与多样性指数】分析引擎...\n")

    file_path = 'MASTER_AUTHOR_PROFILES_12YEARS.csv'
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"❌ 找不到文件 {file_path}")
        return

    # 1. 清洗无效数据
    df_valid = df[~df['primary_origin_field'].isin(['Unknown', 'Error', 'Ghost'])].copy()

    # 2. 锁定 2014-2025 年的有效时间窗
    df_valid = df_valid[(df_valid['T0_year'] >= 2014) & (df_valid['T0_year'] <= 2025)]

    # 3. 提取 Top 6 核心学科，其余归为 Others
    top_fields = df_valid['primary_origin_field'].value_counts().nlargest(6).index.tolist()
    df_valid['Field_Cleaned'] = df_valid['primary_origin_field'].apply(
        lambda x: x if x in top_fields else 'Others'
    )

    # 4. 生成年度占比矩阵 (用于面积图)
    yearly_counts = pd.crosstab(df_valid['T0_year'], df_valid['Field_Cleaned'])
    yearly_pct = yearly_counts.div(yearly_counts.sum(axis=1), axis=0) * 100

    # 强制排序：CS 垫底，Others 置顶
    ordered_columns = [col for col in top_fields if col != 'Computer Science']
    if 'Computer Science' in yearly_pct.columns:
        ordered_columns.insert(0, 'Computer Science')
    ordered_columns.append('Others')
    yearly_pct = yearly_pct[ordered_columns]

    # 5. 计算每年的香农多样性指数 (Shannon Entropy)
    # 计算公式使用实际的占比 (0-1之间)
    yearly_prop = yearly_counts.div(yearly_counts.sum(axis=1), axis=0)

    def calculate_shannon(row):
        # 剔除 0 值以防 ln(0) 报错
        p = row[row > 0]
        return -np.sum(p * np.log(p))

    shannon_index = yearly_prop.apply(calculate_shannon, axis=1)

    print("================================================================")
    print(" 🧬 黄金十年 (2015-2025) 学科多样性熵值 (Shannon Index) 战报")
    print("================================================================")
    for year in range(2015, 2026):
        if year in shannon_index.index:
            print(f"   ➤ {year}年 多样性指数: {shannon_index.loc[year]:.4f} (数值越大，学科融合度越高)")
    print("================================================================\n")

    # 6. 绘制 100% 堆叠面积图 (双Y轴加入熵值曲线)
    print("🎨 正在绘制学科沉积岩面积图...")
    fig, ax1 = plt.subplots(figsize=(14, 8))

    colors = plt.cm.Set3(np.linspace(0, 1, len(ordered_columns)))
    years = yearly_pct.index

    # 绘制堆叠面积图
    ax1.stackplot(years, yearly_pct.T, labels=ordered_columns, colors=colors, alpha=0.85, edgecolor='white',
                  linewidth=0.5)

    ax1.set_xlabel('Year of Entry (T0)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Proportion of Disciplinary Composition (%)', fontsize=14, fontweight='bold')
    ax1.set_xlim(2014, 2025)
    ax1.set_ylim(0, 100)
    ax1.set_xticks(years)

    # 将图注放在上方外侧
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=4, frameon=False, fontsize=10)

    # 加入右侧 Y 轴：香农多样性指数曲线
    ax2 = ax1.twinx()
    ax2.plot(years, shannon_index, color='black', linewidth=3, marker='D', markersize=6,
             label='Shannon Diversity Index')
    ax2.set_ylabel('Shannon Diversity Index (Higher = More Diverse)', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, shannon_index.max() * 1.2)
    ax2.legend(loc='upper left', frameon=False)

    plt.title('Evolution of GenAI Disciplinary Composition & Diversity Entropy (2014-2025)', fontsize=16, pad=45,
              fontweight='bold')

    plt.tight_layout()
    output_img = 'Disciplinary_Composition_AreaChart.png'
    plt.savefig(output_img, dpi=300, bbox_inches='tight')

    print(f"💾 顶刊级堆叠面积图已高清保存为: {output_img}")
    plt.show()


analyze_disciplinary_composition()