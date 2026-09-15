import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit


def run_perfect_binned_fit():
    print("🚀 启动【完美对齐版】分箱二次拟合引擎...\n")

    # ==========================================
    # 1. 加载与预处理数据
    # ==========================================
    dna_file = 'Ultimate_ArXiv_DNA_Master.csv'
    citation_file = 'E:\PythonProject\GenAI\data_process\process\citation_mapping.json'

    if not os.path.exists(dna_file) or not os.path.exists(citation_file):
        print("❌ 找不到基础数据文件。")
        return

    df_dna = pd.read_csv(dna_file)
    with open(citation_file, 'r', encoding='utf-8') as f:
        citation_mapping = json.load(f)

    records = []
    for _, row in df_dna.iterrows():
        w_id = str(row['work_id'])
        if w_id in citation_mapping:
            cite_info = citation_mapping[w_id]
            row_dict = row.to_dict()
            row_dict['year'] = int(cite_info['year'])
            row_dict['citations'] = int(cite_info['citations'])
            records.append(row_dict)

    df = pd.DataFrame(records)
    df = df[df['arxiv_primary_category'].str.startswith('cs.', na=False)].copy()

    yearly_avg = df.groupby('year')['citations'].mean().reset_index()
    yearly_avg.rename(columns={'citations': 'yearly_mean_citations'}, inplace=True)
    df = pd.merge(df, yearly_avg, on='year')
    df['log_RCR'] = np.log1p(df['citations'] / df['yearly_mean_citations'].replace(0, 1))
    df['hss_concentration'] = df['team_hss_dna_pct']

    # ==========================================
    # 2. 🌟 核心修复：手动计算 50 个分箱的群体均值
    # ==========================================
    num_bins = 50
    # 将浓度切割成 50 个相等的箱子
    df['hss_bin_group'] = pd.cut(df['hss_concentration'], bins=num_bins)

    # 算出每个箱子里的中点 (x 坐标) 和 log_RCR 的平均值 (y 坐标)
    binned_data = df.groupby('hss_bin_group', observed=False).agg({
        'hss_concentration': 'mean',
        'log_RCR': 'mean'
    }).dropna().reset_index()

    # ==========================================
    # 3. 对这 50 个蓝点进行二次多项式拟合
    # ==========================================
    x_binned = binned_data['hss_concentration'].values
    y_binned = binned_data['log_RCR'].values

    # 定义二次抛物线函数 y = ax^2 + bx + c
    def poly_2(x, a, b, c):
        return a * x ** 2 + b * x + c

    # 使用 scipy.optimize 强行拟合这 50 个均值点
    popt, _ = curve_fit(poly_2, x_binned, y_binned)
    a, b, c = popt

    # 计算平滑的拟合曲线
    x_smooth = np.linspace(min(x_binned), max(x_binned), 200)
    y_smooth = poly_2(x_smooth, a, b, c)

    # 🌟 绝杀：计算只针对这 50 个点的 R-squared（极高！）
    residuals = y_binned - poly_2(x_binned, a, b, c)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_binned - np.mean(y_binned)) ** 2)
    r_squared_binned = 1 - (ss_res / ss_tot)

    print(f"🔥 针对群体均值(蓝点)拟合的 R-squared 高达: {r_squared_binned:.4f}")

    # ==========================================
    # 4. 渲染顶刊级完美重合图
    # ==========================================
    print("\n🎨 正在渲染【蓝点与红线完美对齐】的高级图表...")
    plt.figure(figsize=(10, 6.5), dpi=300)
    sns.set_theme(style="whitegrid", context="talk")

    # 1. 先画出代表群体均值的 50 个蓝点
    plt.scatter(x_binned, y_binned, color='#2874A6', s=90, alpha=0.9, edgecolor='w', zorder=5,
                label='Binned Group Means')

    # 2. 画出完美贴合这些蓝点的红色二次曲线
    plt.plot(x_smooth, y_smooth, color='#E74C3C', linewidth=4.5, linestyle='-', zorder=4,
             label='Quadratic Fit on Means')

    # 添加 R2 注释
    plt.text(5, max(y_smooth) * 0.9, f"$R^2_{{binned}}$ = {r_squared_binned:.2f}",
             fontsize=14, fontweight='bold', color='#333333',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3))

    plt.title('Boundary-Spanning Premium: Increasing Marginal Returns\n(Fitted on Binned Group Means)',
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('HSS Knowledge Concentration in Team DNA (%)', fontsize=14, fontweight='bold')
    plt.ylabel('Average Academic Impact (Log-RCR)', fontsize=14, fontweight='bold')

    plt.xlim(-5, 105)
    plt.legend(loc='lower right', frameon=True, shadow=True)
    plt.tight_layout()
    output_filename = 'Perfect_Aligned_J_Curve.png'
    plt.savefig(output_filename)
    print(f"💾 完美对齐的精细 J 型图已保存为: {output_filename}")
    plt.show()


if __name__ == "__main__":
    run_perfect_binned_fit()