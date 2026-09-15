import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf


def plot_pure_cs_j_curve():
    print("🚀 启动【纯正 CS 样本：J型曲棍球棍效应】终极渲染引擎...\n")

    # ==========================================
    # 1. 数据加载与缝合
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

    # ==========================================
    # 2. 🛡️ 顶刊级守门员过滤：必须是纯正 CS 论文！
    # ==========================================
    initial_len = len(df)
    # 筛选出 arxiv_primary_category 以 cs. 开头的论文
    df = df[df['arxiv_primary_category'].str.startswith('cs.', na=False)].copy()
    print(f"🛡️ 极其严苛的学科过滤：剔除边缘学科，有效纯正 CS 样本从 {initial_len} 篇精简至 {len(df)} 篇！")

    # ==========================================
    # 3. 计算 RCR
    # ==========================================
    yearly_avg = df.groupby('year')['citations'].mean().reset_index()
    yearly_avg.rename(columns={'citations': 'yearly_mean_citations'}, inplace=True)

    df = pd.merge(df, yearly_avg, on='year')
    df['RCR'] = df['citations'] / df['yearly_mean_citations'].replace(0, 1)
    df['log_RCR'] = np.log1p(df['RCR'])
    df['hss_concentration'] = df['team_hss_dna_pct']

    # ==========================================
    # 4. 渲染极具视觉张力的加速曲线图
    # ==========================================
    print("🎨 正在渲染【边际收益递增】红色渐变加速图...")
    plt.figure(figsize=(11, 7.5), dpi=300)
    sns.set_theme(style="whitegrid", context="talk")

    # 提取实际的回归参数用于画精准的二次曲线
    model = smf.ols(formula='log_RCR ~ hss_concentration + I(hss_concentration**2)', data=df).fit()
    x_vals = np.linspace(0, 100, 500)
    y_vals = model.params['Intercept'] + model.params['hss_concentration'] * x_vals + model.params[
        'I(hss_concentration ** 2)'] * (x_vals ** 2)

    # 绘制背景散点 (极度降低存在感)
    sns.scatterplot(x='hss_concentration', y='log_RCR', data=df, alpha=0.04, s=15, color='#4a90e2', edgecolor='none')

    # 绘制核心主线
    plt.plot(x_vals, y_vals, color='#D62828', linewidth=4.5, label='Quadratic Fit (Increasing Returns)')

    # 填充加速起飞区域的红色渐变阴影
    plt.fill_between(x_vals, y_vals, y_vals.min(), color='#D62828', alpha=0.15)

    # 图表标题与坐标轴美化
    plt.title('The "Hockey Stick" Effect in Pure AI Research:\nAccelerating Premium for HSS Knowledge Concentration',
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('HSS Knowledge Concentration in Team DNA (%)', fontsize=14, fontweight='bold')
    plt.ylabel('Time-Normalized Academic Impact (Log-RCR)', fontsize=14, fontweight='bold')

    # 动态收紧 Y 轴，放大“一飞冲天”的视觉幅度
    plt.ylim(y_vals.min() - 0.05, y_vals.max() + 0.1)
    plt.xlim(-2, 102)
    plt.tight_layout()

    output_filename = 'Pure_CS_J_Curve_Acceleration.png'
    plt.savefig(output_filename)
    print(f"💾 极具震慑力的纯正CS版本 J型加速图 已保存为: {output_filename}")
    plt.show()


if __name__ == "__main__":
    plot_pure_cs_j_curve()