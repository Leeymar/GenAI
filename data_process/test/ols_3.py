import pandas as pd
import numpy as np
import json
import os
import statsmodels.formula.api as smf


def run_phase_heterogeneity_analysis():
    print("🚀 启动【跨时代演进：AI 发展三阶段的异质性回归引擎】...\n")

    # ==========================================
    # 1. 读取并缝合数据
    # ==========================================
    dna_file = 'Ultimate_ArXiv_DNA_Master.csv'
    citation_file = 'E:\PythonProject\GenAI\data_process\process\citation_mapping.json'

    if not os.path.exists(dna_file) or not os.path.exists(citation_file):
        print("❌ 找不到基础数据文件，请确保 Ultimate_ArXiv_DNA_Master.csv 和 citation_mapping.json 存在。")
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
    if len(df) == 0: return

    # ==========================================
    # 2. 预处理：计算 RCR、对数化、分箱
    # ==========================================
    yearly_avg = df.groupby('year')['citations'].mean().reset_index()
    yearly_avg.rename(columns={'citations': 'yearly_mean_citations'}, inplace=True)
    df = pd.merge(df, yearly_avg, on='year')

    df['RCR'] = df['citations'] / df['yearly_mean_citations'].replace(0, 1)
    df['log_RCR'] = np.log1p(df['RCR'])
    df['log_team_size'] = np.log1p(df['total_authors'])

    bins = [-0.1, 0.1, 20.0, 40.0, 60.0, 100.1]
    labels = ['1_Pure_STEM(0%)', '2_Low_HSS(1-20%)', '3_Moderate_HSS(21-40%)', '4_High_HSS(41-60%)',
              '5_HSS_Dominant(>60%)']
    df['HSS_Bin'] = pd.cut(df['team_hss_dna_pct'], bins=bins, labels=labels)
    df['arxiv_category'] = df['arxiv_type'].apply(lambda x: str(x).split('(')[0].strip().replace(' ', '_'))

    # 剔除未知的标签干扰项
    df_clean = df[df['arxiv_category'] != 'Unknown']

    # ==========================================
    # 3. 分阶段动态回归循环
    # ==========================================
    # 确保你的 stage 列名字是 Phase1, Phase2, Phase3，如果有大小写差异请用 str.lower() 兼容
    phases = ['Phase1', 'Phase2', 'Phase3']

    print("=" * 90)
    print("🏆 【跨时代演进战报：文科基因的价值是如何觉醒的？】")
    print("=" * 90)

    for phase in phases:
        df_phase = df_clean[df_clean['stage'] == phase]
        if len(df_phase) < 50:  # 样本太少不回归
            print(f"\n⚠️ {phase} 样本量过少 ({len(df_phase)} 篇)，跳过回归。")
            continue

        print(f"\n" + "█" * 90)
        print(f"🕰️ 【 {phase} 时代回归对决 】 | 有效样本量: {len(df_phase)} 篇")
        print("█" * 90)

        # 终极模型公式（控制年份、控制团队规模）
        # 注意：由于是在一个较短的阶段内，如果年份单一可能会报错，所以 C(year) 依然要留着，statsmodels 会自动处理
        formula = "log_RCR ~ C(arxiv_category, Treatment('Type_0')) + C(HSS_Bin, Treatment('1_Pure_STEM(0%)')) + log_team_size + C(year)"

        try:
            model = smf.ols(formula=formula, data=df_phase).fit()
            print(model.summary().tables[1])

            # 自动提取核心结论
            print("-" * 90)
            print(f"💡 {phase} 快速诊断：")

            # 诊断标签溢价
            type3_pval = model.pvalues.get("C(arxiv_category, Treatment('Type_0'))[T.Type_3]", 1)
            type3_coef = model.params.get("C(arxiv_category, Treatment('Type_0'))[T.Type_3]", 0)
            if type3_pval < 0.05:
                print(
                    f"   ➤ 社科标签 (Type 3)：依然有显著溢价 (coef = {type3_coef:.4f})，说明在这个时代，'包装'还是有点用的。")
            else:
                print(f"   ➤ 社科标签 (Type 3)：毫无用处 (p = {type3_pval:.3f})！学术界已经看穿了虚假的包装。")

            # 诊断基因红利 (以最高浓度为例)
            gene_pval = model.pvalues.get("C(HSS_Bin, Treatment('1_Pure_STEM(0%)'))[T.5_HSS_Dominant(>60%)]", 1)
            gene_coef = model.params.get("C(HSS_Bin, Treatment('1_Pure_STEM(0%)'))[T.5_HSS_Dominant(>60%)]", 0)
            if gene_pval < 0.05:
                print(f"   ➤ 极高文科基因 (>60%)：拥有极其显著的真实溢价 (coef = {gene_coef:.4f})！")
            else:
                print(f"   ➤ 极高文科基因 (>60%)：未能带来显著溢价 (p = {gene_pval:.3f})，这个时代纯技术可能更吃香。")

        except Exception as e:
            print(f"❌ {phase} 回归失败，可能是由于虚拟变量完全多重共线性导致: {e}")


if __name__ == "__main__":
    run_phase_heterogeneity_analysis()