import pandas as pd
import numpy as np
import json
import os
import statsmodels.formula.api as smf


def run_discipline_fixed_effects():
    print("🚀 启动【学科固定效应版】跨学科溢价深度实证模型...\n")

    # 1. 加载数据
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

    # 2. 核心预处理
    yearly_avg = df.groupby('year')['citations'].mean().reset_index()
    yearly_avg.rename(columns={'citations': 'yearly_mean_citations'}, inplace=True)
    df = pd.merge(df, yearly_avg, on='year')

    df['log_RCR'] = np.log1p(df['citations'] / df['yearly_mean_citations'].replace(0, 1))
    df['log_team_size'] = np.log1p(df['total_authors'])

    # 🌟 关键修正：生成 HSS_Bin 变量
    bins = [-0.1, 0.1, 20.0, 40.0, 60.0, 100.1]
    labels = ['1_Pure_STEM(0%)', '2_Low_HSS(1-20%)', '3_Moderate_HSS(21-40%)', '4_High_HSS(41-60%)',
              '5_HSS_Dominant(>60%)']
    df['HSS_Bin'] = pd.cut(df['team_hss_dna_pct'], bins=bins, labels=labels)

    # 3. 🛡️ 顶刊级守门员过滤：强制纯正 CS 样本
    df = df[df['arxiv_primary_category'].str.startswith('cs.', na=False)].copy()

    # 4. 执行回归：引入学科固定效应 (Discipline Fixed Effects)
    print(f"📈 正在对 {len(df)} 篇论文执行学科固定效应回归...")

    # 公式：使用 C() 显式指定分类变量
    formula = "log_RCR ~ C(HSS_Bin, Treatment('1_Pure_STEM(0%)')) + C(arxiv_primary_category) + log_team_size + C(year)"

    try:
        model = smf.ols(formula=formula, data=df).fit()

        print("\n" + "=" * 80)
        print("📊 【学科固定效应模型结果】")
        print("=" * 80)

        # 仅提取 HSS_Bin 和关键控制变量的系数进行展示
        summary_table = model.summary().tables[1]
        print(f"{summary_table.data[0][0]:<45} {summary_table.data[0][1]:>10} {summary_table.data[0][4]:>10}")
        print("-" * 75)
        for row in summary_table.data[1:]:
            name = row[0]
            # 过滤：只打印我们需要展示的核心行，隐藏那几十个学科固定效应的行
            if "HSS_Bin" in name or "log_team_size" in name or "Intercept" in name:
                print(f"{name:<45} {row[1]:>10} {row[4]:>10}")
        print("-" * 75)
        print("*(注：arxiv_primary_category 与 year 已作为固定效应被模型严格控制)*")

    except Exception as e:
        print(f"❌ 回归计算失败: {e}")


if __name__ == "__main__":
    run_discipline_fixed_effects()