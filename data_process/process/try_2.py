import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf


def run_global_analysis():
    print("🚀 启动【1. 全局大盘：演化图与交互回归】引擎...\n")

    # 1. 加载字典
    master_df = pd.read_csv('MASTER_AUTHOR_PROFILES_12YEARS.csv')
    id_to_dna = {str(row['author_id']).strip(): (
        json.loads(row['scopus_dna_weights']) if pd.notna(row['scopus_dna_weights']) else {}) for _, row in
                 master_df.iterrows()}

    hss_fields = {'Social Sciences', 'Arts and Humanities', 'Psychology', 'Business, Management and Accounting',
                  'Economics, Econometrics and Finance', 'Decision Sciences'}

    with open('citation_mapping.json', 'r', encoding='utf-8') as f:
        citation_mapping = json.load(f)

    phase_files = {
        r'E:\PythonProject\GenAI\data_scrap_v2\stage_1\phase1_hybrid_cleaned.jsonl': 'Phase 1\n(Pre-Shock)',
        r'E:\PythonProject\GenAI\data_scrap_v2\stage_2\phase2_hybrid_cleaned.jsonl': 'Phase 2\n(Eruption)',
        r'E:\PythonProject\GenAI\data_scrap_v2\stage_3\phase3_hybrid_cleaned.jsonl': 'Phase 3\n(Democratization)'
    }

    # 2. 提取数据
    paper_records = []
    for path, phase_label in phase_files.items():
        if not os.path.exists(path): continue
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    work = json.loads(line)
                    clean_id = str(work.get('work_id') or work.get('id')).split('/')[-1]
                    if clean_id not in citation_mapping: continue

                    a_ids = [str(aid).strip() for aid in work.get('author_ids', []) if aid]
                    if not a_ids: continue

                    team_dna_sum = {}
                    valid_authors = 0
                    for aid in a_ids:
                        author_dna = id_to_dna.get(aid, {})
                        if author_dna:
                            valid_authors += 1
                            for field, weight in author_dna.items():
                                team_dna_sum[field] = team_dna_sum.get(field, 0.0) + weight

                    if valid_authors == 0: continue

                    hss_concentration = sum(team_dna_sum.get(field, 0.0) for field in hss_fields)
                    paper_records.append({
                        'work_id': clean_id,
                        'year': int(citation_mapping[clean_id]['year']),
                        'citations': int(citation_mapping[clean_id]['citations']),
                        'hss_concentration': (hss_concentration / valid_authors) * 100,
                        'Phase': phase_label
                    })
                except:
                    pass

    df = pd.DataFrame(paper_records)
    if len(df) == 0: return

    # 3. 计算 RCR 与分箱
    yearly_avg = df.groupby('year')['citations'].mean().reset_index().rename(
        columns={'citations': 'yearly_mean_citations'})
    df = pd.merge(df, yearly_avg, on='year')
    df['RCR'] = df['citations'] / df['yearly_mean_citations'].replace(0, 1)
    df['log_RCR'] = np.log1p(df['RCR'])

    bins = [-0.1, 0.1, 20.0, 40.0, 60.0, 100.1]
    labels = ['1_Pure_STEM(0%)', '2_Low_HSS(1-20%)', '3_Moderate_HSS(21-40%)', '4_High_HSS(41-60%)',
              '5_HSS_Dominant(>60%)']
    df['HSS_Bin'] = pd.cut(df['hss_concentration'], bins=bins, labels=labels)

    # ==========================================
    # 🌟 新增：全局交互项回归 (Interaction Regression)
    # ==========================================
    print("\n" + "=" * 80)
    print("📊 【全局大盘】双因素交互回归表 (log_RCR ~ HSS_Bin * Phase):")
    # 使用 * 号代表同时测试主效应和交互效应
    formula = "log_RCR ~ C(HSS_Bin, Treatment('1_Pure_STEM(0%)')) * C(Phase, Treatment('Phase 1\\n(Pre-Shock)'))"
    global_model = smf.ols(formula=formula, data=df).fit()
    print(global_model.summary().tables[1])
    print("=" * 80 + "\n")

    # 4. 作图
    plt.figure(figsize=(15, 8), dpi=300)
    sns.set_theme(style="whitegrid", context="talk")
    ax = sns.barplot(x='HSS_Bin', y='log_RCR', hue='Phase', data=df, palette="magma", capsize=0.05, errcolor=".2",
                     linewidth=1.5, edgecolor=".2")

    clean_labels = [label.get_text().split('_', 1)[1].replace('_', '\n') for label in ax.get_xticklabels()]
    ax.set_xticklabels(clean_labels)
    plt.title('Global Average: Evolution of Interdisciplinary Impact Across Three Eras', fontsize=18, fontweight='bold',
              pad=20)
    plt.xlabel('HSS Knowledge Concentration in Team DNA', fontsize=15, fontweight='bold')
    plt.ylabel('Average Academic Impact (Log-RCR)', fontsize=15, fontweight='bold')
    plt.legend(title='Evolution Phase', loc='upper right', frameon=True)
    plt.tight_layout()
    plt.savefig('Global_Average_with_Regression.png')
    print("💾 全局大盘分析完成！图表已保存。")
    plt.show()


if __name__ == "__main__":
    run_global_analysis()