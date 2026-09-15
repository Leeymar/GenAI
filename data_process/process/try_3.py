import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_global_evolution():
    print("🚀 启动【全局全样本：跨学科溢价三阶段演化】渲染引擎...\n")

    # 1. 快速加载字典
    master_df = pd.read_csv('MASTER_AUTHOR_PROFILES_12YEARS.csv')
    id_to_dna = {}
    for _, row in master_df.iterrows():
        a_id = str(row['author_id']).strip()
        try:
            dna = json.loads(row['scopus_dna_weights'])
        except:
            dna = {}
        id_to_dna[a_id] = dna

    hss_fields = {'Social Sciences', 'Arts and Humanities', 'Psychology',
                  'Business, Management and Accounting', 'Economics, Econometrics and Finance', 'Decision Sciences'}

    with open('citation_mapping.json', 'r', encoding='utf-8') as f:
        citation_mapping = json.load(f)

    phase_files = {
        r'E:\PythonProject\GenAI\data_scrap_v2\stage_1\phase1_hybrid_cleaned.jsonl': 'Phase 1\n(Pre-Shock)',
        r'E:\PythonProject\GenAI\data_scrap_v2\stage_2\phase2_hybrid_cleaned.jsonl': 'Phase 2\n(Eruption)',
        r'E:\PythonProject\GenAI\data_scrap_v2\stage_3\phase3_hybrid_cleaned.jsonl': 'Phase 3\n(Democratization)'
    }

    # 2. 提取全局数据
    paper_records = []
    for path, phase_label in phase_files.items():
        if not os.path.exists(path): continue
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    work = json.loads(line)
                    raw_id = work.get('work_id') or work.get('id')
                    if not raw_id: continue
                    clean_id = str(raw_id).split('/')[-1]

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
                    hss_percentage = (hss_concentration / valid_authors) * 100
                    cite_info = citation_mapping[clean_id]

                    paper_records.append({
                        'work_id': clean_id,
                        'year': int(cite_info['year']),
                        'citations': int(cite_info['citations']),
                        'hss_concentration': hss_percentage,
                        'Phase': phase_label
                    })
                except Exception:
                    pass

    df = pd.DataFrame(paper_records)
    print(f"✅ 成功提取 {len(df)} 篇全局数据！")

    # 3. 计算 RCR 与分箱
    yearly_avg = df.groupby('year')['citations'].mean().reset_index()
    yearly_avg.rename(columns={'citations': 'yearly_mean_citations'}, inplace=True)
    df = pd.merge(df, yearly_avg, on='year')
    df['RCR'] = df['citations'] / df['yearly_mean_citations'].replace(0, 1)
    df['log_RCR'] = np.log1p(df['RCR'])

    bins = [-0.1, 0.1, 20.0, 40.0, 60.0, 100.1]
    labels = ['1_Pure_STEM(0%)', '2_Low_HSS(1-20%)', '3_Moderate_HSS(21-40%)', '4_High_HSS(41-60%)',
              '5_HSS_Dominant(>60%)']
    df['HSS_Bin'] = pd.cut(df['hss_concentration'], bins=bins, labels=labels)

    # 💡 核心改变：我们这次直接用 df (全样本) 画图，不提取 Top 10%
    print("🎨 4. 正在渲染【全局全样本】演化对比图...")
    plt.figure(figsize=(15, 8), dpi=300)
    sns.set_theme(style="whitegrid", context="talk")

    ax = sns.barplot(
        x='HSS_Bin',
        y='log_RCR',
        hue='Phase',
        data=df,  # 直接使用全量数据
        palette="magma",  # 换个颜色区分一下：用暖色调代表全局
        capsize=0.05,
        errcolor=".2",
        linewidth=1.5,
        edgecolor=".2"
    )

    clean_labels = [label.get_text().split('_', 1)[1].replace('_', '\n') for label in ax.get_xticklabels()]
    ax.set_xticklabels(clean_labels)

    # 标题注明这是 Global Average
    plt.title('The Evolution of Interdisciplinary Premium Across Three Eras of GenAI\n(Average Impact of ALL Papers)',
              fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('HSS Knowledge Concentration in Team DNA', fontsize=15, fontweight='bold')
    plt.ylabel('Average Academic Impact (Log-RCR)', fontsize=15, fontweight='bold')

    plt.legend(title='Evolution Phase', title_fontsize='14', loc='upper right', frameon=True)

    plt.tight_layout()
    output_img = 'Global_Average_Evolution.png'
    plt.savefig(output_img)
    print(f"💾 绝杀级【全局演化图】已保存为: {output_img}")
    plt.show()


if __name__ == "__main__":
    plot_global_evolution()