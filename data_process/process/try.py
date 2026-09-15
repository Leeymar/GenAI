import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf


def analyze_premium_and_plot():
    print("🚀 启动【跨学科溢价：分类回归 + 头部萃取】终极引擎...\n")

    # ==========================================
    # 1. 加载字典：DNA 与 被引量
    # ==========================================
    print("🧬 1. 正在加载 12 年全量作者 DNA 与 被引量数据...")

    # 确信你有这个文件
    master_df = pd.read_csv('MASTER_AUTHOR_PROFILES_12YEARS.csv')
    id_to_dna = {}
    for _, row in master_df.iterrows():
        a_id = str(row['author_id']).strip()
        try:
            dna = json.loads(row['scopus_dna_weights'])
        except:
            dna = {}
        id_to_dna[a_id] = dna

    # 文科学科定义
    hss_fields = {
        'Social Sciences', 'Arts and Humanities', 'Psychology',
        'Business, Management and Accounting', 'Economics, Econometrics and Finance', 'Decision Sciences'
    }

    citation_file = 'citation_mapping.json'
    if not os.path.exists(citation_file):
        print("❌ 找不到 citation_mapping.json，请确认被引量已抓取！")
        return

    with open(citation_file, 'r', encoding='utf-8') as f:
        citation_mapping = json.load(f)

    # 如果你的路径不同，请在这里修改
    phase_files = [
    'E:\PythonProject\GenAI\data_scrap_v2\stage_1\phase1_hybrid_cleaned.jsonl',
    'E:\PythonProject\GenAI\data_scrap_v2\stage_2\phase2_hybrid_cleaned.jsonl',
    'E:\PythonProject\GenAI\data_scrap_v2\stage_3\phase3_hybrid_cleaned.jsonl'
    ]

    # ==========================================
    # 2. 计算每篇论文的 Team DNA
    # ==========================================
    print("🧩 2. 正在提取论文浓度并对齐被引量...")
    paper_records = []

    for path in phase_files:
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
                        'hss_concentration': hss_percentage
                    })
                except Exception:
                    pass

    df = pd.DataFrame(paper_records)
    print(f"✅ 成功提取 {len(df)} 篇有效论文数据。")
    if len(df) == 0: return

    # ==========================================
    # 3. 剥离时间偏差：计算 RCR (相对被引率)
    # ==========================================
    print("🧮 3. 正在消除时间偏差，计算同年代相对被引率 (RCR)...")
    yearly_avg = df.groupby('year')['citations'].mean().reset_index()
    yearly_avg.rename(columns={'citations': 'yearly_mean_citations'}, inplace=True)
    df = pd.merge(df, yearly_avg, on='year')

    # 核心：计算 RCR 并对数平滑
    df['RCR'] = df['citations'] / df['yearly_mean_citations'].replace(0, 1)
    df['log_RCR'] = np.log1p(df['RCR'])

    # ==========================================
    # 4. 强制分箱 (Binning)
    # ==========================================
    print("🪓 4. 正在将文科浓度切分为 5 大阵营...")
    bins = [-0.1, 0.1, 20.0, 40.0, 60.0, 100.1]
    # 我们故意在标签前面加上 1_ 2_ 3_，是为了让回归模型和画图时严格按顺序排列！
    labels = ['1_Pure_STEM(0%)', '2_Low_HSS(1-20%)', '3_Moderate_HSS(21-40%)', '4_High_HSS(41-60%)',
              '5_HSS_Dominant(>60%)']
    df['HSS_Bin'] = pd.cut(df['hss_concentration'], bins=bins, labels=labels)

    # ==========================================
    # 5. 顶刊级统计检验：全量分类回归 (ANOVA)
    # ==========================================
    print("\n" + "=" * 70)
    print("📈 5. 正在执行全量数据分类回归 (Categorical OLS Regression)...")
    # 使用 C(HSS_Bin, Treatment('1_Pure_STEM(0%)')) 明确告诉模型：以纯理科作为对比基准 (0点)
    formula = "log_RCR ~ C(HSS_Bin, Treatment('1_Pure_STEM(0%)'))"
    model = smf.ols(formula=formula, data=df).fit()

    print("📊 跨学科溢价回归模型结果 (重点看 P>|t| 和 coef):")
    print(model.summary().tables[1])
    print("=" * 70 + "\n")

    # ==========================================
    # 6. 图表视觉抢救：萃取各阵营 Top 10% 头部神作
    # ==========================================
    print("👑 6. 正在过滤底层炮灰，提取各阵营 Top 10% 的高被引头部神作用于作图...")
    top_10_percent_list = []

    for hss_bin in labels:
        bin_data = df[df['HSS_Bin'] == hss_bin]
        if len(bin_data) > 0:
            threshold = bin_data['log_RCR'].quantile(0.90)  # 90分位数
            top_papers = bin_data[bin_data['log_RCR'] >= threshold]
            top_10_percent_list.append(top_papers)

    top_df = pd.concat(top_10_percent_list)

    # ==========================================
    # 7. 渲染终极溢价柱状图
    # ==========================================
    print("🎨 7. 正在生成高颜值溢价对比图...")
    plt.figure(figsize=(12, 7), dpi=300)
    sns.set_theme(style="whitegrid", context="talk")

    # 绘制带误差棒的柱状图
    ax = sns.barplot(
        x='HSS_Bin',
        y='log_RCR',
        data=top_df,
        palette="coolwarm",
        capsize=0.1,
        errcolor=".2",
        linewidth=1.5,
        edgecolor=".2"
    )

    # 替换 X 轴标签，把前面加的 1_ 2_ 3_ 删掉，让图表更好看
    clean_labels = [label.get_text().split('_', 1)[1].replace('_', ' ') for label in ax.get_xticklabels()]
    ax.set_xticklabels(clean_labels)

    plt.title('Interdisciplinary Premium of the Academic Elite:\nAverage Impact of Top 10% Papers by Team DNA',
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('HSS Knowledge Concentration in Team DNA', fontsize=14, fontweight='bold')
    plt.ylabel('Academic Impact of Top 10% Papers (Log-RCR)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    output_img = 'Final_Premium_Analysis.png'
    plt.savefig(output_img)
    print(f"💾 分析完毕！回归表已在控制台打印，终极图表已保存为: {output_img}")
    plt.show()


if __name__ == "__main__":
    analyze_premium_and_plot()