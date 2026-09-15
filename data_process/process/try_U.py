import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf


def analyze_and_plot_rcr_premium():
    print("🚀 启动【跨学科溢价与知识扩散规律】终极分析引擎...\n")

    # ==========================================
    # 1. 加载所有基础字典
    # ==========================================
    print("🧬 1. 正在加载 12 年全量作者的 DNA 图谱...")
    master_df = pd.read_csv('MASTER_AUTHOR_PROFILES_12YEARS.csv')

    id_to_dna = {}
    for _, row in master_df.iterrows():
        a_id = str(row['author_id']).strip()
        try:
            dna = json.loads(row['scopus_dna_weights'])
        except:
            dna = {}
        id_to_dna[a_id] = dna

    hss_fields = {
        'Social Sciences', 'Arts and Humanities', 'Psychology',
        'Business, Management and Accounting', 'Economics, Econometrics and Finance', 'Decision Sciences'
    }

    print("📚 2. 正在加载云端抓取的最新被引量数据...")
    citation_file = 'citation_mapping.json'
    if not os.path.exists(citation_file):
        print("❌ 找不到 citation_mapping.json 文件，请确认是否已抓取完毕！")
        return

    with open(citation_file, 'r', encoding='utf-8') as f:
        citation_mapping = json.load(f)

    # ==========================================
    # 2. 核心大缝合：提取浓度并匹配被引量
    # ==========================================
    phase_files = [
        'E:\PythonProject\GenAI\data_scrap_v2\stage_1\phase1_hybrid_cleaned.jsonl',
        'E:\PythonProject\GenAI\data_scrap_v2\stage_2\phase2_hybrid_cleaned.jsonl',
        'E:\PythonProject\GenAI\data_scrap_v2\stage_3\phase3_hybrid_cleaned.jsonl'
    ]

    print("🧩 3. 正在逐篇计算 Team DNA 并匹配被引量...")
    paper_records = []

    for path in phase_files:
        if not os.path.exists(path): continue
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    work = json.loads(line)

                    # 提取论文 ID 并清洗
                    raw_id = work.get('work_id') or work.get('id')
                    if not raw_id: continue
                    clean_id = str(raw_id).split('/')[-1]

                    # 💡 关键卡点：这篇论文必须有被引量数据！
                    if clean_id not in citation_mapping:
                        continue

                    # 提取作者计算 Team DNA
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

                    # 计算 HSS (文科) 浓度百分比
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
    print(f"✅ 完美缝合！共提取到 {len(df)} 篇拥有完整【基因图谱+被引量】的论文。")

    if len(df) == 0:
        print("❌ 数据为空，请检查文件路径或数据格式。")
        return

    # ==========================================
    # 3. 计算相对被引率 (RCR) 并进行对数平滑
    # ==========================================
    print("🧮 4. 正在剥离时间偏差，计算 RCR (相对被引率)...")

    # 计算每年全行业的平均被引量 (Baseline)
    yearly_avg = df.groupby('year')['citations'].mean().reset_index()
    yearly_avg.rename(columns={'citations': 'yearly_mean_citations'}, inplace=True)

    # 匹配回原表并算 RCR
    df = pd.merge(df, yearly_avg, on='year')
    df['RCR'] = df['citations'] / df['yearly_mean_citations'].replace(0, 1)  # 防除以0

    # 核心：使用 Log1p 拉平幂律分布（长尾极值）
    df['log_RCR'] = np.log1p(df['RCR'])

    # ==========================================
    # 4. 二次多项式回归与图表渲染
    # ==========================================
    print("📈 5. 正在执行二次非线性回归验证规律...")

    # 跑 OLS 回归：看看平方项是否显著为负
    model = smf.ols(formula='log_RCR ~ hss_concentration + I(hss_concentration**2)', data=df).fit()

    print("\n" + "=" * 55)
    print("📊 顶刊级回归模型结果 (重点关注 I(hss_concentration ** 2) 的 P 值):")
    print(model.summary().tables[1])
    print("=" * 55 + "\n")

    print("🎨 6. 正在渲染【知识扩散新规律】倒 U 型黄金曲线图...")
    plt.figure(figsize=(10, 6), dpi=300)
    sns.set_theme(style="whitegrid", context="talk")

    # 画出带有 95% 置信区间的二次拟合曲线
    ax = sns.regplot(
        x='hss_concentration',
        y='log_RCR',
        data=df,
        order=2,  # 强制执行二次拟合！
        scatter_kws={'alpha': 0.05, 's': 8, 'color': '#A9C3DE', 'edgecolors': 'none'},
        line_kws={'color': '#D62828', 'linewidth': 3.5}
    )

    # 求解抛物线顶点坐标：x = -b / (2a)
    params = model.params
    b = params['hss_concentration']
    a = params['I(hss_concentration ** 2)']

    if a < 0:
        optimal_x = -b / (2 * a)

        # 画一条垂直辅助线标出巅峰位置
        plt.axvline(x=optimal_x, color='#003049', linestyle='--', linewidth=2, zorder=0)

        # 在图中标注黄金比例
        plt.text(
            optimal_x + 2, df['log_RCR'].quantile(0.95),
            f'Optimal HSS DNA:\n{optimal_x:.1f}%',
            color='#003049', fontweight='bold', fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.5')
        )
        print(f"👑 测算完毕！全场最高影响力的跨界黄金比例为: {optimal_x:.2f}%")
    else:
        print("⚠️ 注意：模型拟合出的曲线开口向上，不是倒 U 型。可能是散点分布原因。")

    # 美化图表元素
    plt.title('Knowledge Diffusion Paradigm:\nThe Inverted-U Relationship between Team DNA and Citation Impact',
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('HSS Knowledge Concentration in Team DNA (%)', fontsize=13, fontweight='bold')
    plt.ylabel('Academic Impact (Log-RCR)', fontsize=13, fontweight='bold')

    # 严格限制 X 轴的物理意义范围
    plt.xlim(0, 100)

    plt.tight_layout()
    output_img = 'Inverted_U_Knowledge_Diffusion.png'
    plt.savefig(output_img)
    print(f"💾 绝杀级被引量图表已高清保存为: {output_img}")
    plt.show()


if __name__ == "__main__":
    analyze_and_plot_rcr_premium()