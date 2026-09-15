import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


def analyze_disciplinary_diffusion():
    print("🌊 启动【学科扩散与知识溢出 (Disciplinary Diffusion)】追踪引擎...")

    # ==========================================
    # 1. 加载主表，构建 work_id 到 HSS 浓度梯队的映射
    # ==========================================
    csv_path = 'Ultimate_Regression_Base.csv'
    try:
        df_base = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"❌ 找不到文件 {csv_path}")
        return

    # 剔除缺失值
    df_base = df_base.dropna(subset=['work_id', 'team_hss_dna_pct'])

    # 将 HSS 浓度分为 5 个极其严谨的学术梯队 (与回归模型完全一致)
    # 假设你的 team_hss_dna_pct 是 0-100 的数值。如果是 0-1，请自行乘以 100。
    def categorize_hss(pct):
        if pct == 0:
            return '0% (Pure STEM)'
        elif 0 < pct <= 20:
            return '1%-20% (Low)'
        elif 20 < pct <= 40:
            return '21%-40% (Moderate)'
        elif 40 < pct <= 60:
            return '41%-60% (High)'
        else:
            return '>60% (Dominant)'

    df_base['HSS_Group'] = df_base['team_hss_dna_pct'].apply(categorize_hss)

    # 制作一个极其快速的查表字典 {work_id: HSS_Group}
    wid_to_group = dict(zip(df_base['work_id'], df_base['HSS_Group']))
    print(f"✅ 成功映射 {len(wid_to_group)} 篇 arXiv 论文的浓度梯队。")

    # ==========================================
    # 2. 逐行解析巨型 JSONL 被引数据并进行归集
    # ==========================================
    jsonl_path = r'E:\PythonProject\GenAI\data_process\process\citing_disciplines_master.jsonl'

    # 数据结构: group_citation_counts['41%-60% (High)']['Psychology'] = 累计被引次数
    group_citation_counts = defaultdict(lambda: defaultdict(int))

    print("📖 正在逐行读取 JSONL 并追踪十万级论文的知识溢出轨迹 (请稍候)...")
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                wid = data.get('work_id')

                # 如果这篇论文在我们的回归样本库里
                if wid in wid_to_group:
                    group = wid_to_group[wid]
                    citing_dict = data.get('citing_disciplines', {})

                    for discipline, count in citing_dict.items():
                        # 清洗一下学科名称，确保首字母大写等格式统一
                        clean_disc = str(discipline).strip().title()
                        group_citation_counts[group][clean_disc] += count
    except FileNotFoundError:
        print(f"❌ 找不到文件 {jsonl_path}")
        return

    # ==========================================
    # 3. 数据透视与百分比归一化 (核心逻辑)
    # ==========================================
    # 转换成 DataFrame (行是学科，列是浓度梯队)
    df_matrix = pd.DataFrame(group_citation_counts).fillna(0)

    # 为了图表美观，我们过滤掉那些微不足道的长尾学科，只保留总被引量排名前 N 的学科
    # 强烈建议保留 15-20 个核心学科，太少了看不出交叉，太多了图会糊
    top_disciplines = df_matrix.sum(axis=1).sort_values(ascending=False).head(15).index
    df_matrix = df_matrix.loc[top_disciplines]

    # 转置一下：行是浓度梯队，列是学科，这样符合阅读习惯
    df_matrix = df_matrix.T

    # 🚨 极其关键：按行求百分比 (Row-wise Normalization)
    # 解释：对于 ">60% (Dominant)" 这个梯队，它的所有下游被引加起来算作 100%
    # 看看这 100% 里面，有多少分给了 Computer Science，多少分给了 Psychology
    df_norm = df_matrix.div(df_matrix.sum(axis=1), axis=0) * 100

    # 强制排序梯队，确保图表的 Y 轴是从 Pure STEM 到 Dominant 递增的
    group_order = ['0% (Pure STEM)', '1%-20% (Low)', '21%-40% (Moderate)', '41%-60% (High)', '>60% (Dominant)']
    # 确保只包含存在的组别
    group_order = [g for g in group_order if g in df_norm.index]
    df_norm = df_norm.reindex(group_order)

    # ==========================================
    # 4. 绘制顶刊级学术热力图 (Heatmap)
    # ==========================================
    print("🎨 正在生成归一化扩散热力图...")
    fig, ax = plt.subplots(figsize=(14, 6))

    # 使用 cmap='YlGnBu' (黄绿蓝) 或者是 'Reds'，这是极其专业的学术配色
    # annot=True 会把具体的百分比数字写在格子里，fmt=".1f" 保留一位小数
    sns.heatmap(df_norm, cmap='YlGnBu', annot=True, fmt=".1f", linewidths=.5, ax=ax,
                cbar_kws={'label': 'Proportion of Downstream Citations (%)'})

    # 图表美化 (绝对不要写主标题)
    ax.set_ylabel('Team HSS DNA Concentration', fontsize=13, fontweight='bold')
    ax.set_xlabel('Downstream Citing Disciplines', fontsize=13, fontweight='bold')

    # 把 X 轴的学科名字稍微倾斜，防止挤在一起
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)

    plt.tight_layout()
    output_img = 'Disciplinary_Diffusion_Heatmap.png'
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"✅ 绝杀扩散热力图已保存: {output_img}")
    plt.show()


# 引爆知识溢出引擎！
analyze_disciplinary_diffusion()