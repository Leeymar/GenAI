import pandas as pd
import numpy as np
import json
import os
import statsmodels.formula.api as smf


def run_ultimate_regressions():
    print("🚀 启动【跨界溢价：中介机制与多元回归】终极实证引擎...\n")

    # ==========================================
    # 1. 读取三大数据源并无缝缝合
    # ==========================================
    print("🧩 1. 正在读取并融合数据源...")

    # A. 读取刚才生成的终极属性表 (包含 DNA 浓度和 arXiv 标签)
    dna_file = 'Ultimate_ArXiv_DNA_Master.csv'
    if not os.path.exists(dna_file):
        print(f"❌ 找不到 {dna_file}，请确认已运行上一步脚本。")
        return
    df_dna = pd.read_csv(dna_file)

    # B. 读取被引量映射字典
    citation_file = 'E:\PythonProject\GenAI\data_process\process\citation_mapping.json'
    if not os.path.exists(citation_file):
        print(f"❌ 找不到 {citation_file}。")
        return
    with open(citation_file, 'r', encoding='utf-8') as f:
        citation_mapping = json.load(f)

    # C. 将引用量拼进 DataFrame
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
    print(f"✅ 成功缝合！获得 {len(df)} 篇具备完整【基因+意图+引用】特征的有效样本。")

    if len(df) == 0: return

    # ==========================================
    # 2. 消除时间偏差：计算 RCR 和对数化
    # ==========================================
    print("🧮 2. 正在计算相对被引率 (RCR)...")
    yearly_avg = df.groupby('year')['citations'].mean().reset_index()
    yearly_avg.rename(columns={'citations': 'yearly_mean_citations'}, inplace=True)
    df = pd.merge(df, yearly_avg, on='year')

    df['RCR'] = df['citations'] / df['yearly_mean_citations'].replace(0, 1)
    df['log_RCR'] = np.log1p(df['RCR'])

    # 对控制变量（团队人数）取对数，经济学常规操作
    df['log_team_size'] = np.log1p(df['total_authors'])

    # ==========================================
    # 3. 强制分箱处理文科基因 (HSS DNA)
    # ==========================================
    print("🪓 3. 正在切分文科基因浓度阵营...")
    bins = [-0.1, 0.1, 20.0, 40.0, 60.0, 100.1]
    labels = ['1_Pure_STEM(0%)', '2_Low_HSS(1-20%)', '3_Moderate_HSS(21-40%)', '4_High_HSS(41-60%)',
              '5_HSS_Dominant(>60%)']
    df['HSS_Bin'] = pd.cut(df['team_hss_dna_pct'], bins=bins, labels=labels)

    # 清理一下 arXiv Type 的格式，方便 statsmodels 解析
    df['arxiv_category'] = df['arxiv_type'].apply(lambda x: str(x).split('(')[0].strip().replace(' ', '_'))

    print("\n" + "=" * 80)
    print("🏆 【顶级计量回归战报】")
    print("=" * 80)

    # ==========================================
    # 📊 回归模型 1：纯基因驱动力
    # ==========================================
    print("\n🟢 模型一：文科基因浓度是否直接拉升了知识溢价？ (Baseline Model)")
    formula_1 = "log_RCR ~ C(HSS_Bin, Treatment('1_Pure_STEM(0%)')) + log_team_size + C(year)"
    model_1 = smf.ols(formula=formula_1, data=df).fit()
    print(">> 核心看点：2_Low_HSS 到 5_HSS_Dominant 的系数 (coef) 是不是正的，P>|t| 是不是 <0.05。")
    print(model_1.summary().tables[1])

    # ==========================================
    # 📊 回归模型 2：跨界野心的溢价
    # ==========================================
    print("\n🟡 模型二：不管基因如何，只要敢在 arXiv 上贴社科标签，是否就有溢价？")
    # 剔除掉 Unknown 干扰项
    df_clean = df[df['arxiv_category'] != 'Unknown']
    formula_2 = "log_RCR ~ C(arxiv_category, Treatment('Type_0')) + log_team_size + C(year)"
    model_2 = smf.ols(formula=formula_2, data=df_clean).fit()
    print(">> 核心看点：Type_3 的 coef 是不是正的？它代表了‘社科跨界’带来的绝对溢价收益！")
    print(model_2.summary().tables[1])

    # ==========================================
    # 📊 回归模型 3：基因与野心的巅峰对决 (Full Model)
    # ==========================================
    print("\n🔴 模型三：终极对决！同时放入基因和标签，到底是谁说了算？")
    formula_3 = "log_RCR ~ C(arxiv_category, Treatment('Type_0')) + C(HSS_Bin, Treatment('1_Pure_STEM(0%)')) + log_team_size + C(year)"
    model_3 = smf.ols(formula=formula_3, data=df_clean).fit()
    print(">> 核心看点：当控制了文科基因后，Type_3 的系数依然显著吗？如果显著，说明‘跨界’本身就是一种极其稀缺的红利！")
    print(model_3.summary().tables[1])
    print("=" * 80)


if __name__ == "__main__":
    run_ultimate_regressions()