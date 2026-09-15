import json
import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

# ================= 配置区 =================
# 1. 论文元数据 (你的三份原始数据)
PAPER_FILES = [
    r'E:\PythonProject\GenAI\data_scrap_v2\stage_1\phase1_hybrid_cleaned.jsonl',
    r'E:\PythonProject\GenAI\data_scrap_v2\stage_2\phase2_hybrid_cleaned.jsonl',
    r'E:\PythonProject\GenAI\data_scrap_v2\stage_3\phase3_hybrid_cleaned.jsonl'
]

# 2. 引用数据字典
CITATION_FILE = r'E:\PythonProject\GenAI\data_process\process\citation_mapping.json'

# 3. 你上一阶段做好的精英作者名单
FINAL_AUTHORS_CSV = r'Final_Author_Metadata_for_DID.csv'

# 4. 最终输出的 DID 面板数据！
OUTPUT_PANEL = r'Ultimate_DID_Panel.csv'


# ==========================================

def extract_id(url):
    """辅助函数：将 'https://openalex.org/W2556467266' 提取为纯净的 'W2556467266'"""
    if pd.isna(url): return None
    return str(url).split('/')[-1]


def build_did_panel():
    print("1. 📖 正在加载全局引用字典 (Citation Mapping)...")
    with open(CITATION_FILE, 'r', encoding='utf-8') as f:
        cit_mapping = json.load(f)

    # 将字典转为快速查询格式: {"W123": 10, "W456": 0}
    # 注意：citation_mapping 里的 key 已经是 W 开头了
    cit_dict = {k: v.get('citations', 0) for k, v in cit_mapping.items()}
    print(f"   ✅ 成功加载 {len(cit_dict):,} 条论文引用记录。")

    print("\n2. 📚 正在遍历原始论文库，提取作者与引用关系...")
    paper_records = []

    for file_path in PAPER_FILES:
        print(f"   ▶ 处理文件: {os.path.basename(file_path)}")
        if not os.path.exists(file_path):
            print(f"   ❌ 找不到文件: {file_path}")
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                raw_work_id = data.get('work_id')
                year = data.get('publication_year')
                authors = data.get('author_ids', [])

                if not raw_work_id or not year or not authors:
                    continue

                work_id = extract_id(raw_work_id)
                citations = cit_dict.get(work_id, 0)

                # 清洗作者 ID
                clean_authors = [extract_id(a) for a in authors if a]

                paper_records.append({
                    'work_id': work_id,
                    'year': int(year),
                    'citations': citations,
                    'author_ids': clean_authors
                })

    df_papers = pd.DataFrame(paper_records)
    print(f"   ✅ 共提取 {len(df_papers):,} 篇有效 GenAI 论文。")

    print("\n3. 🧮 正在计算每年的 GenAI 全局基准引用量与论文 logRCR...")
    # 计算每年的平均被引（用来作为 RCR 的分母）
    yearly_avg_citations = df_papers.groupby('year')['citations'].mean().to_dict()

    def calculate_log_rcr(row):
        yr = row['year']
        cites = row['citations']
        expected_cites = yearly_avg_citations.get(yr, 0)

        # 防止除以 0 (某一年所有论文都是 0 引用时)
        if expected_cites == 0:
            return 0.0

        # RCR = 真实被引 / 同年平均被引
        rcr = cites / expected_cites
        # logRCR = ln(RCR + 1)
        return np.log1p(rcr)

    df_papers['logRCR'] = df_papers.apply(calculate_log_rcr, axis=1)

    print("\n4. 🧬 正在展开作者网络并与【DID 存活名单】合并...")
    # 把 ['A1', 'A2'] 炸开成多行，每行一个作者对应一篇文章
    df_exploded = df_papers.explode('author_ids')
    df_exploded.rename(columns={'author_ids': 'author_id'}, inplace=True)

    # 读入你千辛万苦洗出来的 DID 静态作者表
    df_static = pd.read_csv(FINAL_AUTHORS_CSV)
    # 统一清洗格式，确保精准匹配
    df_static['author_id'] = df_static['author_id'].apply(extract_id)

    # Inner Join 灵魂合并！被 PSM 淘汰的人的论文会在这里自动灰飞烟灭
    df_merged = pd.merge(df_exploded, df_static, on='author_id', how='inner')

    print("\n5. 📊 正在聚合【Author-Year】长面板数据并计算 DID 交互项...")
    # 按“作者”和“年份”分组，计算该作者这一年发的所有 GenAI 论文的平均 logRCR
    df_panel = df_merged.groupby(
        ['author_id', 'year', 'is_treated', 'event_year_aligned', 'is_industry_elite', 'is_academic_elite',
         'country_code']).agg(
        avg_logRCR=('logRCR', 'mean'),
        yearly_paper_count=('work_id', 'count')  # 顺手算一下他当年发了几篇
    ).reset_index()

    # ====== 核心：计算 DID 面板的关键变量 ======
    # 1. 相对年份：论文发表年份 - 他跨界的那一年
    df_panel['relative_year'] = df_panel['year'] - df_panel['event_year_aligned']

    # 2. Post 虚拟变量：是否处于干预后 (relative_year >= 0 则为 1，否则为 0)
    df_panel['Post'] = (df_panel['relative_year'] >= 0).astype(int)

    # 3. DID 交互项：是实验组 且 在干预后
    df_panel['DID_term'] = df_panel['is_treated'] * df_panel['Post']

    # 按作者和年份排个序，方便看
    df_panel = df_panel.sort_values(by=['author_id', 'year'])

    print("\n=========================================================")
    print("🎉 终极 DID 面板数据 (Ultimate Panel) 构建完成！")
    print(f"   最终面板包含 {df_panel['author_id'].nunique():,} 位精英作者的历年轨迹。")
    print(f"   总计生成 {len(df_panel):,} 条 年度观测值 (Observations)。")
    print("=========================================================")

    df_panel.to_csv(OUTPUT_PANEL, index=False, encoding='utf-8-sig')
    print(f"💾 终极成果已保存至: {OUTPUT_PANEL}")

    # 预览一下最核心的数据长什么样
    print("\n👇 面板数据前 5 行预览：")
    columns_to_show = ['author_id', 'year', 'relative_year', 'is_treated', 'Post', 'DID_term', 'avg_logRCR']
    print(df_panel[columns_to_show].head(5).to_string(index=False))


if __name__ == "__main__":
    build_did_panel()