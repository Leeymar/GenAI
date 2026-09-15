import pandas as pd
import json
import os
from tqdm import tqdm

# ==========================================
# ⚙️ 核心映射与判定规则配置
# ==========================================
# 1. arXiv 跨界野心降维映射字典
BROAD_DOMAINS = {
    'Type 3 (HSS Cross-over)': ['econ', 'q-fin'],
    'Type 2 (Hard Science)': ['q-bio', 'physics', 'astro-ph', 'cond-mat', 'quant-ph', 'hep-ex', 'hep-ph', 'nlin',
                              'math-ph', 'hep-th', 'gr-qc', 'hep-lat', 'nucl-th', 'nucl-ex'],
    'Type 1 (Math & Engineering)': ['stat', 'math', 'eess'],
    'Type 0 (Pure CS)': ['cs']
}

# 2. 你的神级 Scopus 文商社科定义
HSS_FIELDS = {
    'Social Sciences', 'Arts and Humanities', 'Psychology',
    'Business, Management and Accounting', 'Economics, Econometrics and Finance', 'Decision Sciences'
}


def get_arxiv_type(all_categories_str):
    if pd.isna(all_categories_str) or all_categories_str == "Unknown":
        return "Type 0 (Pure CS)"  # 默认降级为纯计算机

    tags = str(all_categories_str).split('|')
    domains = [tag.split('.')[0] if '.' in tag else tag for tag in tags]

    for domain in domains:
        if domain in BROAD_DOMAINS['Type 3 (HSS Cross-over)']: return 'Type 3 (HSS)'
    for domain in domains:
        if domain in BROAD_DOMAINS['Type 2 (Hard Science)']: return 'Type 2 (Science)'
    for domain in domains:
        if domain in BROAD_DOMAINS['Type 1 (Math & Engineering)']: return 'Type 1 (Engineering)'
    return 'Type 0 (Pure CS)'


def build_ultimate_master_table():
    print("🚀 启动【终极缝合引擎】：历史 DNA 浓度 ✖️ arXiv 跨界野心...\n")

    # ==========================================
    # 1. 加载字典：作者 DNA 底库
    # ==========================================
    print("🧬 1. 正在加载 12 年全量作者历史 DNA...")
    try:
        master_author_df = pd.read_csv('E:\PythonProject\GenAI\data_process\process\MASTER_AUTHOR_PROFILES_12YEARS.csv')
        id_to_dna = {}
        for _, row in master_author_df.iterrows():
            a_id = str(row['author_id']).strip()
            try:
                id_to_dna[a_id] = json.loads(row['scopus_dna_weights'])
            except:
                id_to_dna[a_id] = {}
        print(f"✅ 成功加载 {len(id_to_dna)} 位作者的历史 DNA 档案。")
    except Exception as e:
        print(f"❌ 无法读取 MASTER_AUTHOR_PROFILES_12YEARS.csv: {e}")
        return

    # ==========================================
    # 2. 循环处理三大阶段
    # ==========================================
    PHASES = {
        'Phase1': {'arxiv': 'phase1_arxiv_categories.csv',
                   'jsonl': r'E:\PythonProject\GenAI\data_scrap_v2\stage_1\phase1_hybrid_cleaned.jsonl'},
        'Phase2': {'arxiv': 'phase2_arxiv_categories.csv',
                   'jsonl': r'E:\PythonProject\GenAI\data_scrap_v2\stage_2\phase2_hybrid_cleaned.jsonl'},
        'Phase3': {'arxiv': 'phase3_arxiv_categories.csv',
                   'jsonl': r'E:\PythonProject\GenAI\data_scrap_v2\stage_3\phase3_hybrid_cleaned.jsonl'}
    }

    all_records = []

    for phase_name, paths in PHASES.items():
        arxiv_file = paths['arxiv']
        jsonl_file = paths['jsonl']

        if not os.path.exists(arxiv_file) or not os.path.exists(jsonl_file): continue
        print(f"\n🎯 正在处理 {phase_name} ...")

        # 读取该阶段的 arXiv 标签并分类
        df_arxiv = pd.read_csv(arxiv_file, dtype=str)
        df_arxiv['ArXiv_Type'] = df_arxiv['all_categories'].apply(get_arxiv_type)
        arxiv_dict = df_arxiv.set_index('work_id').to_dict('index')
        target_w_ids = set(arxiv_dict.keys())

        phase_records = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    work = json.loads(line)
                    w_id = str(work.get('work_id') or work.get('id')).split('/')[-1]

                    # 只有当我们成功抓取到了它的 arXiv 标签时，才计算它的团队 DNA
                    if w_id in target_w_ids:
                        a_ids = [str(aid).strip() for aid in work.get('author_ids', []) if aid]
                        if not a_ids: continue

                        # 你的核心 DNA 算法
                        team_dna_sum = {}
                        valid_authors = 0
                        for aid in a_ids:
                            author_dna = id_to_dna.get(aid, {})
                            if author_dna:
                                valid_authors += 1
                                for field, weight in author_dna.items():
                                    team_dna_sum[field] = team_dna_sum.get(field, 0.0) + weight

                        if valid_authors == 0: continue

                        hss_concentration = sum(team_dna_sum.get(field, 0.0) for field in HSS_FIELDS)
                        hss_percentage = (hss_concentration / valid_authors) * 100

                        arx_info = arxiv_dict[w_id]

                        phase_records.append({
                            'work_id': w_id,
                            'stage': phase_name,
                            'arxiv_id': arx_info['arxiv_id'],
                            'arxiv_primary_category': arx_info['primary_category'],
                            'arxiv_type': arx_info['ArXiv_Type'],
                            'total_authors': valid_authors,
                            'team_hss_dna_pct': round(hss_percentage, 2)
                        })
                except Exception:
                    pass

        all_records.extend(phase_records)
        print(f"✅ {phase_name} 缝合完毕，得到 {len(phase_records)} 篇有效交叉文献。")

    # ==========================================
    # 3. 输出终极分析表
    # ==========================================
    if not all_records: return

    master_df = pd.DataFrame(all_records)
    output_file = 'Ultimate_ArXiv_DNA_Master.csv'
    master_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print("\n" + "=" * 60)
    print("🏆 终极版大表生成成功！")
    print(f"💾 已保存至: {output_file}")
    print("\n你可以用这段极简代码把它拼进你的画图脚本里：")
    print("df = pd.merge(df, pd.read_csv('Ultimate_ArXiv_DNA_Master.csv'), on='work_id', how='inner')")
    print("=" * 60)


if __name__ == "__main__":
    build_ultimate_master_table()