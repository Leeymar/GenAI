import json
import os
import pandas as pd
from collections import defaultdict


def build_ultimate_master_matrix():
    print("🌌 启动【12年全战区】终极真理矩阵融合引擎...\n")

    # 1. 12 年的绝对真理底库 (文献全集)
    paper_files = [
        'E:\PythonProject\GenAI\data_scrap_v2\stage_1\phase1_hybrid_cleaned.jsonl',
        'E:\PythonProject\GenAI\data_scrap_v2\stage_2\phase2_hybrid_cleaned.jsonl',
        'E:\PythonProject\GenAI\data_scrap_v2\stage_3\phase3_hybrid_cleaned.jsonl'
    ]

    # 2. 所有的 DNA 画像碎片库 (把洗出来的漏网之鱼也放进来，榨干剩余价值)
    dna_files = [
        'E:\PythonProject\GenAI\data_scrap_v2\stage_1\phase1_newcomers_scopus_pwfc.jsonl',
        'E:\PythonProject\GenAI\data_scrap_v2\stage_2\phase2_newcomers_scopus_pwfc.jsonl',
        'E:\PythonProject\GenAI\data_scrap_v2\stage_3\phase3_newcomers_scopus_pwfc_FIXED.jsonl',
    ]

    master_output_csv = 'MASTER_AUTHOR_PROFILES_12YEARS.csv'

    # ---------------------------------------------------------
    # 步骤一：从 70 多万篇文献中，提取绝对的 T0 和 终极火力值
    # ---------------------------------------------------------
    print("📥 正在全盘扫描 12 年所有文献，重构绝对 T0 与总火力...")
    # author_stats 结构: {author_id: {'T0_year': 2026, 'total_papers': 0}}
    author_stats = defaultdict(lambda: {'T0_year': 2026, 'total_papers': 0})
    total_papers_scanned = 0

    for p_file in paper_files:
        if not os.path.exists(p_file):
            print(f"⚠️ 警告: 找不到文献库 {p_file}")
            continue

        with open(p_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    work = json.loads(line)
                    total_papers_scanned += 1
                    year = work.get('publication_year', 2026)

                    for a_id in work.get('author_ids', []):
                        if not a_id: continue
                        a_id = str(a_id).strip()

                        author_stats[a_id]['total_papers'] += 1
                        if year < author_stats[a_id]['T0_year']:
                            author_stats[a_id]['T0_year'] = year
                except Exception:
                    pass

    print(f"   ➤ 扫描完毕！共读取文献 {total_papers_scanned:,} 篇。")
    print(f"   ➤ 12 年总参战人数 (绝对去重): {len(author_stats):,} 人。\n")

    # ---------------------------------------------------------
    # 步骤二：加载所有提取过的 PWFC 学科基因
    # ---------------------------------------------------------
    print("🧬 正在拼装全量学者的 PWFC 基因图谱...")
    dna_vault = {}

    for d_file in dna_files:
        if not os.path.exists(d_file):
            print(f"⚠️ 警告: 找不到基因库 {d_file}")
            continue

        with open(d_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    profile = json.loads(line)
                    a_id = str(profile.get('author_ids') or profile.get('author_id')).strip()

                    # 只要提取学科标签和权重即可
                    dna_vault[a_id] = {
                        'primary_origin_field': profile.get('primary_origin_field', 'Unknown'),
                        'scopus_dna_weights': json.dumps(profile.get('scopus_dna_weights', {}))  # 转成字符串方便存 CSV
                    }
                except Exception:
                    pass

    print(f"   ➤ 基因图谱拼装完毕！共掌握 {len(dna_vault):,} 人的学科序列。\n")

    # ---------------------------------------------------------
    # 步骤三：终极大融合
    # ---------------------------------------------------------
    print("🪐 正在执行大一统融合并生成最终 Master 表...")
    master_records = []

    for a_id, stats in author_stats.items():
        # 获取他的学科基因，如果没有查过 API，则标记为 Unknown
        dna = dna_vault.get(a_id, {
            'primary_origin_field': 'Unknown',
            'scopus_dna_weights': '{}'
        })

        # 组装 12 年绝对正确的完美档案
        master_records.append({
            'author_id': a_id,
            'T0_year': stats['T0_year'],
            'total_papers': stats['total_papers'],
            'primary_origin_field': dna['primary_origin_field'],
            'scopus_dna_weights': dna['scopus_dna_weights']
        })

    # 转换为 Pandas DataFrame 并按 T0 年份排序
    df_master = pd.DataFrame(master_records)
    df_master = df_master.sort_values(by=['T0_year', 'total_papers'], ascending=[True, False])

    df_master.to_csv(master_output_csv, index=False)

    print("================================================================")
    print(" 🏆 12 年生成式 AI 人口大普查终极战报")
    print("================================================================")
    print(f"👥 总入局学者: {len(df_master):,} 人")

    # 打印各阶段的绝对真实人数（彻底消除了任何穿越 Bug）
    print("\n📊 各阶段入局人数 (按 T0 严格切分):")
    phase1_count = len(df_master[df_master['T0_year'] <= 2016])
    phase2_count = len(df_master[(df_master['T0_year'] >= 2017) & (df_master['T0_year'] <= 2021)])
    phase3_count = len(df_master[df_master['T0_year'] >= 2022])
    print(f"   ➤ 创世期 (2014-2016): {phase1_count:,} 人")
    print(f"   ➤ 前夜期 (2017-2021): {phase2_count:,} 人")
    print(f"   ➤ 爆发期 (2022-2026): {phase3_count:,} 人")
    print("================================================================\n")
    print(f"💾 终极真理库已安全保存为: {master_output_csv}")


# 启动大一统！
build_ultimate_master_matrix()