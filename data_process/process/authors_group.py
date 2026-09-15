import json
import os
import pandas as pd
from collections import defaultdict


def extract_cross_era_team_evolution():
    print("🕸️ 启动【跨时代枢纽学者】团队演化追踪引擎...\n")

    # 1. 12年文献全集路径
    phase_files = {
        'Phase1': 'E:\PythonProject\GenAI\data_scrap_v2\stage_1\phase1_hybrid_cleaned.jsonl',
        'Phase2': 'E:\PythonProject\GenAI\data_scrap_v2\stage_2\phase2_hybrid_cleaned.jsonl',
        'Phase3': 'E:\PythonProject\GenAI\data_scrap_v2\stage_3\phase3_hybrid_cleaned.jsonl'
    }

    # ---------------------------------------------------------
    # 步骤一：全盘扫描，记录每位学者的“活跃期”与“团队规模”
    # ---------------------------------------------------------
    print("📥 正在扫描全量文献，构建学者时空轨迹...")

    # 记录作者在哪些阶段发过文: {author_id: set('Phase1', 'Phase3')}
    author_active_phases = defaultdict(set)

    # 记录作者在每个阶段发表的每篇论文的团队人数: {author_id: {'Phase1': [3, 4], 'Phase2': [5]}}
    author_team_sizes = defaultdict(lambda: {'Phase1': [], 'Phase2': [], 'Phase3': []})

    total_papers = 0

    for phase_name, file_path in phase_files.items():
        if not os.path.exists(file_path):
            print(f"   ⚠️ 警告: 找不到 {file_path}")
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    work = json.loads(line)
                    total_papers += 1

                    # 提取该篇论文的完整作者列表
                    author_ids = [str(a).strip() for a in work.get('author_ids', []) if a]
                    team_size = len(author_ids)

                    if team_size == 0: continue

                    for a_id in author_ids:
                        author_active_phases[a_id].add(phase_name)
                        author_team_sizes[a_id][phase_name].append(team_size)
                except Exception:
                    pass

    print(f"   ➤ 扫描完毕！共处理 {total_papers:,} 篇文献，捕获 {len(author_active_phases):,} 名独立学者。\n")

    # ---------------------------------------------------------
    # 步骤二：筛选“跨时代枢纽” (活跃阶段 >= 2 的人)
    # ---------------------------------------------------------
    print("🔍 正在大浪淘沙，锁定跨时代枢纽大牛...")
    cross_era_authors = []

    for a_id, phases in author_active_phases.items():
        if len(phases) >= 2:  # 🎯 核心逻辑：横跨至少两个阶段

            # 计算该学者在各个阶段的平均团队规模 (如果有发文的话)
            p1_sizes = author_team_sizes[a_id]['Phase1']
            p2_sizes = author_team_sizes[a_id]['Phase2']
            p3_sizes = author_team_sizes[a_id]['Phase3']

            avg_p1 = sum(p1_sizes) / len(p1_sizes) if p1_sizes else None
            avg_p2 = sum(p2_sizes) / len(p2_sizes) if p2_sizes else None
            avg_p3 = sum(p3_sizes) / len(p3_sizes) if p3_sizes else None

            cross_era_authors.append({
                'author_id': a_id,
                'active_phases': " | ".join(sorted(list(phases))),
                'phases_count': len(phases),
                'total_papers_12y': len(p1_sizes) + len(p2_sizes) + len(p3_sizes),
                'avg_team_size_P1': avg_p1,
                'avg_team_size_P2': avg_p2,
                'avg_team_size_P3': avg_p3
            })

    df_cross = pd.DataFrame(cross_era_authors)
    df_cross = df_cross.sort_values(by=['phases_count', 'total_papers_12y'], ascending=[False, False])

    output_csv = 'Cross_Era_Hub_Scholars_Team_Evolution.csv'
    df_cross.to_csv(output_csv, index=False)

    print("================================================================")
    print(" 🏆 跨时代枢纽学者（至少跨2个阶段）提取战报")
    print("================================================================")
    print(f"   ➤ 提取出跨时代大牛总计: {len(df_cross):,} 人")

    # 打印一些神仙级别的数据
    p1_p2_p3 = len(df_cross[df_cross['phases_count'] == 3])
    print(f"   ➤ 贯穿 12 年全部 3 个阶段的骨灰级教父: {p1_p2_p3:,} 人")

    print("================================================================\n")
    print(f"💾 团队演化详细矩阵已保存至: {output_csv}")


# 启动引擎
extract_cross_era_team_evolution()