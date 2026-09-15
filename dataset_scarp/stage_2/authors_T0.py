import json
import pandas as pd
import os
import csv


def extract_phase2_active_authors_with_history(phase1_csv, phase2_jsonl, output_csv):
    print("🧬 启动【第二阶段活跃学者】T0 历史追溯引擎...\n")

    # ---------------------------------------------------------
    # 1. 提取第一阶段 (Phase 1) 老兵底档 (仅作字典备查)
    # ---------------------------------------------------------
    phase1_history = {}
    if os.path.exists(phase1_csv):
        print(f"📥 正在读取第一阶段历史档案: {phase1_csv}")
        with open(phase1_csv, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header_skipped = False
            for row in reader:
                if not row: continue
                # 跳过表头
                if not header_skipped and any(c.isalpha() for c in row[1]):
                    header_skipped = True
                    continue

                short_id = row[0].split('/')[-1].strip()
                t0_year = int(row[1].strip())
                phase1_count = int(row[2].strip())

                # 记录老兵的历史 T0 和 第一阶段的发文量
                phase1_history[short_id] = {
                    'T0': t0_year,
                    'Phase1_Count': phase1_count
                }
        print(f"   ➤ 成功提取 {len(phase1_history):,} 位老兵的历史档案。\n")
    else:
        print(f"⚠️ 未找到 {phase1_csv}，将无法追溯老兵历史。")

    # ---------------------------------------------------------
    # 2. 扫描第二阶段 (Phase 2) 确定活跃名单
    # ---------------------------------------------------------
    print(f"📥 正在扫描第二阶段活跃名单: {phase2_jsonl}")
    phase2_authors = {}

    with open(phase2_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                year = data.get('publication_year')
                author_ids = data.get('author_ids', [])

                if not year or not author_ids: continue

                for raw_a_id in author_ids:
                    short_id = str(raw_a_id).split('/')[-1]

                    if short_id not in phase2_authors:
                        # 第一次在第二阶段遇到这个人
                        phase2_authors[short_id] = {
                            'Phase2_T0': int(year),
                            'Phase2_Count': 1
                        }
                    else:
                        # 再次遇到，发文量+1，并确保 T0 是第二阶段最早的年份
                        phase2_authors[short_id]['Phase2_Count'] += 1
                        if int(year) < phase2_authors[short_id]['Phase2_T0']:
                            phase2_authors[short_id]['Phase2_T0'] = int(year)
            except Exception as e:
                continue

    print(f"   ➤ 成功锁定 {len(phase2_authors):,} 位第二阶段活跃作者！\n")

    # ---------------------------------------------------------
    # 3. 历史合并与溯源 (核心逻辑)
    # ---------------------------------------------------------
    print("🧮 正在对这批活跃学者进行 T0 历史追溯与战绩合并...")
    final_records = []

    veteran_count = 0  # 统计有多少老兵留存到了第二阶段

    for short_id, p2_data in phase2_authors.items():
        if short_id in phase1_history:
            # 🎯 关键逻辑：他是老兵！
            # 1. T0 强制改写为第一阶段的入行年份
            final_t0 = phase1_history[short_id]['T0']
            # 2. 总战绩 = 第一阶段战绩 + 第二阶段战绩
            final_count = phase1_history[short_id]['Phase1_Count'] + p2_data['Phase2_Count']
            veteran_count += 1
        else:
            # 👶 关键逻辑：他是第二阶段才入局的新兵
            final_t0 = p2_data['Phase2_T0']
            final_count = p2_data['Phase2_Count']

        final_records.append({
            'authors_id': f"https://openalex.org/{short_id}",
            'T0_year': final_t0,
            'paper_count': final_count
        })

    # ---------------------------------------------------------
    # 4. 终极排序与导出
    # ---------------------------------------------------------
    df = pd.DataFrame(final_records)

    # 排序：按入局年份(T0_year)升序排，同一年入局的按发文量(paper_count)降序排
    df = df.sort_values(by=['T0_year', 'paper_count'], ascending=[True, False])
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')

    print("================================================================")
    print(" 🏆 第二阶段活跃学者全景战力榜 (前 10 名最资深大佬)")
    print("================================================================")
    for index, row in df.head(10).iterrows():
        print(
            f"   ➤ 真实入行年份: {row['T0_year']} | 累计总发文: {row['paper_count']:>4} 篇 | ID: {row['authors_id'].split('/')[-1]}")
    print("================================================================\n")
    print(f"🔥 数据洞察：在这 19 万大军中，共有 {veteran_count:,} 位是从第一阶段（2014-2016）杀过来的史前巨佬！")
    print(f"💾 第二阶段专属追溯名单已保存至: {output_csv}")


# 填入你的文件路径，直接开跑！
extract_phase2_active_authors_with_history(
    phase1_csv='E:\PythonProject\GenAI\dataset_scarp\stage_1\phase1_authors_T0_final.csv',
    phase2_jsonl='phase2_hybrid_cleaned.jsonl',
    output_csv='phase2_active_authors_with_historical_T0.csv'
)