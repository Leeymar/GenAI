import json
import os
import pandas as pd
from collections import defaultdict


def phase3_author_split_engine():
    print("🚀 启动【第三阶段 (2022-2026)】作者极速剥离与继承引擎...\n")

    phase3_data = 'phase3_hybrid_cleaned.jsonl'

    # 历史档案库 (Phase 1 和 Phase 2 的全部画像记录)
    historical_roster_files = [
        'E:\PythonProject\GenAI\dataset_scarp\stage_2\phase2_veterans_full_profiles.jsonl',
        'E:\PythonProject\GenAI\data_process\stage_2\phase2_newcomers_full_profiles.jsonl'
    ]

    out_veterans = 'phase3_veterans_full_profiles.jsonl'
    out_newcomers = 'phase3_newcomers_to_fetch.jsonl'

    # ---------------------------------------------------------
    # 1. 扫描 54 万篇 Phase 3 论文，提取全量作者战绩
    # ---------------------------------------------------------
    print(f"📥 正在解析 54 万篇大模型神作，提取作者火力分布...")
    # 结构: {author_id: {'paper_count': 10, 'T0_year': 2022}}
    p3_authors_stats = defaultdict(lambda: {'paper_count': 0, 'T0_year': 2026})

    paper_count = 0
    with open(phase3_data, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                work = json.loads(line)
                paper_count += 1
                year = work.get('publication_year')
                if not year: year = 2026

                for a_id in work.get('author_ids', []):
                    if not a_id: continue
                    a_id = str(a_id).strip()

                    p3_authors_stats[a_id]['paper_count'] += 1
                    # 记录该作者在第三阶段的最早发文年份
                    if year < p3_authors_stats[a_id]['T0_year']:
                        p3_authors_stats[a_id]['T0_year'] = year
            except Exception:
                pass

    total_p3_authors = len(p3_authors_stats)
    print(f"   ➤ 解析完成！共扫描 {paper_count:,} 篇论文。")
    print(f"   ➤ 捕获到第三阶段参战作者总计: {total_p3_authors:,} 人。\n")

    # ---------------------------------------------------------
    # 2. 匹配历史档案，进行老兵继承与新兵剥离
    # ---------------------------------------------------------
    print("🔍 正在与 19.3 万历史总名册进行血脉匹配...")
    veterans_count = 0

    # 构建历史作者的字典，方便 O(1) 极速查找
    historical_profiles = {}
    for hist_file in historical_roster_files:
        if os.path.exists(hist_file):
            with open(hist_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        profile = json.loads(line)
                        a_id = profile.get('author_ids')
                        if a_id:
                            historical_profiles[a_id] = profile
                    except:
                        pass

    # 开始分流
    with open(out_veterans, 'w', encoding='utf-8') as f_vet, \
            open(out_newcomers, 'w', encoding='utf-8') as f_new:

        for a_id, p3_stats in p3_authors_stats.items():
            if a_id in historical_profiles:
                # 🎯 老兵命中！继承并发扬光大
                veterans_count += 1
                vet_profile = historical_profiles[a_id].copy()

                # 总发文量 = 历史发文量 + Phase 3 发文量
                vet_profile['paper_count'] = vet_profile.get('paper_count', 0) + p3_stats['paper_count']
                # T0 年份绝对不能变！那是老兵最引以为傲的资历

                f_vet.write(json.dumps(vet_profile, ensure_ascii=False) + '\n')
            else:
                # 👶 纯新兵！打包成待查清单
                newcomer_record = {
                    "author_ids": a_id,
                    "T0_year": p3_stats['T0_year'],  # 新兵的 T0 就是在 Phase 3 的最早年份
                    "paper_count": p3_stats['paper_count']
                }
                f_new.write(json.dumps(newcomer_record, ensure_ascii=False) + '\n')

    newcomers_count = total_p3_authors - veterans_count

    print("================================================================")
    print(" 🎯 第三阶段 (2022-2026) 人员分流终极战报")
    print("================================================================")
    print(f"🎖️ 成功继承的跨周期老兵 (免查 API):  {veterans_count:>9,.0f} 人")
    print(f"👶 ChatGPT 时代涌入的新兵 (需测序):  {newcomers_count:>9,.0f} 人")
    print("================================================================\n")
    print(f"💾 老兵超级档案已保存至: {out_veterans}")
    print(f"💾 新兵待查清单已保存至: {out_newcomers}")
    print("💡 下一步，我们就要对这批庞大的新兵大军动用高并发测序仪了！")


# 跑起来！
phase3_author_split_engine()