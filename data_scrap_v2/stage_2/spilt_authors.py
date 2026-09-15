import json
import os
import pandas as pd
from collections import defaultdict


def phase2_author_split_engine():
    print("🚀 启动【第二阶段】作者极速剥离与老兵继承引擎...")

    phase2_data = 'phase2_hybrid_cleaned.jsonl'
    phase1_profile = 'E:\PythonProject\GenAI\data_scrap_v2\stage_1\phase1_newcomers_scopus_pwfc.jsonl'  # 第一阶段跑出来的终极 DNA 库

    out_veterans = 'phase2_veterans_full_profiles.jsonl'
    out_newcomers_csv = 'phase2_newcomers_to_fetch.csv'

    # ---------------------------------------------------------
    # 1. 扫描 16.2 万篇 Phase 2 论文，提取这 5 年的火力分布
    # ---------------------------------------------------------
    print(f"📥 正在解析 16.2 万篇前夜神作，提取作者火力...")
    p2_authors_stats = defaultdict(lambda: {'paper_count': 0, 'T0_year': 2021})

    paper_count = 0
    with open(phase2_data, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                work = json.loads(line)
                paper_count += 1
                year = work.get('publication_year')
                if not year: year = 2021

                for a_id in work.get('author_ids', []):
                    if not a_id: continue
                    a_id = str(a_id).strip()

                    p2_authors_stats[a_id]['paper_count'] += 1
                    if year < p2_authors_stats[a_id]['T0_year']:
                        p2_authors_stats[a_id]['T0_year'] = year
            except Exception:
                pass

    print(f"   ➤ 捕获到第二阶段参战作者总计: {len(p2_authors_stats):,} 人。\n")

    # ---------------------------------------------------------
    # 2. 读取第一阶段历史档案
    # ---------------------------------------------------------
    print("🔍 正在加载 Phase 1 创世老兵档案进行血脉匹配...")
    historical_profiles = {}
    if os.path.exists(phase1_profile):
        with open(phase1_profile, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    profile = json.loads(line)
                    a_id = profile.get('author_ids')
                    if a_id:
                        historical_profiles[a_id] = profile
                except:
                    pass

    # ---------------------------------------------------------
    # 3. 核心分流逻辑 (极其致命：防丢失机制)
    # ---------------------------------------------------------
    veterans_updated_count = 0

    with open(out_veterans, 'w', encoding='utf-8') as f_vet:
        # 遍历所有的老兵（不管他在第二阶段发没发文章，都要继承到未来的历史库里！）
        for a_id, vet_profile in historical_profiles.items():
            profile_to_save = vet_profile.copy()

            # 如果老兵在第二阶段参战了，火力值累加
            if a_id in p2_authors_stats:
                profile_to_save['paper_count'] += p2_authors_stats[a_id]['paper_count']
                veterans_updated_count += 1
                # 匹配完后，从 p2 字典里剔除他
                del p2_authors_stats[a_id]

            f_vet.write(json.dumps(profile_to_save, ensure_ascii=False) + '\n')

    # 此时 p2_authors_stats 里剩下的，全都是纯纯的第二阶段新兵！
    newcomers_count = len(p2_authors_stats)

    # 将新兵导出为 CSV，准备送进测序仪
    records = []
    for a_id, stats in p2_authors_stats.items():
        records.append({
            'author_id': a_id,
            'T0_year': stats['T0_year'],
            'paper_count': stats['paper_count']
        })

    df_newcomers = pd.DataFrame(records).sort_values(by=['T0_year', 'author_id'])
    df_newcomers.to_csv(out_newcomers_csv, index=False)

    print("\n" + "=" * 60)
    print(" 🎯 第二阶段 (2017-2021) 人员分流终极战报")
    print("=" * 60)
    print(f"🎖️ Phase 1 老兵总数: {len(historical_profiles):,} 人")
    print(f"⚔️ 其中在 Phase 2 继续跨界作战的老兵: {veterans_updated_count:,} 人")
    print(f"👶 Transformer 时代涌入的纯新兵 (需测序): {newcomers_count:,} 人")
    print("=" * 60)
    print(f"💾 【累计老兵超级档案】已保存至: {out_veterans}")
    print(f"💾 【纯新兵待查 CSV 清单】已保存至: {out_newcomers_csv}")


# 执行分流！
phase2_author_split_engine()