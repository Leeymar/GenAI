import json
import pandas as pd
import os
from collections import defaultdict


def extract_phase1_t0_and_count():
    print("🧬 启动【第一阶段】创世老兵 T0 与本地火力提取器 (对齐 Phase 3 逻辑)...")

    filepath = 'phase1_hybrid_cleaned.jsonl'
    if not os.path.exists(filepath):
        print(f"❌ 找不到文件 {filepath}")
        return

    # 结构: {author_id: {'paper_count': 0, 'T0_year': 2016}}
    # 完美对齐你的 Phase 3 逻辑！
    p1_authors_stats = defaultdict(lambda: {'paper_count': 0, 'T0_year': 2016})
    paper_count_total = 0

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                work = json.loads(line)
                pub_year = work.get('publication_year')
                author_ids = work.get('author_ids', [])

                if not pub_year or not author_ids:
                    continue

                paper_count_total += 1

                for a_id in author_ids:
                    if not a_id: continue
                    a_id = str(a_id).strip()

                    # 1. 累加他在生成式 AI 领域的真实发文量
                    p1_authors_stats[a_id]['paper_count'] += 1

                    # 2. 锁定他的最小 T0 年份
                    if pub_year < p1_authors_stats[a_id]['T0_year']:
                        p1_authors_stats[a_id]['T0_year'] = pub_year

            except Exception:
                pass

    # 将字典转换成列表，方便导出
    records = []
    for a_id, stats in p1_authors_stats.items():
        records.append({
            'author_id': a_id,
            'T0_year': stats['T0_year'],
            'paper_count': stats['paper_count']  # 这里存的就是极其纯净的 GenAI 发文量
        })

    df_authors = pd.DataFrame(records)
    df_authors = df_authors.sort_values(by=['T0_year', 'author_id'])

    # 导出为 CSV，供下一步的 PWFC 测序仪使用
    output_csv = 'phase1_authors_T0.csv'
    df_authors.to_csv(output_csv, index=False)

    print("\n" + "=" * 50)
    print(" 🏆 第一阶段老兵名册已生成 (含纯净火力值)")
    print("=" * 50)
    print(f"📄 扫描文献: {paper_count_total:,} 篇")
    print(f"👥 独立老兵: {len(df_authors):,} 人")
    print(f"📂 名册保存: {output_csv}")
    print("=" * 50)


# 跑起来！
extract_phase1_t0_and_count()