import json
import numpy as np

RAW_DATA_PATH = r'E:\PythonProject\GenAI\data_process\DID_authors\Author_Global_Metadata_90k.jsonl'
BASELINE_PATH = r'E:\PythonProject\GenAI\data_process\DID_authors\Author_Baseline_for_PSM.jsonl'
OUTPUT_PATH = r'E:\PythonProject\GenAI\data_process\DID_authors\Author_Baseline_Rebuilt_v2.jsonl'
REFERENCE_YEAR = 2023


# ----------------------------------------------------------
# 辅助函数
# ----------------------------------------------------------
def compute_baseline_vars(counts_by_year, first_pub_year, T):
    """计算 T 之前的累计发文量、累计引用量"""
    pre_works = 0
    pre_cites = 0
    for entry in (counts_by_year or []):
        y = entry.get('year', 0)
        if y < T:
            pre_works += entry.get('works_count', 0)
            pre_cites += entry.get('cited_by_count', 0)
    return {
        'first_pub_year': first_pub_year,
        'pre_total_works': pre_works,
        'pre_total_citations': pre_cites,
    }


def compute_pre_growth_rate(counts_by_year, T):
    """
    计算跨界前 3 年（T-3 到 T-1）的发文年均复合增长率（CAGR）。

    例如 T=2023，则取 2020、2021、2022 年的发文量，
    计算 (w_2022 / w_2020)^(1/2) - 1

    返回值：
        float，范围大致在 [-1, +∞)
        如果数据不足或起点为0，返回 0.0
    """
    works = {}
    for entry in (counts_by_year or []):
        y = entry.get('year', 0)
        # 取 T-3, T-2, T-1 三年
        if T - 3 <= y <= T - 1:
            works[y] = entry.get('works_count', 0)

    years = sorted(works.keys())

    # 至少要有两个年份才能算增长率
    if len(years) < 2:
        return 0.0

    w_start = works[years[0]]
    w_end = works[years[-1]]
    n_years = years[-1] - years[0]  # 时间跨度

    if w_start == 0 or n_years == 0:
        return 0.0

    cagr = (w_end / w_start) ** (1.0 / n_years) - 1
    return round(float(cagr), 6)


# ----------------------------------------------------------
# 加载原始数据（含 counts_by_year）
# ----------------------------------------------------------
print("加载原始数据...")
raw_data = {}
with open(RAW_DATA_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        obj = json.loads(line)
        aid = str(obj.get('id', obj.get('author_id', ''))).split('/')[-1]
        raw_data[aid] = obj
print(f"  原始数据：{len(raw_data)} 条")

# ----------------------------------------------------------
# 加载基准元数据（is_treated / h_index / event_year 等）
# ----------------------------------------------------------
print("加载基准元数据...")
baseline_data = []
with open(BASELINE_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        obj = json.loads(line)
        obj['author_id'] = str(obj.get('author_id', '')).split('/')[-1]
        baseline_data.append(obj)

import pandas as pd

df_meta = pd.DataFrame(baseline_data)
print(f"  基准元数据：{len(df_meta)} 条 | "
      f"实验组：{(df_meta['is_treated'] == 1).sum()} | "
      f"对照组：{(df_meta['is_treated'] == 0).sum()}")

# ----------------------------------------------------------
# 逐条计算，生成新的基线文件
# ----------------------------------------------------------
print("计算基线变量（含 pre_growth_rate）...")
results = []
missing = 0
zero_start = 0  # 记录因起点为0无法算增长率的数量

for _, row in df_meta.iterrows():
    aid = row['author_id']
    T = (int(row['event_year'])
         if row['is_treated'] == 1 and pd.notna(row.get('event_year'))
         else REFERENCE_YEAR)

    raw = raw_data.get(aid)
    if raw is None:
        missing += 1
        counts_by_year = []
        first_pub_year = row.get('first_pub_year')
    else:
        counts_by_year = raw.get('counts_by_year', [])
        first_pub_year = raw.get('first_pub_year', row.get('first_pub_year'))

    base_vars = compute_baseline_vars(counts_by_year, first_pub_year, T)
    growth_rate = compute_pre_growth_rate(counts_by_year, T)

    if growth_rate == 0.0:
        zero_start += 1

    results.append({
        'author_id': aid,
        'is_treated': int(row['is_treated']),
        'event_year': row.get('event_year'),
        'T': T,
        'country_code': row.get('country_code'),
        'h_index': row.get('h_index', 0),
        'pre_growth_rate': growth_rate,  # ← 新增
        **base_vars
    })

# ----------------------------------------------------------
# 写出
# ----------------------------------------------------------
with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

print(f"\n✅ 生成完毕：{len(results)} 条 → {OUTPUT_PATH}")
print(f"   原始数据中找不到的作者：{missing} 个")
print(f"   pre_growth_rate 为 0（数据不足）的作者：{zero_start} 个")
