import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────
# 第一步：加载数据
# ─────────────────────────────────────────────────────────────────
panel = pd.read_csv('Ultimate_DID_Panel.csv')
panel['year'] = panel['year'].astype(int)
print(f"原始行数：{len(panel):,} | 作者数：{panel['author_id'].nunique():,}")

# ─────────────────────────────────────────────────────────────────
# 第二步：字段清洗
# ─────────────────────────────────────────────────────────────────
panel['is_academic_elite'] = panel['is_academic_elite'].fillna(0).astype(int)
panel['is_industry_elite'] = panel['is_industry_elite'].fillna(0).astype(int)
panel['is_elite_overall']  = panel['is_elite_overall'].fillna(0).astype(int)

# 互斥性：同时命中归为精英高校
panel.loc[panel['is_academic_elite'] == 1, 'is_industry_elite'] = 0

# Unknown/NaN 统一归为 Other
panel['country_code'] = panel['country_code'].replace('Unknown', np.nan).fillna('Other')

# ─────────────────────────────────────────────────────────────────
# 第三步：构造实验二核心变量
# ─────────────────────────────────────────────────────────────────
panel['Post_chatgpt']     = (panel['year'] >= 2022).astype(int)
panel['Post_transformer'] = (panel['year'] >= 2017).astype(int)
panel['DID_chatgpt']      = panel['is_treated'] * panel['Post_chatgpt']
panel['DID_transformer']  = panel['is_treated'] * panel['Post_transformer']

# ─────────────────────────────────────────────────────────────────
# 第四步：地域分组（中美 vs 其他）
# ─────────────────────────────────────────────────────────────────
panel['region'] = panel['country_code'].apply(
    lambda x: 'US' if x == 'US' else ('CN' if x == 'CN' else 'Other')
)

# ─────────────────────────────────────────────────────────────────
# 第五步：构造所有交互项
# ─────────────────────────────────────────────────────────────────
panel['elite_univ_post']   = panel['is_academic_elite'] * panel['Post_chatgpt']
panel['industry_top_post'] = panel['is_industry_elite'] * panel['Post_chatgpt']
panel['region_US_post']    = (panel['region'] == 'US').astype(int) * panel['Post_chatgpt']
panel['region_CN_post']    = (panel['region'] == 'CN').astype(int) * panel['Post_chatgpt']

# ─────────────────────────────────────────────────────────────────
# 第六步：清洗并构建完整面板（全量，用于 B2/E1/E2/E3）
# ─────────────────────────────────────────────────────────────────
panel_clean = panel.dropna(subset=['avg_logRCR', 'is_treated'])
panel_clean = panel_clean[~np.isinf(panel_clean['avg_logRCR'])]

panel_idx = (
    panel_clean.set_index(['author_id', 'year'])
    .groupby(level=['author_id', 'year'])
    .mean(numeric_only=True)
)
print(f"✅ 全量面板：{len(panel_idx):,} 行 | 作者：{panel_idx.index.get_level_values(0).nunique():,} 人")

# ─────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────
def stars(p):
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    else: return 'n.s.'

def extract_did(model):
    coef = model.params['DID_chatgpt']
    pval = model.pvalues['DID_chatgpt']
    ci   = model.conf_int().loc['DID_chatgpt']
    return coef, pval, ci

def print_did(label, coef, pval, ci):
    print(f"  DID_chatgpt：{coef:+.4f}  {stars(pval):<5}  "
          f"p={pval:.4f}  CI=[{ci['lower']:+.4f}, {ci['upper']:+.4f}]")

def print_controls(model, col_label_pairs):
    for col, label in col_label_pairs:
        c  = model.params[col]
        p  = model.pvalues[col]
        ci = model.conf_int().loc[col]
        print(f"    {label:<28}：{c:+.4f}  {stars(p):<5}  "
              f"p={p:.4f}  CI=[{ci['lower']:+.4f}, {ci['upper']:+.4f}]")

# ─────────────────────────────────────────────────────────────────
# 模型 B2：基准（TWFE + Transformer，无机构/地域控制）
# ─────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("模型 B2：基准（无机构/地域控制）")
print('='*65)

mB2 = PanelOLS.from_formula(
    'avg_logRCR ~ DID_chatgpt + DID_transformer + EntityEffects + TimeEffects',
    data=panel_idx
).fit(cov_type='clustered', cluster_entity=True)

coef_B2, pval_B2, ci_B2 = extract_did(mB2)
print_did('B2', coef_B2, pval_B2, ci_B2)

# ─────────────────────────────────────────────────────────────────
# 模型 E1：TWFE + Transformer + 机构控制（不含地域）
# ─────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("模型 E1：+机构层级控制（精英高校 + 产业精英）")
print('='*65)

mE1 = PanelOLS.from_formula(
    '''avg_logRCR ~ DID_chatgpt + DID_transformer
       + elite_univ_post + industry_top_post
       + EntityEffects + TimeEffects''',
    data=panel_idx
).fit(cov_type='clustered', cluster_entity=True)

coef_E1, pval_E1, ci_E1 = extract_did(mE1)
print_did('E1', coef_E1, pval_E1, ci_E1)
print("  机构控制项：")
print_controls(mE1, [
    ('elite_univ_post',   'QS精英高校 × Post'),
    ('industry_top_post', '产业精英机构 × Post'),
])

pct_E1 = abs(coef_E1 - coef_B2) / abs(coef_B2) * 100
print(f"  系数变化（B2→E1）：{pct_E1:.1f}%  → {'✅ 稳健' if pct_E1 < 10 else '⚠️ 存在混淆'}")

# ─────────────────────────────────────────────────────────────────
# 模型 E2：TWFE + Transformer + 地域控制（不含机构）
# ─────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("模型 E2：+地域控制（美国 + 中国）")
print('='*65)

mE2 = PanelOLS.from_formula(
    '''avg_logRCR ~ DID_chatgpt + DID_transformer
       + region_US_post + region_CN_post
       + EntityEffects + TimeEffects''',
    data=panel_idx
).fit(cov_type='clustered', cluster_entity=True)

coef_E2, pval_E2, ci_E2 = extract_did(mE2)
print_did('E2', coef_E2, pval_E2, ci_E2)
print("  地域控制项：")
print_controls(mE2, [
    ('region_US_post', '美国 × Post'),
    ('region_CN_post', '中国 × Post'),
])

pct_E2 = abs(coef_E2 - coef_B2) / abs(coef_B2) * 100
print(f"  系数变化（B2→E2）：{pct_E2:.1f}%  → {'✅ 稳健' if pct_E2 < 10 else '⚠️ 存在混淆'}")

# ─────────────────────────────────────────────────────────────────
# 模型 E3：完整控制（机构 + 地域）★
# ─────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("模型 E3：完整控制（机构 + 地域）★")
print('='*65)

mE3 = PanelOLS.from_formula(
    '''avg_logRCR ~ DID_chatgpt + DID_transformer
       + elite_univ_post + industry_top_post
       + region_US_post + region_CN_post
       + EntityEffects + TimeEffects''',
    data=panel_idx
).fit(cov_type='clustered', cluster_entity=True)

coef_E3, pval_E3, ci_E3 = extract_did(mE3)
print_did('E3', coef_E3, pval_E3, ci_E3)
print("  机构控制项：")
print_controls(mE3, [
    ('elite_univ_post',   'QS精英高校 × Post'),
    ('industry_top_post', '产业精英机构 × Post'),
])
print("  地域控制项：")
print_controls(mE3, [
    ('region_US_post', '美国 × Post'),
    ('region_CN_post', '中国 × Post'),
])

pct_E3 = abs(coef_E3 - coef_B2) / abs(coef_B2) * 100
print(f"  系数变化（B2→E3）：{pct_E3:.1f}%  → {'✅ 稳健' if pct_E3 < 10 else '⚠️ 存在混淆'}")

# ─────────────────────────────────────────────────────────────────
# 稳健性对比总表：B2 → E1 → E2 → E3
# ─────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("DID_chatgpt 系数稳健性总览")
print('='*65)
print(f"{'模型':<28} {'系数':>8} {'显著性':>6} {'p值':>8} {'CI':>22} {'变化%':>7}")
print('-'*65)

results = [
    ('B2：基准',              coef_B2, pval_B2, ci_B2, None),
    ('E1：+机构控制',         coef_E1, pval_E1, ci_E1, pct_E1),
    ('E2：+地域控制',         coef_E2, pval_E2, ci_E2, pct_E2),
    ('E3：+机构+地域 ★',     coef_E3, pval_E3, ci_E3, pct_E3),
]
for label, coef, pval, ci, pct in results:
    pct_str = f"{pct:.1f}%" if pct is not None else "—"
    print(f"{label:<28} {coef:>+8.4f} {stars(pval):>6} {pval:>8.4f} "
          f"  [{ci['lower']:+.4f}, {ci['upper']:+.4f}]  {pct_str:>7}")

print("\n注：标准误按作者聚类（cluster-robust）")
print("    基准组 = 其他机构（非精英）+ 其他地区（非中美）")

# ─────────────────────────────────────────────────────────────────
# 子组分析：分地域回归 DID_chatgpt（美国 / 中国 / 其他）
# ─────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("子组分析：跨界合作效应的地域异质性")
print("研究问题：ChatGPT时代，跨界合作效应在中国是否被抵消？")
print('='*65)

region_results = {}

for region_name in ['US', 'CN', 'Other']:
    # 取该地域的作者ID
    region_authors = set(
        panel_clean[panel_clean['region'] == region_name]['author_id']
    )

    # 从全量面板中筛选该子组
    sub_idx = panel_idx[
        panel_idx.index.get_level_values('author_id').isin(region_authors)
    ]

    n_authors = sub_idx.index.get_level_values('author_id').nunique()
    n_rows    = len(sub_idx)

    if n_authors < 100:
        print(f"\n  [{region_name}] 样本量不足（{n_authors} 人），跳过")
        continue

    try:
        m_sub = PanelOLS.from_formula(
            'avg_logRCR ~ DID_chatgpt + DID_transformer + EntityEffects + TimeEffects',
            data=sub_idx
        ).fit(cov_type='clustered', cluster_entity=True)

        coef = m_sub.params['DID_chatgpt']
        pval = m_sub.pvalues['DID_chatgpt']
        ci   = m_sub.conf_int().loc['DID_chatgpt']

        region_results[region_name] = (coef, pval, ci, n_authors)

        label_map = {'US': '美国', 'CN': '中国', 'Other': '其他地区'}
        print(f"\n  [{label_map[region_name]}]  n={n_authors:,} 人 | {n_rows:,} 行")
        print(f"  DID_chatgpt：{coef:+.4f}  {stars(pval):<5}  "
              f"p={pval:.4f}  CI=[{ci['lower']:+.4f}, {ci['upper']:+.4f}]")

    except Exception as e:
        print(f"\n  [{region_name}] 回归失败：{e}")

# ─────────────────────────────────────────────────────────────────
# 子组汇总表
# ─────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("子组分析汇总表（基准对照：全样本 B2 = +0.0402*）")
print('='*65)
print(f"{'地域':<12} {'样本(人)':>10} {'DID系数':>10} {'显著性':>6} {'p值':>8} {'CI':>22}")
print('-'*65)

label_map = {'US': '美国', 'CN': '中国', 'Other': '其他地区'}
for region_name, (coef, pval, ci, n) in region_results.items():
    print(f"{label_map[region_name]:<12} {n:>10,} {coef:>+10.4f} {stars(pval):>6} "
          f"{pval:>8.4f}   [{ci['lower']:+.4f}, {ci['upper']:+.4f}]")

# ─────────────────────────────────────────────────────────────────
# 子组解读提示
# ─────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("解读指引")
print('='*65)

if 'CN' in region_results and 'US' in region_results:
    coef_cn, pval_cn, _, _ = region_results['CN']
    coef_us, pval_us, _, _ = region_results['US']

    print(f"\n  美国跨界合作效应：{coef_us:+.4f}  {'显著 ✅' if pval_us < 0.05 else '不显著'}")
    print(f"  中国跨界合作效应：{coef_cn:+.4f}  {'显著 ✅' if pval_cn < 0.05 else '不显著 ❌'}")
    print(f"  效应差值（美国－中国）：{coef_us - coef_cn:+.4f}")

    if pval_cn >= 0.05 and pval_us < 0.05:
        print("\n  ★ 结论：跨界合作效应在美国显著为正，在中国不显著")
        print("    → 支持ChatGPT可及性机制：AI工具的地域访问壁垒")
        print("       部分抵消了中国跨界合作者的引用溢价")
    elif pval_cn < 0.05 and coef_cn > 0:
        print("\n  ★ 结论：跨界合作效应在中美均显著为正")
        print("    → 跨界合作效应具有普遍性，超越地域AI可及性差异")
    elif coef_cn < 0:
        print("\n  ★ 结论：中国跨界合作者引用不升反降")
        print("    → ChatGPT访问受限 + 全球引用竞争加剧的双重压力")

print("\n注：子组回归使用作者固定效应 + 年份固定效应 + 实体聚类标准误")
