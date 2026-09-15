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
# 第二步：数据质量检查
# ─────────────────────────────────────────────────────────────────
print("\n── 机构精英字段空值检查 ──")
print(panel[['is_academic_elite', 'is_industry_elite', 'is_elite_overall']].isnull().sum())

print("\n── country_code 分布（前10）──")
print(panel['country_code'].value_counts(dropna=False).head(10))
print(f"Unknown 比例：{(panel['country_code']=='Unknown').mean():.1%}")

# ─────────────────────────────────────────────────────────────────
# 第三步：字段清洗
# ─────────────────────────────────────────────────────────────────
# 精英字段空值填0（未知机构视为非精英）
panel['is_academic_elite'] = panel['is_academic_elite'].fillna(0).astype(int)
panel['is_industry_elite'] = panel['is_industry_elite'].fillna(0).astype(int)
panel['is_elite_overall']  = panel['is_elite_overall'].fillna(0).astype(int)

# 互斥性保证：同时命中的归为精英高校
panel.loc[panel['is_academic_elite'] == 1, 'is_industry_elite'] = 0

# Unknown country_code 统一处理为 'Other'
panel['country_code'] = panel['country_code'].replace('Unknown', np.nan).fillna('Other')

# ─────────────────────────────────────────────────────────────────
# 第四步：构造实验二的 Post 变量（独立于实验一的 Post 列）
# ─────────────────────────────────────────────────────────────────
panel['Post_chatgpt']     = (panel['year'] >= 2022).astype(int)  # ChatGPT节点
panel['Post_transformer'] = (panel['year'] >= 2017).astype(int)  # Transformer节点

# 实验二 DID 变量
panel['DID_chatgpt']      = panel['is_treated'] * panel['Post_chatgpt']
panel['DID_transformer']  = panel['is_treated'] * panel['Post_transformer']

# ─────────────────────────────────────────────────────────────────
# 第五步：构造地域分组（中美 vs 其他）
# ─────────────────────────────────────────────────────────────────
panel['region'] = panel['country_code'].apply(
    lambda x: 'US' if x == 'US' else ('CN' if x == 'CN' else 'Other')
)

print("\n── 地域分布（作者人数）──")
print(panel.groupby('region')['author_id'].nunique().sort_values(ascending=False))

# ─────────────────────────────────────────────────────────────────
# 第六步：构造机构 × Post 交互项（基准 = 其他机构）
# ─────────────────────────────────────────────────────────────────
panel['elite_univ_post']   = panel['is_academic_elite'] * panel['Post_chatgpt']
panel['industry_top_post'] = panel['is_industry_elite'] * panel['Post_chatgpt']

# ─────────────────────────────────────────────────────────────────
# 第七步：构造地域 × Post 交互项（基准 = 其他地区）
# ─────────────────────────────────────────────────────────────────
panel['region_US_post'] = (panel['region'] == 'US').astype(int) * panel['Post_chatgpt']
panel['region_CN_post'] = (panel['region'] == 'CN').astype(int) * panel['Post_chatgpt']

# ─────────────────────────────────────────────────────────────────
# 第八步：机构/地域分布校验
# ─────────────────────────────────────────────────────────────────
print("\n── 机构类型分布（作者人数）──")
print(f"精英高校（is_academic_elite=1）：{panel[panel['is_academic_elite']==1]['author_id'].nunique():,} 人")
print(f"产业精英（is_industry_elite=1）：{panel[panel['is_industry_elite']==1]['author_id'].nunique():,} 人")
print(f"其他机构                        ：{panel[(panel['is_academic_elite']==0) & (panel['is_industry_elite']==0)]['author_id'].nunique():,} 人")

# 处理组 vs 对照组 在机构/地域上的分布（平衡性检查）
print("\n── 处理组机构精英比例 ──")
for g, name in [(1, '处理组'), (0, '对照组')]:
    sub = panel[panel['is_treated'] == g].drop_duplicates('author_id')
    print(f"  {name}：精英高校 {sub['is_academic_elite'].mean():.1%} | "
          f"产业精英 {sub['is_industry_elite'].mean():.1%} | "
          f"美国 {(sub['region']=='US').mean():.1%} | "
          f"中国 {(sub['region']=='CN').mean():.1%}")

# ─────────────────────────────────────────────────────────────────
# 第九步：共线性检查
# ─────────────────────────────────────────────────────────────────
interact_cols = ['elite_univ_post', 'industry_top_post',
                 'region_US_post', 'region_CN_post']
corr_matrix = panel[interact_cols].corr()
print("\n── 交互项相关矩阵 ──")
print(corr_matrix.round(3))

high_corr = [
    (c1, c2) for i, c1 in enumerate(interact_cols)
    for c2 in interact_cols[i+1:]
    if abs(corr_matrix.loc[c1, c2]) > 0.7
]
if high_corr:
    print(f"⚠️  高相关对：{high_corr}（考虑合并）")
else:
    print("✅ 无严重共线性")

# ─────────────────────────────────────────────────────────────────
# 第十步：构建 MultiIndex 面板
# ─────────────────────────────────────────────────────────────────
panel = panel.dropna(subset=['avg_logRCR', 'is_treated'])
panel = panel[~np.isinf(panel['avg_logRCR'])]

panel_idx = (
    panel.set_index(['author_id', 'year'])
    .groupby(level=['author_id', 'year'])
    .mean(numeric_only=True)
)
print(f"\n✅ MultiIndex 面板：{len(panel_idx):,} 行 | "
      f"作者：{panel_idx.index.get_level_values(0).nunique():,} 人")

# ─────────────────────────────────────────────────────────────────
# 模型 B2：基准（不含机构/地域控制）
# ─────────────────────────────────────────────────────────────────
mB2 = PanelOLS.from_formula(
    'avg_logRCR ~ DID_chatgpt + DID_transformer + EntityEffects + TimeEffects',
    data=panel_idx
).fit(cov_type='clustered', cluster_entity=True)

coef_B2 = mB2.params['DID_chatgpt']
pval_B2 = mB2.pvalues['DID_chatgpt']
ci_B2   = mB2.conf_int().loc['DID_chatgpt']

# ─────────────────────────────────────────────────────────────────
# 模型 E3：完整控制（精英高校 + 产业精英 + 中美地域）★
# ─────────────────────────────────────────────────────────────────
mE3 = PanelOLS.from_formula(
    '''avg_logRCR ~ DID_chatgpt + DID_transformer
       + elite_univ_post + industry_top_post
       + region_US_post + region_CN_post
       + EntityEffects + TimeEffects''',
    data=panel_idx
).fit(cov_type='clustered', cluster_entity=True)

coef_E3 = mE3.params['DID_chatgpt']
pval_E3 = mE3.pvalues['DID_chatgpt']
ci_E3   = mE3.conf_int().loc['DID_chatgpt']

# ─────────────────────────────────────────────────────────────────
# 输出结果
# ─────────────────────────────────────────────────────────────────
def stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'

print(f"\n{'='*65}")
print("模型 B2 vs E3 稳健性对比")
print(f"{'='*65}")
print(f"{'模型':<30} {'系数':>8} {'显著性':>6} {'p值':>8} {'CI':>20}")
print("-"*65)
for label, coef, pval, ci in [
    ('B2：基准',             coef_B2, pval_B2, ci_B2),
    ('E3：+机构+地域 ★',    coef_E3, pval_E3, ci_E3),
]:
    print(f"{label:<30} {coef:>+8.4f} {stars(pval):>6} {pval:>8.4f} "
          f"  [{ci['lower']:+.4f}, {ci['upper']:+.4f}]")

pct = abs(coef_E3 - coef_B2) / abs(coef_B2) * 100
print(f"\n系数变化幅度：{pct:.1f}%  → ", end="")
print("✅ 稳健" if pct < 10 else "⚠️ 存在混淆，需讨论")

print(f"\n── 机构/地域控制项系数 ──")
for col, label in [
    ('elite_univ_post',   'QS精英高校 × Post'),
    ('industry_top_post', '产业精英机构 × Post'),
    ('region_US_post',    '美国 × Post'),
    ('region_CN_post',    '中国 × Post'),
]:
    c  = mE3.params[col]
    p  = mE3.pvalues[col]
    ci = mE3.conf_int().loc[col]
    print(f"  {label:<22}：{c:+.4f}  {stars(p)}  p={p:.4f}  "
          f"CI=[{ci['lower']:+.4f}, {ci['upper']:+.4f}]")

print("\n注：标准误按作者聚类（cluster-robust）")
print("    基准组 = 其他机构 + 其他地区")
