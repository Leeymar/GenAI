import pandas as pd
import numpy as np

panel = pd.read_csv('Ultimate_DID_Panel.csv')
panel['year'] = panel['year'].astype(int)

# ── 修复一：截断年份，只保留2012-2024 ──────────────────────────
panel = panel[panel['year'].between(2012, 2024)]
print(f"截断后行数：{len(panel):,}")

# ── 修复二：检查并修正因变量 ────────────────────────────────────
print("\n检查 avg_logRCR 的构建方式：")
print(f"  当前最小值：{panel['avg_logRCR'].min():.4f}")
print(f"  当前负值数量：{(panel['avg_logRCR'] < 0).sum():,}")

# 检查原始RCR列是否存在
rcr_cols = [c for c in panel.columns if 'rcr' in c.lower() or 'RCR' in c]
print(f"\n  数据中所有RCR相关列：{rcr_cols}")

# 如果有原始RCR列，重新正确计算log(RCR)
# ⚠️ 以下假设原始列名为 avg_RCR，根据实际情况修改
if 'avg_RCR' in panel.columns:
    panel['avg_logRCR_correct'] = np.log(panel['avg_RCR'].clip(lower=1e-6))
    print(f"\n  重新计算后最小值：{panel['avg_logRCR_correct'].min():.4f}")
    print(f"  重新计算后负值数量：{(panel['avg_logRCR_correct'] < 0).sum():,}")
    print(f"  重新计算后均值：{panel['avg_logRCR_correct'].mean():.4f}")
    # 用修正后的值替换
    panel['avg_logRCR'] = panel['avg_logRCR_correct']

# ── 修复三：清洗 ────────────────────────────────────────────────
panel['is_academic_elite'] = panel['is_academic_elite'].fillna(0).astype(int)
panel['is_industry_elite']  = panel['is_industry_elite'].fillna(0).astype(int)
panel.loc[panel['is_academic_elite'] == 1, 'is_industry_elite'] = 0
panel['country_code'] = panel['country_code'].replace('Unknown', np.nan).fillna('Other')

panel['Post_chatgpt']     = (panel['year'] >= 2022).astype(int)
panel['Post_transformer'] = (panel['year'] >= 2017).astype(int)

panel['region'] = panel['country_code'].apply(
    lambda x: 'US' if x == 'US' else ('CN' if x == 'CN' else 'Other')
)
panel['is_CN'] = (panel['region'] == 'CN').astype(int)
panel['is_US'] = (panel['region'] == 'US').astype(int)

panel_clean = panel.dropna(subset=['avg_logRCR', 'is_treated'])
panel_clean = panel_clean[~np.isinf(panel_clean['avg_logRCR'])]

print(f"\n✅ 修复后面板：{len(panel_clean):,} 行 | "
      f"作者：{panel_clean['author_id'].nunique():,} 人")

# ── 重新输出描述性统计 ──────────────────────────────────────────
print("\n修复后因变量分布：")
desc = panel_clean['avg_logRCR'].describe(percentiles=[.25, .5, .75])
for k, v in desc.items():
    print(f"  {k:<12}：{v:.4f}")
print(f"  负值比例   ：{(panel_clean['avg_logRCR'] < 0).mean()*100:.1f}%")

# ── 重新输出2×2表 ───────────────────────────────────────────────
print("\n修复后 2×2 均值矩阵：")
pre  = panel_clean[panel_clean['Post_chatgpt'] == 0]
post = panel_clean[panel_clean['Post_chatgpt'] == 1]

for label, t in [('处理组（跨界合作）', 1), ('对照组（单一领域）', 0)]:
    pre_val  = pre[pre['is_treated'] == t]['avg_logRCR'].mean()
    post_val = post[post['is_treated'] == t]['avg_logRCR'].mean()
    diff     = post_val - pre_val
    print(f"  {label:<14}  Pre={pre_val:>+.4f}  Post={post_val:>+.4f}  Δ={diff:>+.4f}")

pre_t  = pre[pre['is_treated']==1]['avg_logRCR'].mean()
pre_c  = pre[pre['is_treated']==0]['avg_logRCR'].mean()
post_t = post[post['is_treated']==1]['avg_logRCR'].mean()
post_c = post[post['is_treated']==0]['avg_logRCR'].mean()
naive_did = (post_t - pre_t) - (post_c - pre_c)
print(f"\n  原始DID估算：{naive_did:>+.4f}")

# ── 年度趋势 ────────────────────────────────────────────────────
print("\n修复后年度趋势：")
print(f"  {'年份':>6} {'处理组':>10} {'对照组':>10} {'差值':>10} {'时段':>14}")
print(f"  {'-'*55}")
annual = (panel_clean.groupby(['year','is_treated'])['avg_logRCR']
          .mean().unstack('is_treated')
          .rename(columns={0:'对照组', 1:'处理组'}))
annual['差值'] = annual['处理组'] - annual['对照组']
for yr, row in annual.iterrows():
    period = '基准期' if yr < 2017 else ('Transformer期' if yr < 2022 else 'ChatGPT期 ★')
    print(f"  {yr:>6} {row['处理组']:>10.4f} {row['对照组']:>10.4f} "
          f"{row['差值']:>10.4f} {period:>14}")
