import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────
# 加载数据
# ─────────────────────────────────────────────────────────────────
panel = pd.read_csv('Ultimate_DID_Panel.csv')
panel['year'] = panel['year'].astype(int)

panel['is_academic_elite'] = panel['is_academic_elite'].fillna(0).astype(int)
panel['is_industry_elite']  = panel['is_industry_elite'].fillna(0).astype(int)
panel.loc[panel['is_academic_elite'] == 1, 'is_industry_elite'] = 0
panel['country_code'] = panel['country_code'].replace('Unknown', np.nan).fillna('Other')

panel['Post_chatgpt']     = (panel['year'] >= 2022).astype(int)
panel['Post_transformer'] = (panel['year'] >= 2017).astype(int)
panel['DID_chatgpt']      = panel['is_treated'] * panel['Post_chatgpt']

panel['region'] = panel['country_code'].apply(
    lambda x: 'US' if x == 'US' else ('CN' if x == 'CN' else 'Other')
)
panel['is_CN'] = (panel['region'] == 'CN').astype(int)
panel['is_US'] = (panel['region'] == 'US').astype(int)

panel_clean = panel.dropna(subset=['avg_logRCR', 'is_treated'])
panel_clean = panel_clean[~np.isinf(panel_clean['avg_logRCR'])]

# ─────────────────────────────────────────────────────────────────
# 第一部分：整体样本描述性统计
# ─────────────────────────────────────────────────────────────────
print("=" * 70)
print("表1：主要变量描述性统计")
print("=" * 70)

vars_desc = {
    'avg_logRCR'       : '对数RCR均值（因变量）',
    'is_treated'       : '跨界合作者（处理组）',
    'is_academic_elite': '精英高校归属',
    'is_industry_elite': '产业精英机构归属',
    'is_US'            : '美国',
    'is_CN'            : '中国',
    'Post_chatgpt'     : 'ChatGPT后（Post≥2022）',
    'Post_transformer' : 'Transformer后（Post≥2017）',
}

rows = []
for var, label in vars_desc.items():
    col = panel_clean[var].dropna()
    rows.append({
        '变量': label,
        '观测数': f"{len(col):,}",
        '均值': f"{col.mean():.4f}",
        '标准差': f"{col.std():.4f}",
        '最小值': f"{col.min():.4f}",
        'P25': f"{col.quantile(0.25):.4f}",
        '中位数': f"{col.median():.4f}",
        'P75': f"{col.quantile(0.75):.4f}",
        '最大值': f"{col.max():.4f}",
    })

df_desc = pd.DataFrame(rows)
print(df_desc.to_string(index=False))

# ─────────────────────────────────────────────────────────────────
# 第二部分：处理组 vs 对照组 对比
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("表2：处理组 vs 对照组 — 引用影响力对比")
print("=" * 70)

# 按处理状态分组
treated_group = panel_clean[panel_clean['is_treated'] == 1]
control_group = panel_clean[panel_clean['is_treated'] == 0]

def group_stats(df, label):
    return {
        '组别': label,
        '作者数': f"{df['author_id'].nunique():,}",
        '观测数': f"{len(df):,}",
        'RCR均值': f"{df['avg_logRCR'].mean():.4f}",
        'RCR标准差': f"{df['avg_logRCR'].std():.4f}",
        'RCR中位数': f"{df['avg_logRCR'].median():.4f}",
    }

rows2 = [
    group_stats(panel_clean, '全样本'),
    group_stats(treated_group, '处理组（跨界合作者）'),
    group_stats(control_group, '对照组（单一领域）'),
]
df_group = pd.DataFrame(rows2)
print(df_group.to_string(index=False))

# ─────────────────────────────────────────────────────────────────
# 第三部分：处理组 vs 对照组 × ChatGPT 前后 — 2×2 均值表
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("表3：2×2 均值矩阵（处理组/对照组 × ChatGPT前/后）")
print("=" * 70)

pre  = panel_clean[panel_clean['Post_chatgpt'] == 0]
post = panel_clean[panel_clean['Post_chatgpt'] == 1]

def cell(df, treated):
    sub = df[df['is_treated'] == treated]['avg_logRCR']
    return f"{sub.mean():.4f}  (n={len(sub):,})"

print(f"  {'':20} {'ChatGPT前（<2022）':>28} {'ChatGPT后（≥2022）':>28} {'差值（后-前）':>12}")
print(f"  {'-'*75}")
for label, t in [('处理组（跨界合作）', 1), ('对照组（单一领域）', 0)]:
    pre_val  = pre[pre['is_treated'] == t]['avg_logRCR'].mean()
    post_val = post[post['is_treated'] == t]['avg_logRCR'].mean()
    diff     = post_val - pre_val
    pre_n    = len(pre[pre['is_treated'] == t])
    post_n   = len(post[post['is_treated'] == t])
    print(f"  {label:<20} {pre_val:>+.4f}  (n={pre_n:,})       "
          f"{post_val:>+.4f}  (n={post_n:,})       {diff:>+.4f}")

# DID 简单估算
pre_t  = pre[pre['is_treated']==1]['avg_logRCR'].mean()
pre_c  = pre[pre['is_treated']==0]['avg_logRCR'].mean()
post_t = post[post['is_treated']==1]['avg_logRCR'].mean()
post_c = post[post['is_treated']==0]['avg_logRCR'].mean()
naive_did = (post_t - pre_t) - (post_c - pre_c)
print(f"\n  原始DID估算（未控制固定效应）：{naive_did:>+.4f}")
print(f"  （参考：B2模型控制后 = +0.0402*）")

# ─────────────────────────────────────────────────────────────────
# 第四部分：分地域 × 处理状态 样本分布
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("表4：样本地域分布")
print("=" * 70)

author_level = panel_clean.drop_duplicates(subset=['author_id'])

for region_label, cond in [('美国', author_level['is_US']==1),
                             ('中国', author_level['is_CN']==1),
                             ('其他', (author_level['is_US']==0)&(author_level['is_CN']==0))]:
    sub = author_level[cond]
    n_total   = len(sub)
    n_treated = sub['is_treated'].sum()
    n_control = n_total - n_treated
    pct       = n_treated / n_total * 100 if n_total > 0 else 0
    print(f"  {region_label:<6} | 作者总数：{n_total:>7,} | "
          f"处理组：{int(n_treated):>6,} ({pct:.1f}%) | "
          f"对照组：{int(n_control):>6,}")

# ─────────────────────────────────────────────────────────────────
# 第五部分：年度均值趋势（用于平行趋势目视检验）
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("表5：年度引用均值趋势（处理组 vs 对照组）")
print("=" * 70)
print(f"  {'年份':>6} {'处理组':>12} {'对照组':>12} {'差值(T-C)':>12} {'时段':>12}")
print(f"  {'-'*55}")

annual = (panel_clean.groupby(['year', 'is_treated'])['avg_logRCR']
          .mean().unstack('is_treated')
          .rename(columns={0: '对照组', 1: '处理组'}))
annual['差值'] = annual['处理组'] - annual['对照组']

for yr, row in annual.iterrows():
    if yr < 2017:
        period = '基准期'
    elif yr < 2022:
        period = 'Transformer期'
    else:
        period = 'ChatGPT期 ★'
    print(f"  {yr:>6} {row['处理组']:>12.4f} {row['对照组']:>12.4f} "
          f"{row['差值']:>12.4f} {period:>12}")

# ─────────────────────────────────────────────────────────────────
# 第六部分：输出 LaTeX 格式描述性统计表
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("LaTeX 描述性统计表（可直接粘贴）")
print("=" * 70)

latex_rows = []
for var, label in vars_desc.items():
    col = panel_clean[var].dropna()
    latex_rows.append(
        f"    {label} & {len(col):,} & {col.mean():.4f} & "
        f"{col.std():.4f} & {col.min():.4f} & "
        f"{col.median():.4f} & {col.max():.4f} \\\\"
    )

print(r"""
\begin{table}[htbp]
\centering
\caption{主要变量描述性统计}
\label{tab:descriptive}
\begin{tabular}{lrrrrrr}
\toprule
变量 & 观测数 & 均值 & 标准差 & 最小值 & 中位数 & 最大值 \\
\midrule""")
for r in latex_rows:
    print(r)
print(r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item 注：分析单元为作者-年份（author-year）。样本期为2012—2024年，
共28,891位作者，110,150个观测。对数RCR均值（avg\_logRCR）为作者当年
所发论文RCR的对数均值，RCR以领域-年份中位数归一化。
\end{tablenotes}
\end{table}""")
