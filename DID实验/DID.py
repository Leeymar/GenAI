# ============================================================
# ChatGPT DID 分析（最终完整版）
# 处理时点：ChatGPT = 2022，Transformer = 2017
# 修复：事件研究中移除 DID_transformer，加 drop_absorbed=True
# ============================================================

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from linearmodels.panel import PanelOLS
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 第一步：加载数据 & 筛选列
# ============================================================
panel = pd.read_csv('DID_Panel_Ready.csv')
panel['year'] = panel['year'].astype(int)

keep_cols = ['author_id', 'year', 'avg_logRCR', 'is_treated', 'yearly_paper_count']
keep_cols = [c for c in keep_cols if c in panel.columns]
panel = panel[keep_cols]

panel = panel.dropna(subset=['avg_logRCR', 'year', 'author_id', 'is_treated'])
panel = panel[~np.isinf(panel['avg_logRCR'])]

print(f"数据：{len(panel):,} 行 | 作者：{panel['author_id'].nunique():,} 人")
print(f"年份范围：{panel['year'].min()} – {panel['year'].max()}")

# ============================================================
# 第二步：构造 DID 变量
# ChatGPT 处理时点 = 2022
# Transformer 冲击时点 = 2017
# ============================================================
panel['Post_chatgpt']     = (panel['year'] >= 2022).astype(int)
panel['DID_chatgpt']      = panel['is_treated'] * panel['Post_chatgpt']

panel['Post_transformer'] = (panel['year'] >= 2017).astype(int)
panel['DID_transformer']  = panel['is_treated'] * panel['Post_transformer']

print("\nDID 变量描述：")
print(panel[['DID_chatgpt', 'DID_transformer']].describe())

# ============================================================
# 第三步：构建 panel_idx（TWFE 用）
# ============================================================
panel_idx = (
    panel.set_index(['author_id', 'year'])
    .groupby(level=['author_id', 'year'])
    .mean(numeric_only=True)
)

# ============================================================
# 第四步：模型 A —— 传统 OLS DID（对照基准）
# ============================================================
mA = smf.ols(
    'avg_logRCR ~ is_treated + Post_chatgpt + DID_chatgpt',
    data=panel
).fit(cov_type='cluster', cov_kwds={'groups': panel['author_id']})

coef_A = mA.params['DID_chatgpt']
pval_A = mA.pvalues['DID_chatgpt']
ci_A   = mA.conf_int().loc['DID_chatgpt']

print("\n" + "="*60)
print("模型 A：传统 OLS DID（基准，不含 Transformer 控制）")
print("="*60)
print(mA.summary().tables[1])

# ============================================================
# 第五步：模型 B —— TWFE（主模型，不含 Transformer 控制）
# ============================================================
mB = PanelOLS.from_formula(
    'avg_logRCR ~ DID_chatgpt + EntityEffects + TimeEffects',
    data=panel_idx
).fit(cov_type='clustered', cluster_entity=True)

coef_B = mB.params['DID_chatgpt']
pval_B = mB.pvalues['DID_chatgpt']
ci_B   = mB.conf_int().loc['DID_chatgpt']

print("\n" + "="*60)
print("模型 B：TWFE 主模型（不含 Transformer 控制）")
print("="*60)
print(f"DID_chatgpt 系数 ：{coef_B:+.4f}")
print(f"p 值             ：{pval_B:.4f}")
print(f"95% CI           ：[{ci_B['lower']:+.4f}, {ci_B['upper']:+.4f}]")
print(f"样本量           ：{int(mB.nobs):,}")

# ============================================================
# 第六步：模型 B2 —— TWFE + Transformer 控制（核心稳健性）★
# ============================================================
mB2 = PanelOLS.from_formula(
    'avg_logRCR ~ DID_chatgpt + DID_transformer + EntityEffects + TimeEffects',
    data=panel_idx
).fit(cov_type='clustered', cluster_entity=True)

coef_B2_gpt = mB2.params['DID_chatgpt']
pval_B2_gpt = mB2.pvalues['DID_chatgpt']
ci_B2_gpt   = mB2.conf_int().loc['DID_chatgpt']

coef_B2_tfm = mB2.params['DID_transformer']
pval_B2_tfm = mB2.pvalues['DID_transformer']
ci_B2_tfm   = mB2.conf_int().loc['DID_transformer']

print("\n" + "="*60)
print("模型 B2：TWFE + Transformer 冲击控制 ★ 核心稳健性")
print("="*60)
print(f"DID_chatgpt 系数     ：{coef_B2_gpt:+.4f}  p={pval_B2_gpt:.4f}")
print(f"  95% CI             ：[{ci_B2_gpt['lower']:+.4f}, {ci_B2_gpt['upper']:+.4f}]")
print(f"DID_transformer 系数 ：{coef_B2_tfm:+.4f}  p={pval_B2_tfm:.4f}")
print(f"  95% CI             ：[{ci_B2_tfm['lower']:+.4f}, {ci_B2_tfm['upper']:+.4f}]")
print(f"样本量               ：{int(mB2.nobs):,}")

# ============================================================
# 第七步：模型 C —— TWFE + Transformer + yearly_paper_count
# ============================================================
if 'yearly_paper_count' in panel_idx.columns:
    mC = PanelOLS.from_formula(
        'avg_logRCR ~ DID_chatgpt + DID_transformer + yearly_paper_count + EntityEffects + TimeEffects',
        data=panel_idx
    ).fit(cov_type='clustered', cluster_entity=True)

    coef_C = mC.params['DID_chatgpt']
    pval_C = mC.pvalues['DID_chatgpt']
    ci_C   = mC.conf_int().loc['DID_chatgpt']

    print("\n" + "="*60)
    print("模型 C：TWFE + Transformer + yearly_paper_count")
    print("="*60)
    print(f"DID_chatgpt 系数 ：{coef_C:+.4f}  p={pval_C:.4f}")
    print(f"95% CI           ：[{ci_C['lower']:+.4f}, {ci_C['upper']:+.4f}]")
else:
    print("⚠️ yearly_paper_count 不存在，模型 C 以模型 B2 代替")
    coef_C, pval_C, ci_C = coef_B2_gpt, pval_B2_gpt, ci_B2_gpt

# ============================================================
# 第八步：模型 D —— 安慰剂检验（伪处理时点 = 2020）
# ============================================================
panel_placebo = panel[panel['year'] < 2022].copy()
panel_placebo['Post_fake']        = (panel_placebo['year'] >= 2020).astype(int)
panel_placebo['DID_fake']         = panel_placebo['is_treated'] * panel_placebo['Post_fake']
panel_placebo['Post_transformer'] = (panel_placebo['year'] >= 2017).astype(int)
panel_placebo['DID_transformer']  = panel_placebo['is_treated'] * panel_placebo['Post_transformer']

panel_placebo_idx = (
    panel_placebo.set_index(['author_id', 'year'])
    .groupby(level=['author_id', 'year'])
    .mean(numeric_only=True)
)

mD = PanelOLS.from_formula(
    'avg_logRCR ~ DID_fake + DID_transformer + EntityEffects + TimeEffects',
    data=panel_placebo_idx
).fit(cov_type='clustered', cluster_entity=True)

coef_D = mD.params['DID_fake']
pval_D = mD.pvalues['DID_fake']
ci_D   = mD.conf_int().loc['DID_fake']
sig_D  = "✅ 安慰剂通过（不显著）" if pval_D > 0.05 else "❌ 安慰剂未通过（显著）"

print("\n" + "="*60)
print("模型 D：安慰剂检验（伪处理时点 = 2020，含 Transformer 控制）")
print("="*60)
print(f"DID_fake 系数    ：{coef_D:+.4f}")
print(f"p 值             ：{pval_D:.4f}  {sig_D}")
print(f"95% CI           ：[{ci_D['lower']:+.4f}, {ci_D['upper']:+.4f}]")

# ============================================================
# 第九步：汇总表
# ============================================================
def sig_star(p):
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    else: return 'n.s.'

summary = pd.DataFrame({
    '模型': [
        'A：OLS DID（基准）',
        'B：TWFE（无 Transformer 控制）',
        'B2：TWFE + Transformer 控制 ★',
        'C：TWFE + Transformer + 发表量',
        'D：安慰剂（伪2020）'
    ],
    '系数': [coef_A, coef_B, coef_B2_gpt, coef_C, coef_D],
    '显著性': [
        sig_star(pval_A), sig_star(pval_B),
        sig_star(pval_B2_gpt), sig_star(pval_C), sig_star(pval_D)
    ],
    'p 值': [
        round(pval_A,4), round(pval_B,4),
        round(pval_B2_gpt,4), round(pval_C,4), round(pval_D,4)
    ],
    'CI 下限': [
        round(ci_A[0],4), round(ci_B['lower'],4),
        round(ci_B2_gpt['lower'],4), round(ci_C['lower'],4), round(ci_D['lower'],4)
    ],
    'CI 上限': [
        round(ci_A[1],4), round(ci_B['upper'],4),
        round(ci_B2_gpt['upper'],4), round(ci_C['upper'],4), round(ci_D['upper'],4)
    ],
})

print("\n" + "="*70)
print("完整模型汇总（DID_chatgpt 系数）")
print("="*70)
print(summary.to_string(index=False))
print("\n注：标准误按作者聚类（cluster-robust）")
print("    * p<0.05  ** p<0.01  *** p<0.001")
print("    ★ 模型 B2 为核心稳健性证据")
summary.to_csv('model_summary.csv', index=False, encoding='utf-8-sig')
print("✅ 汇总表已保存：model_summary.csv")

# ============================================================
# 第十步（修复版）：平行趋势检验（事件研究图）
# 基准年 = 2021
# ============================================================
panel_es  = panel.copy()
years     = sorted(panel_es['year'].unique())
base_year = 2021

for yr in years:
    if yr != base_year:
        panel_es[f'D_{yr}'] = (
            (panel_es['year'] == yr) * panel_es['is_treated']
        ).astype(float)

interact_cols = [f'D_{yr}' for yr in years if yr != base_year]

panel_es_idx = (
    panel_es.set_index(['author_id', 'year'])
    .groupby(level=['author_id', 'year'])
    .mean(numeric_only=True)
)

formula_es = (
    f'avg_logRCR ~ {" + ".join(interact_cols)}'
    f' + EntityEffects + TimeEffects'
)

# ✅ drop_absorbed 在 from_formula() 中传入
mES = PanelOLS.from_formula(
    formula_es,
    data=panel_es_idx,
    drop_absorbed=True
).fit(
    cov_type='clustered',
    cluster_entity=True
)

# ---- 提取各年系数 ----
es_coefs, es_low, es_high, es_years = [], [], [], []

for yr in years:
    if yr == base_year:
        es_coefs.append(0.0)
        es_low.append(0.0)
        es_high.append(0.0)
    else:
        col = f'D_{yr}'
        if col in mES.params.index:
            es_coefs.append(mES.params[col])
            es_low.append(mES.conf_int().loc[col, 'lower'])
            es_high.append(mES.conf_int().loc[col, 'upper'])
        else:
            print(f"  ⚠️  {yr} 年被自动 drop（数据稀疏），图中标注缺失")
            es_coefs.append(np.nan)
            es_low.append(np.nan)
            es_high.append(np.nan)
    es_years.append(yr)

# ---- 构建结果 DataFrame ----
es_df = pd.DataFrame({
    'year':    es_years,
    'coef':    es_coefs,
    'ci_low':  es_low,
    'ci_high': es_high,
})
es_df['sig'] = es_df.apply(
    lambda r: '✅ 显著' if (
        pd.notna(r['ci_low']) and (r['ci_low'] > 0 or r['ci_high'] < 0)
    ) else '— 不显著',
    axis=1
)
es_df['period'] = es_df['year'].apply(
    lambda y: 'BASE' if y == base_year else ('POST' if y >= 2022 else 'PRE')
)

# ---- 打印 ----
print("\n" + "="*70)
print("平行趋势图（事件研究）：各年系数")
print("="*70)
print(f"{'年份':>6}  {'系数':>8}  {'CI 下限':>8}  {'CI 上限':>8}  {'期':>6}  显著性")
print("-"*70)
for _, row in es_df.iterrows():
    coef_str = f"{row['coef']:>+8.4f}" if pd.notna(row['coef']) else "    drop"
    low_str  = f"{row['ci_low']:>+8.4f}"  if pd.notna(row['ci_low'])  else "    drop"
    high_str = f"{row['ci_high']:>+8.4f}" if pd.notna(row['ci_high']) else "    drop"
    print(f"{int(row['year']):>6}  {coef_str}  {low_str}  {high_str}  "
          f"{row['period']:>6}  {row['sig']}")

pre_df  = es_df[(es_df['period'] == 'PRE')  & es_df['coef'].notna()]
post_df = es_df[(es_df['period'] == 'POST') & es_df['coef'].notna()]

print("\n" + "="*70)
print("平行趋势数值诊断")
print("="*70)
print(f"  Pre 期系数均值  ：{pre_df['coef'].mean():+.4f}（接近 0 则平行趋势成立）")
print(f"  Pre 期显著个数  ：{(pre_df['sig']=='✅ 显著').sum()} / {len(pre_df)}（应 = 0）")
print(f"  Post 期系数均值 ：{post_df['coef'].mean():+.4f}")
print(f"  Post 期显著个数 ：{(post_df['sig']=='✅ 显著').sum()} / {len(post_df)}")

es_df.to_csv('parallel_trend_data_v2.csv', index=False, encoding='utf-8-sig')
print("✅ 拐点数据已保存：parallel_trend_data_v2.csv")

# ---- 画图 ----
es_df_plot = es_df.dropna(subset=['coef'])

pre_mask  = es_df_plot['period'] == 'PRE'
post_mask = es_df_plot['period'] == 'POST'
base_mask = es_df_plot['period'] == 'BASE'

def get_vals(mask):
    sub = es_df_plot[mask]
    coefs = sub['coef'].tolist()
    errs  = [
        [c - l for c, l in zip(sub['coef'], sub['ci_low'])],
        [h - c for h, c in zip(sub['ci_high'], sub['coef'])]
    ]
    yrs = sub['year'].tolist()
    return yrs, coefs, errs

pre_yrs,  pre_c,  pre_e  = get_vals(pre_mask)
post_yrs, post_c, post_e = get_vals(post_mask)
base_yrs, base_c, base_e = get_vals(base_mask)

fig, ax = plt.subplots(figsize=(14, 5))

ax.errorbar(pre_yrs,  pre_c,  yerr=pre_e,
            fmt='o-', color='steelblue',  linewidth=2, capsize=4,
            label='Pre 期系数')
ax.errorbar(post_yrs, post_c, yerr=post_e,
            fmt='s-', color='darkorange', linewidth=2, capsize=4,
            label='Post 期系数（ChatGPT 效应）')
ax.errorbar(base_yrs, base_c, yerr=base_e,
            fmt='D',  color='black',      markersize=8,
            label=f'基准年（{base_year}）')

ax.axhline(0, color='gray',   linestyle='--', linewidth=1)
ax.axvline(2021.5, color='red',    linestyle='--', linewidth=1.5,
           label='ChatGPT 发布（2022）')
ax.axvline(2016.5, color='purple', linestyle=':',  linewidth=1.5,
           label='Transformer 发布（2017，已隐式控制）')

ax.set_xlabel('年份', fontsize=12)
ax.set_ylabel('avg_logRCR 相对变化', fontsize=12)
ax.set_title(
    '平行趋势检验：事件研究图\n（基准年 = 2021，Transformer 冲击已通过逐年交互项隐式控制）',
    fontsize=13
)
ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
ax.legend(fontsize=9, loc='upper left')
plt.tight_layout()
plt.savefig('parallel_trend_chatgpt_v3.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ 平行趋势图已保存：parallel_trend_chatgpt_v3.png")


# ============================================================
# 第十一步：Forest Plot（五模型对比）
# ============================================================
fig2, ax2 = plt.subplots(figsize=(10, 5))

labels_fp = [
    'A：OLS DID',
    'B：TWFE\n（无 Transformer 控制）',
    'B2：TWFE\n＋Transformer 控制 ★',
    'C：TWFE＋Transformer\n＋yearly_paper_count',
    'D：安慰剂\n（伪2020）'
]
coefs_fp  = [coef_A, coef_B, coef_B2_gpt, coef_C, coef_D]
low_fp    = [ci_A[0], ci_B['lower'], ci_B2_gpt['lower'], ci_C['lower'], ci_D['lower']]
high_fp   = [ci_A[1], ci_B['upper'], ci_B2_gpt['upper'], ci_C['upper'], ci_D['upper']]
colors_fp = ['steelblue', 'steelblue', 'darkgreen', 'steelblue', 'gray']
sizes_fp  = [8, 8, 12, 8, 8]

for i, (lbl, coef, lo, hi, col, sz) in enumerate(
        zip(labels_fp, coefs_fp, low_fp, high_fp, colors_fp, sizes_fp)):
    lw = 3 if col == 'darkgreen' else 2
    ax2.plot([lo, hi], [i, i], color=col, linewidth=lw, alpha=0.8)
    ax2.plot(coef, i, 'o', color=col, markersize=sz)
    ax2.text(hi + 0.003, i, f'{coef:+.4f}', va='center', fontsize=9, color=col)

ax2.axvline(0, color='red', linestyle='--', linewidth=1)
ax2.set_yticks(range(len(labels_fp)))
ax2.set_yticklabels(labels_fp, fontsize=10)
ax2.set_xlabel('DID_chatgpt 系数（ChatGPT 对 log-RCR 的净效应）', fontsize=11)
ax2.set_title('五模型系数对比（Forest Plot）\n★ = 核心稳健性模型', fontsize=12)
plt.tight_layout()
plt.savefig('forest_plot_chatgpt_v2.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Forest Plot 已保存：forest_plot_chatgpt_v2.png")
