import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────
# 加载面板
# ─────────────────────────────────────────────────────────────────
panel = pd.read_csv('DID_Panel_Ready.csv')
print(f"✅ 面板加载完成：{len(panel):,} 行 | 作者 {panel['author_id'].nunique():,} 人")

# ─────────────────────────────────────────────────────────────────
# 构建 baseline_rcr 协变量（每位作者的处理前基准）
# ─────────────────────────────────────────────────────────────────
panel['relative_year'] = pd.to_numeric(panel['relative_year'], errors='coerce')

treat_pre = (
    panel[(panel['is_treated'] == 1) & (panel['relative_year'] < 0)]
    .groupby('author_id')['avg_logRCR']
    .mean()
    .reset_index()
    .rename(columns={'avg_logRCR': 'baseline_rcr'})
)
ctrl_all = (
    panel[panel['is_treated'] == 0]
    .groupby('author_id')['avg_logRCR']
    .mean()
    .reset_index()
    .rename(columns={'avg_logRCR': 'baseline_rcr'})
)
baseline = pd.concat([treat_pre, ctrl_all], ignore_index=True)
panel = panel.merge(baseline, on='author_id', how='left')
panel['baseline_rcr'] = panel['baseline_rcr'].fillna(panel['baseline_rcr'].median())

# ─────────────────────────────────────────────────────────────────
# 固定效应编码
# ─────────────────────────────────────────────────────────────────
panel['relative_year'] = panel['relative_year'].fillna(0)
panel['field_fe']      = panel['primary_origin_field'].fillna('Unknown').astype(str)
panel['year_fe']       = panel['year'].astype(str)
panel['cohort_fe']     = panel['event_year'].astype(str)   # 队列固定效应

# ─────────────────────────────────────────────────────────────────
# 模型 1：基础 DID
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("模型 1：基础 DID（无 baseline 控制）")
print("=" * 60)

m1 = smf.ols(
    'avg_logRCR ~ DID_term + is_treated + Post'
    ' + C(year_fe) + C(field_fe)',
    data=panel
).fit(cov_type='HC3')

print(f"DID 系数（跨界效应）：{m1.params['DID_term']:+.4f}")
print(f"p 值                ：{m1.pvalues['DID_term']:.4f}"
      f"  {'***' if m1.pvalues['DID_term']<0.001 else '**' if m1.pvalues['DID_term']<0.01 else '*' if m1.pvalues['DID_term']<0.05 else 'n.s.'}")
print(f"95% CI              ：[{m1.conf_int().loc['DID_term',0]:+.4f}, "
      f"{m1.conf_int().loc['DID_term',1]:+.4f}]")
print(f"样本量              ：{int(m1.nobs):,}")
print(f"R²                  ：{m1.rsquared:.4f}")

# ─────────────────────────────────────────────────────────────────
# 模型 2：RA-DID（主模型）
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("模型 2：RA-DID（控制 baseline_rcr）← 主模型")
print("=" * 60)

m2 = smf.ols(
    'avg_logRCR ~ DID_term + is_treated + Post + baseline_rcr'
    ' + C(year_fe) + C(field_fe)',
    data=panel
).fit(cov_type='HC3')

print(f"DID 系数（跨界效应）：{m2.params['DID_term']:+.4f}")
print(f"p 值                ：{m2.pvalues['DID_term']:.4f}"
      f"  {'***' if m2.pvalues['DID_term']<0.001 else '**' if m2.pvalues['DID_term']<0.01 else '*' if m2.pvalues['DID_term']<0.05 else 'n.s.'}")
print(f"95% CI              ：[{m2.conf_int().loc['DID_term',0]:+.4f}, "
      f"{m2.conf_int().loc['DID_term',1]:+.4f}]")
print(f"样本量              ：{int(m2.nobs):,}")
print(f"R²                  ：{m2.rsquared:.4f}")

# ─────────────────────────────────────────────────────────────────
# 模型 3：RA-DID + 队列固定效应（处理 2023 年队列偏重问题）
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("模型 3：RA-DID + 队列固定效应（稳健性检验）")
print("=" * 60)

m3 = smf.ols(
    'avg_logRCR ~ DID_term + is_treated + Post + baseline_rcr'
    ' + C(year_fe) + C(field_fe) + C(cohort_fe)',
    data=panel
).fit(cov_type='HC3')

print(f"DID 系数（跨界效应）：{m3.params['DID_term']:+.4f}")
print(f"p 值                ：{m3.pvalues['DID_term']:.4f}"
      f"  {'***' if m3.pvalues['DID_term']<0.001 else '**' if m3.pvalues['DID_term']<0.01 else '*' if m3.pvalues['DID_term']<0.05 else 'n.s.'}")
print(f"95% CI              ：[{m3.conf_int().loc['DID_term',0]:+.4f}, "
      f"{m3.conf_int().loc['DID_term',1]:+.4f}]")
print(f"样本量              ：{int(m3.nobs):,}")
print(f"R²                  ：{m3.rsquared:.4f}")

# ─────────────────────────────────────────────────────────────────
# 模型 4：RA-DID + 论文数量控制
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("模型 4：RA-DID + 论文数量控制")
print("=" * 60)

m4 = smf.ols(
    'avg_logRCR ~ DID_term + is_treated + Post + baseline_rcr'
    ' + yearly_paper_count + C(year_fe) + C(field_fe)',
    data=panel
).fit(cov_type='HC3')

print(f"DID 系数（跨界效应）：{m4.params['DID_term']:+.4f}")
print(f"p 值                ：{m4.pvalues['DID_term']:.4f}"
      f"  {'***' if m4.pvalues['DID_term']<0.001 else '**' if m4.pvalues['DID_term']<0.01 else '*' if m4.pvalues['DID_term']<0.05 else 'n.s.'}")
print(f"95% CI              ：[{m4.conf_int().loc['DID_term',0]:+.4f}, "
      f"{m4.conf_int().loc['DID_term',1]:+.4f}]")
print(f"样本量              ：{int(m4.nobs):,}")
print(f"R²                  ：{m4.rsquared:.4f}")

# ─────────────────────────────────────────────────────────────────
# 汇总表
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("四模型 DID 系数汇总")
print("=" * 60)

def sig_stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'n.s.'

results = pd.DataFrame([
    {
        '模型'      : '① 基础 DID',
        'DID 系数'  : round(m1.params['DID_term'], 4),
        '显著性'    : sig_stars(m1.pvalues['DID_term']),
        'p 值'      : round(m1.pvalues['DID_term'], 4),
        'CI 下限'   : round(m1.conf_int().loc['DID_term', 0], 4),
        'CI 上限'   : round(m1.conf_int().loc['DID_term', 1], 4),
        'R²'        : round(m1.rsquared, 4),
        '控制变量'  : '年份、学科 FE',
    },
    {
        '模型'      : '② RA-DID（主模型）',
        'DID 系数'  : round(m2.params['DID_term'], 4),
        '显著性'    : sig_stars(m2.pvalues['DID_term']),
        'p 值'      : round(m2.pvalues['DID_term'], 4),
        'CI 下限'   : round(m2.conf_int().loc['DID_term', 0], 4),
        'CI 上限'   : round(m2.conf_int().loc['DID_term', 1], 4),
        'R²'        : round(m2.rsquared, 4),
        '控制变量'  : '+baseline_rcr',
    },
    {
        '模型'      : '③ RA-DID + 队列 FE',
        'DID 系数'  : round(m3.params['DID_term'], 4),
        '显著性'    : sig_stars(m3.pvalues['DID_term']),
        'p 值'      : round(m3.pvalues['DID_term'], 4),
        'CI 下限'   : round(m3.conf_int().loc['DID_term', 0], 4),
        'CI 上限'   : round(m3.conf_int().loc['DID_term', 1], 4),
        'R²'        : round(m3.rsquared, 4),
        '控制变量'  : '+cohort FE',
    },
    {
        '模型'      : '④ RA-DID + 论文数',
        'DID 系数'  : round(m4.params['DID_term'], 4),
        '显著性'    : sig_stars(m4.pvalues['DID_term']),
        'p 值'      : round(m4.pvalues['DID_term'], 4),
        'CI 下限'   : round(m4.conf_int().loc['DID_term', 0], 4),
        'CI 上限'   : round(m4.conf_int().loc['DID_term', 1], 4),
        'R²'        : round(m4.rsquared, 4),
        '控制变量'  : '+论文数量',
    },
])

print(results.to_string(index=False))
print("\n注：* p<0.05  ** p<0.01  *** p<0.001  HC3 稳健标准误")
