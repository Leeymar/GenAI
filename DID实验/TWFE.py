import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────
# 加载 & 预处理
# ─────────────────────────────────────────────────────────────────
panel = pd.read_csv('DID_Panel_Ready.csv')
panel['relative_year'] = pd.to_numeric(panel['relative_year'], errors='coerce')
panel = panel.dropna(subset=['avg_logRCR', 'DID_term', 'is_treated', 'Post', 'year'])
panel = panel[~np.isinf(panel['avg_logRCR'])]
panel['year'] = panel['year'].astype(int)

print(f"✅ 清理后：{len(panel):,} 行 | 作者 {panel['author_id'].nunique():,} 人")

# ─────────────────────────────────────────────────────────────────
# 设置 MultiIndex + 处理重复索引
# ─────────────────────────────────────────────────────────────────
panel = panel.set_index(['author_id', 'year'])
dup = panel.index.duplicated().sum()
if dup > 0:
    print(f"⚠️  发现 {dup} 个重复索引，取均值合并...")
    panel = panel.groupby(level=['author_id', 'year']).mean()

print(f"✅ MultiIndex 面板：{len(panel):,} 行")

def stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'

def print_result(label, coef, pval, ci_lower, ci_upper, nobs):
    print(f"\n{'=' * 60}")
    print(label)
    print('=' * 60)
    print(f"系数                   ：{coef:+.4f}")
    print(f"p 值                   ：{pval:.4f}  {stars(pval)}")
    print(f"95% CI                 ：[{ci_lower:+.4f}, {ci_upper:+.4f}]")
    print(f"样本量                 ：{int(nobs):,}")

# ─────────────────────────────────────────────────────────────────
# 模型 A：TWFE DID（主模型）
# ─────────────────────────────────────────────────────────────────
mA = PanelOLS.from_formula(
    'avg_logRCR ~ DID_term + EntityEffects + TimeEffects',
    data=panel
).fit(cov_type='clustered', cluster_entity=True)

coef_A = mA.params['DID_term']
pval_A = mA.pvalues['DID_term']
ci_A   = mA.conf_int().loc['DID_term']
print_result(
    "模型 A：TWFE DID（作者 FE + 年份 FE）← 主模型",
    coef_A, pval_A, ci_A['lower'], ci_A['upper'], mA.nobs
)

# ─────────────────────────────────────────────────────────────────
# 模型 B：TWFE + 论文数量控制（稳健性）
# ─────────────────────────────────────────────────────────────────
mB = PanelOLS.from_formula(
    'avg_logRCR ~ DID_term + yearly_paper_count + EntityEffects + TimeEffects',
    data=panel
).fit(cov_type='clustered', cluster_entity=True)

coef_B = mB.params['DID_term']
pval_B = mB.pvalues['DID_term']
ci_B   = mB.conf_int().loc['DID_term']
print_result(
    "模型 B：TWFE + 论文数量控制（稳健性）",
    coef_B, pval_B, ci_B['lower'], ci_B['upper'], mB.nobs
)

# ─────────────────────────────────────────────────────────────────
# 模型 C：仅实验组内 Before/After
# ─────────────────────────────────────────────────────────────────
panel_treat = panel[panel['is_treated'] == 1].copy()

mC = PanelOLS.from_formula(
    'avg_logRCR ~ Post + EntityEffects + TimeEffects',
    data=panel_treat
).fit(cov_type='clustered', cluster_entity=True)

coef_C = mC.params['Post']
pval_C = mC.pvalues['Post']
ci_C   = mC.conf_int().loc['Post']
print_result(
    "模型 C：实验组内 Before/After（EntityEffects + TimeEffects）",
    coef_C, pval_C, ci_C['lower'], ci_C['upper'], mC.nobs
)

# ─────────────────────────────────────────────────────────────────
# 模型 D：安慰剂检验（对照组伪处理，year > 2019）
# ─────────────────────────────────────────────────────────────────
panel_ctrl = panel[panel['is_treated'] == 0].copy()
panel_ctrl['pseudo_post'] = (
    panel_ctrl.index.get_level_values('year') > 2019
).astype(float)

mD = PanelOLS.from_formula(
    'avg_logRCR ~ pseudo_post + EntityEffects + TimeEffects',
    data=panel_ctrl
).fit(cov_type='clustered', cluster_entity=True)

coef_D = mD.params['pseudo_post']
pval_D = mD.pvalues['pseudo_post']
ci_D   = mD.conf_int().loc['pseudo_post']
print_result(
    "模型 D：安慰剂检验（对照组伪处理，year > 2019）",
    coef_D, pval_D, ci_D['lower'], ci_D['upper'], mD.nobs
)

if pval_D >= 0.05:
    print("✅ 安慰剂检验通过：对照组无虚假处理效应")
else:
    print("⚠️  安慰剂检验未通过：可能存在时间趋势混淆，需额外检验")

# ─────────────────────────────────────────────────────────────────
# 完整汇总表
# ─────────────────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("完整模型汇总")
print('=' * 60)

summary = pd.DataFrame([
    {
        '模型'        : 'A：TWFE（主模型）',
        '系数'        : round(coef_A, 4),
        '显著性'      : stars(pval_A),
        'p 值'        : round(pval_A, 4),
        'CI 下限'     : round(ci_A['lower'], 4),
        'CI 上限'     : round(ci_A['upper'], 4),
        '样本'        : '全样本',
        '控制变量'    : '作者 FE + 年份 FE',
    },
    {
        '模型'        : 'B：TWFE + 论文数量',
        '系数'        : round(coef_B, 4),
        '显著性'      : stars(pval_B),
        'p 值'        : round(pval_B, 4),
        'CI 下限'     : round(ci_B['lower'], 4),
        'CI 上限'     : round(ci_B['upper'], 4),
        '样本'        : '全样本',
        '控制变量'    : '+ 论文数量',
    },
    {
        '模型'        : 'C：实验组 Before/After',
        '系数'        : round(coef_C, 4),
        '显著性'      : stars(pval_C),
        'p 值'        : round(pval_C, 4),
        'CI 下限'     : round(ci_C['lower'], 4),
        'CI 上限'     : round(ci_C['upper'], 4),
        '样本'        : '仅实验组',
        '控制变量'    : '作者 FE + 年份 FE',
    },
    {
        '模型'        : 'D：安慰剂（对照组伪处理）',
        '系数'        : round(coef_D, 4),
        '显著性'      : stars(pval_D),
        'p 值'        : round(pval_D, 4),
        'CI 下限'     : round(ci_D['lower'], 4),
        'CI 上限'     : round(ci_D['upper'], 4),
        '样本'        : '仅对照组',
        '控制变量'    : '作者 FE + 年份 FE',
    },
])

print(summary.to_string(index=False))
print("\n注：标准误按作者聚类（cluster-robust）")
print("    * p<0.05  ** p<0.01  *** p<0.001")
print("    模型 D 期待不显著（安慰剂通过条件）")
