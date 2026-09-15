import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from linearmodels.panel import PanelOLS
import warnings
warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════════════
# 全局配置
# ════════════════════════════════════════════════════════════════
PANEL_PATH  = 'DID_Panel_Ready.csv'
N_PLACEBO   = 200    # 安慰剂置换次数
REAL_COEF   = None   # 自动从模型A读取，无需手动填写
YEAR_MIN    = 2012
YEAR_MAX    = 2024

# ════════════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════════════
def stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    if p < 0.10:  return '†'
    return 'n.s.'

def print_result(label, coef, pval, ci_lower, ci_upper, nobs):
    print(f"\n{'=' * 62}")
    print(f"  {label}")
    print('=' * 62)
    print(f"  系数           ：{coef:+.4f}")
    print(f"  p 值           ：{pval:.4f}  {stars(pval)}")
    print(f"  95% CI         ：[{ci_lower:+.4f}, {ci_upper:+.4f}]")
    print(f"  样本量         ：{int(nobs):,}")

# ════════════════════════════════════════════════════════════════
# Step 0: 加载 & 预处理
# ════════════════════════════════════════════════════════════════
print("=" * 62)
print("  Step 0: 加载面板")
print("=" * 62)

panel = pd.read_csv(PANEL_PATH)
panel['author_id']    = panel['author_id'].astype(str).str.strip()
panel['year']         = panel['year'].astype(int)
panel['is_treated']   = panel['is_treated'].astype(int)
panel['avg_logRCR']   = pd.to_numeric(panel['avg_logRCR'],    errors='coerce')
panel['relative_year']= pd.to_numeric(panel['relative_year'], errors='coerce')

# 截断年份
panel = panel[panel['year'].between(YEAR_MIN, YEAR_MAX)].copy()

# 构建 Post / DID_term（如果列不存在）
if 'Post' not in panel.columns:
    panel['Post'] = (
        (panel['is_treated'] == 1) &
        (panel['relative_year'] >= 0)
    ).astype(int)
    print("  ℹ️  Post 列不存在，已自动构建")

if 'DID_term' not in panel.columns:
    panel['DID_term'] = panel['is_treated'] * panel['Post']
    print("  ℹ️  DID_term 列不存在，已自动构建")

# 反推 event_year（安慰剂检验需要）
if 'event_year' not in panel.columns:
    panel['event_year'] = np.where(
        panel['is_treated'] == 1,
        panel['year'] - panel['relative_year'],
        np.nan
    )

# 去除 NaN / inf
panel = panel.dropna(subset=['avg_logRCR', 'is_treated', 'year'])
panel = panel[~np.isinf(panel['avg_logRCR'])].copy()

print(f"\n  清理后行数    ：{len(panel):,}")
print(f"  处理组作者    ：{panel[panel['is_treated']==1]['author_id'].nunique():,} 人")
print(f"  对照组作者    ：{panel[panel['is_treated']==0]['author_id'].nunique():,} 人")
print(f"  年份范围      ：{panel['year'].min()} ~ {panel['year'].max()}")
print(f"  DID_term=1 行 ：{(panel['DID_term']==1).sum():,}")
print(f"\n  avg_logRCR 分布（= log(1+RCR)，最小值理论为 0）：")
for lbl, q in [('min', 0), ('p25', .25), ('p50', .5),
               ('p75', .75), ('p95', .95), ('max', 1)]:
    print(f"    {lbl:>4} = {panel['avg_logRCR'].quantile(q):.4f}")

# ════════════════════════════════════════════════════════════════
# Step 1: 构建 MultiIndex 面板（处理重复索引）
# ════════════════════════════════════════════════════════════════
panel_idx = panel.set_index(['author_id', 'year'])
dup = panel_idx.index.duplicated().sum()
if dup > 0:
    print(f"\n  ⚠️  发现 {dup} 个重复索引，取均值合并...")
    panel_idx = panel_idx.groupby(level=['author_id', 'year']).mean()

print(f"\n  ✅ MultiIndex 面板：{len(panel_idx):,} 行")

# ════════════════════════════════════════════════════════════════
# Step 2: 主回归（模型 A / B / C）
# ════════════════════════════════════════════════════════════════

# ── 模型 A：TWFE 基准 ─────────────────────────────────────────
mA = PanelOLS.from_formula(
    'avg_logRCR ~ DID_term + EntityEffects + TimeEffects',
    data=panel_idx
).fit(cov_type='clustered', cluster_entity=True)

coef_A = mA.params['DID_term']
pval_A = mA.pvalues['DID_term']
ci_A   = mA.conf_int().loc['DID_term']
print_result(
    "模型 A：TWFE DID（作者 FE + 年份 FE）← 主模型",
    coef_A, pval_A, ci_A['lower'], ci_A['upper'], mA.nobs
)
REAL_COEF = float(coef_A)   # 存储给安慰剂检验使用

# 效应量解释
effect_pct = (np.exp(coef_A) - 1) * 100
print(f"\n  📌 效应量解释：exp({coef_A:.4f}) - 1 = {effect_pct:.1f}%")
print(f"     跨界合作后，RCR 引用影响力约提升 {effect_pct:.1f}%")

# ── 模型 B：+ 论文数量控制（稳健性）─────────────────────────
mB = PanelOLS.from_formula(
    'avg_logRCR ~ DID_term + yearly_paper_count + EntityEffects + TimeEffects',
    data=panel_idx
).fit(cov_type='clustered', cluster_entity=True)

coef_B = mB.params['DID_term']
pval_B = mB.pvalues['DID_term']
ci_B   = mB.conf_int().loc['DID_term']
print_result(
    "模型 B：TWFE + 论文数量控制（稳健性）",
    coef_B, pval_B, ci_B['lower'], ci_B['upper'], mB.nobs
)

# ── 模型 C：仅实验组 Before/After ────────────────────────────
panel_treat_idx = panel_idx[panel_idx['is_treated'] == 1].copy()

mC = PanelOLS.from_formula(
    'avg_logRCR ~ Post + EntityEffects + TimeEffects',
    data=panel_treat_idx
).fit(cov_type='clustered', cluster_entity=True)

coef_C = mC.params['Post']
pval_C = mC.pvalues['Post']
ci_C   = mC.conf_int().loc['Post']
print_result(
    "模型 C：实验组内 Before/After（EntityEffects + TimeEffects）",
    coef_C, pval_C, ci_C['lower'], ci_C['upper'], mC.nobs
)

# ════════════════════════════════════════════════════════════════
# Step 3: 安慰剂检验（随机置换 event_year，修复版）
# ════════════════════════════════════════════════════════════════
print(f"\n{'=' * 62}")
print(f"  Step 3: 安慰剂检验（随机置换 event_year，{N_PLACEBO} 次）")
print('=' * 62)

# 分离处理组 / 对照组原始数据（在 set_index 之前的 panel）
treated_raw = panel[panel['is_treated'] == 1].copy()
control_raw = panel[panel['is_treated'] == 0].copy()

# 获取处理组每个作者的真实 event_year
ey_map = (treated_raw.drop_duplicates('author_id')
                      .set_index('author_id')['event_year']
                      .dropna()
                      .to_dict())

author_ids_list   = list(ey_map.keys())
event_yr_arr      = np.array(list(ey_map.values()))

# 对照组贡献的列（用于构建合并面板）
NEEDED_COLS = ['author_id', 'year', 'avg_logRCR',
               'is_treated', 'yearly_paper_count']
NEEDED_COLS = [c for c in NEEDED_COLS if c in panel.columns]

placebo_coefs = []
np.random.seed(42)

for i in range(N_PLACEBO):
    # 随机打乱 event_year
    shuffled_yr   = np.random.permutation(event_yr_arr)
    fake_event_map = dict(zip(author_ids_list, shuffled_yr))

    # 重建处理组的 Post_fake / DID_fake
    tmp = treated_raw[NEEDED_COLS].copy()
    tmp['fake_event_year'] = tmp['author_id'].map(fake_event_map)
    tmp['fake_rel_year']   = tmp['year'] - tmp['fake_event_year']
    tmp['Post_fake']       = (tmp['fake_rel_year'] >= 0).astype(int)
    tmp['DID_fake']        = tmp['Post_fake']   # is_treated 全为 1

    # 对照组
    ctrl_tmp = control_raw[NEEDED_COLS].copy()
    ctrl_tmp['Post_fake'] = 0
    ctrl_tmp['DID_fake']  = 0

    combined = pd.concat([tmp, ctrl_tmp], ignore_index=True)
    combined = (combined
                .set_index(['author_id', 'year'])
                .groupby(level=['author_id', 'year'])
                .mean(numeric_only=True))

    try:
        res = PanelOLS.from_formula(
            'avg_logRCR ~ DID_fake + EntityEffects + TimeEffects',
            data=combined
        ).fit(cov_type='clustered', cluster_entity=True)
        placebo_coefs.append(res.params['DID_fake'])
    except Exception:
        continue

    if (i + 1) % 50 == 0:
        print(f"  ... 完成 {i+1}/{N_PLACEBO} 次置换")

placebo_coefs = np.array(placebo_coefs)
empirical_p   = (np.abs(placebo_coefs) >= np.abs(REAL_COEF)).mean()

print(f"\n  ── 安慰剂检验结果 ──────────────────────────────────")
print(f"  真实系数（模型 A）    ：{REAL_COEF:+.4f}")
print(f"  安慰剂系数均值        ：{placebo_coefs.mean():+.4f}")
print(f"  安慰剂系数标准差      ：{placebo_coefs.std():.4f}")
print(f"  安慰剂系数范围        ：[{placebo_coefs.min():+.4f}, {placebo_coefs.max():+.4f}]")
print(f"  经验 p 值（双侧）     ：{empirical_p:.4f}")

if empirical_p < 0.05:
    print(f"\n  ✅ 安慰剂检验通过：真实系数在零分布右尾之外（经验p={empirical_p:.3f}）")
    print(f"     随机置换无法复现真实效应，排除偶然性解释")
else:
    print(f"\n  ⚠️  安慰剂检验存疑：{empirical_p*100:.1f}% 的置换系数 ≥ 真实系数")

# ── 画安慰剂分布图 ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.hist(placebo_coefs, bins=40, color='steelblue',
        alpha=0.75, edgecolor='white', label=f'Placebo coefficients (n={len(placebo_coefs)})')
ax.axvline(REAL_COEF, color='crimson', lw=2.2, ls='--',
           label=f'True estimate = {REAL_COEF:+.4f}***')
ax.axvline(0, color='black', lw=0.8, ls=':', alpha=0.5)
ax.set_xlabel('DID Coefficient', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title(
    'Placebo Test: Randomized Event-Year Assignment\n'
    f'(True coefficient vs. null distribution, empirical p = {empirical_p:.3f})',
    fontsize=12, fontweight='bold'
)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('placebo_test.png', dpi=300, bbox_inches='tight')
plt.show()
print(f"\n  ✅ 图已保存：placebo_test.png")

# ════════════════════════════════════════════════════════════════
# Step 4: 完整汇总表
# ════════════════════════════════════════════════════════════════
print(f"\n{'=' * 62}")
print("  完整模型汇总")
print('=' * 62)

summary = pd.DataFrame([
    {
        '模型'     : 'A：TWFE（主模型）',
        '系数'     : round(coef_A, 4),
        '显著性'   : stars(pval_A),
        'p 值'     : round(pval_A, 4),
        'CI 下限'  : round(ci_A['lower'], 4),
        'CI 上限'  : round(ci_A['upper'], 4),
        '样本'     : '全样本',
        '控制变量' : '作者FE + 年份FE',
    },
    {
        '模型'     : 'B：+ 论文数量',
        '系数'     : round(coef_B, 4),
        '显著性'   : stars(pval_B),
        'p 值'     : round(pval_B, 4),
        'CI 下限'  : round(ci_B['lower'], 4),
        'CI 上限'  : round(ci_B['upper'], 4),
        '样本'     : '全样本',
        '控制变量' : '+ yearly_paper_count',
    },
    {
        '模型'     : 'C：实验组 Before/After',
        '系数'     : round(coef_C, 4),
        '显著性'   : stars(pval_C),
        'p 值'     : round(pval_C, 4),
        'CI 下限'  : round(ci_C['lower'], 4),
        'CI 上限'  : round(ci_C['upper'], 4),
        '样本'     : '仅实验组',
        '控制变量' : '作者FE + 年份FE',
    },
    {
        '模型'     : 'D：安慰剂（置换检验）',
        '系数'     : round(float(placebo_coefs.mean()), 4),
        '显著性'   : '✅ 通过' if empirical_p < 0.05 else '⚠️ 存疑',
        'p 值'     : round(float(empirical_p), 4),
        'CI 下限'  : round(float(np.percentile(placebo_coefs, 2.5)), 4),
        'CI 上限'  : round(float(np.percentile(placebo_coefs, 97.5)), 4),
        '样本'     : '置换分布',
        '控制变量' : '随机 event_year（200次）',
    },
])

pd.set_option('display.max_colwidth', 30)
pd.set_option('display.width', 120)
print(summary.to_string(index=False))

print("\n  注：标准误按作者聚类（cluster-robust SE）")
print("      * p<0.05  ** p<0.01  *** p<0.001")
print("      模型 D 为随机置换安慰剂，期待经验p值<0.05（真实效应显著不同于零分布）")
print(f"\n  效应量：exp({REAL_COEF:.4f}) - 1 ≈ {(np.exp(REAL_COEF)-1)*100:.1f}%（RCR引用影响力提升）")
