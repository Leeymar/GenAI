import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from linearmodels.panel import PanelOLS
import warnings
warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════════════
# 全局配置
# ════════════════════════════════════════════════════════════════
PANEL_PATH = 'Ultimate_DID_Panel.csv'
WINDOW     = 3
REF_YEAR   = -1
YEAR_MIN   = 2014
YEAR_MAX   = 2024

# ════════════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════════════
def stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    if p < 0.10:  return '†'
    return 'n.s.'


# ════════════════════════════════════════════════════════════════
# Step 0: 加载 & 预处理
# ════════════════════════════════════════════════════════════════
print("=" * 65)
print("  Step 0: 加载面板")
print("=" * 65)

raw = pd.read_csv(PANEL_PATH)
raw['author_id']     = raw['author_id'].astype(str).str.strip()
raw['year']          = raw['year'].astype(int)
raw['is_treated']    = pd.to_numeric(raw['is_treated'],    errors='coerce').fillna(0).astype(int)
raw['avg_logRCR']    = pd.to_numeric(raw['avg_logRCR'],    errors='coerce')
raw['relative_year'] = pd.to_numeric(raw['relative_year'], errors='coerce')

# 截断年份
raw = raw[raw['year'].between(YEAR_MIN, YEAR_MAX)].copy()
raw = raw.dropna(subset=['avg_logRCR', 'is_treated', 'year'])
raw = raw[~np.isinf(raw['avg_logRCR'])].copy()

# ── 构建 Post / DID_term ────────────────────────────────────────
raw['Post']     = ((raw['is_treated'] == 1) & (raw['relative_year'] >= 0)).astype(int)
raw['DID_term'] = raw['is_treated'] * raw['Post']

# ── 地域变量（剔除 NAN / UNKNOWN）──────────────────────────────
raw['country_code'] = raw['country_code'].astype(str).str.strip().str.upper()
INVALID_CC = {'NAN', 'UNKNOWN', 'NONE', '', 'NAT', 'NA'}

# 新版（NAN/UNKNOWN 归入非中美）
raw['is_CN_US']     = raw['country_code'].isin(['CN', 'US']).astype(int)
raw['is_non_CN_US'] = (1 - raw['is_CN_US'])   # 所有非中美（含NAN/UNKNOWN）

total   = len(raw)
cn_us_n = raw['is_CN_US'].sum()
other_n = raw['is_non_CN_US'].sum()

print(f"\n  ── 地域分布 ──────────────────────────────────────────")
print(f"  中美（CN+US）  ：{cn_us_n:>8,} 行  ({cn_us_n/total*100:.1f}%)")
print(f"  非中美（含未知）：{other_n:>8,} 行  ({other_n/total*100:.1f}%)")
print(f"     其中 NAN/UNKNOWN：{raw['country_code'].isin(INVALID_CC).sum():,} 行（归入非中美）")

top_other = (raw[raw['is_non_CN_US'] == 1]['country_code']
             .value_counts().head(10))
print(f"\n  非中美 Top10：\n{top_other.to_string()}")


total     = len(raw)
cn_us_n   = raw['is_CN_US'].sum()
other_n   = raw['is_non_CN_US'].sum()
# 新版（直接统计 INVALID_CC 行数，不依赖 cc_valid 列）
invalid_n = raw['country_code'].isin(INVALID_CC).sum()
print(f"\n  ── 地域分布 ──────────────────────────────────────────")
print(f"  中美（CN+US）      ：{cn_us_n:>8,} 行  ({cn_us_n/total*100:.1f}%)")
print(f"  非中美（含未知）    ：{other_n:>8,} 行  ({other_n/total*100:.1f}%)")
print(f"     其中 NAN/UNKNOWN：{invalid_n:>8,} 行（已归入非中美）")


print(f"\n  ── 地域分布（清洗后）─────────────────────────────────")
print(f"  中美（CN+US）  ：{cn_us_n:>8,} 行  ({cn_us_n/total*100:.1f}%)")
print(f"  其他有效地区   ：{other_n:>8,} 行  ({other_n/total*100:.1f}%)")
print(f"  ⚠️  剔除无效CC ：{invalid_n:>8,} 行  (NAN/UNKNOWN 不参与地域分组)")

top_other = (raw[raw['is_non_CN_US'] == 1]['country_code']
             .value_counts().head(8))
print(f"\n  非中美 Top8（已剔除 NAN/UNKNOWN）：\n{top_other.to_string()}")

# ── 机构变量 ────────────────────────────────────────────────────
if 'is_elite_overall' in raw.columns:
    raw['is_elite_overall'] = pd.to_numeric(
        raw['is_elite_overall'], errors='coerce').fillna(0).astype(int)
    ELITE_COL = 'is_elite_overall'
elif 'is_academic_elite' in raw.columns:
    raw['is_academic_elite'] = pd.to_numeric(
        raw['is_academic_elite'], errors='coerce').fillna(0).astype(int)
    ELITE_COL = 'is_academic_elite'
else:
    ELITE_COL = None

if ELITE_COL:
    print(f"\n  ✅ 机构变量：{ELITE_COL}  "
          f"(精英比例={raw[ELITE_COL].mean():.3f})")

# ── ！关键修复：去重时分开处理数值列和分类列 ────────────────────
# 分类列（不能 mean）
CAT_COLS = ['author_id', 'year', 'is_treated', 'relative_year',
            'is_CN_US', 'is_non_CN_US', 'cc_valid'] + \
           ([ELITE_COL] if ELITE_COL else [])
CAT_COLS = [c for c in CAT_COLS if c in raw.columns]

# 数值列（可以 mean）
NUM_COLS_EXTRA = ['yearly_paper_count']
NUM_COLS_EXTRA = [c for c in NUM_COLS_EXTRA if c in raw.columns]

# 先用 first() 保留分类列
cat_part = (raw[CAT_COLS + ['avg_logRCR'] + NUM_COLS_EXTRA]
            .sort_values(['author_id', 'year'])
            .groupby(['author_id', 'year'], as_index=False)
            .first())         # first() 保留整数不变

# relative_year 强制转为整数（去 NaN 后）
cat_part['relative_year'] = (pd.to_numeric(cat_part['relative_year'], errors='coerce')
                              .round(0)
                              .astype('Int64'))   # 可空整数

panel = cat_part.copy()

# ── 窗口过滤：处理组保留 ±WINDOW，对照组全保留 ─────────────────
treat_mask = ((panel['is_treated'] == 1) &
              (panel['relative_year'].notna()) &
              (panel['relative_year'].between(-WINDOW, WINDOW)))
ctrl_mask  = (panel['is_treated'] == 0)
panel = panel[treat_mask | ctrl_mask].copy()

print(f"\n  ✅ 清理完成：{len(panel):,} 行 | "
      f"处理组 {panel[panel['is_treated']==1]['author_id'].nunique():,} 人 | "
      f"对照组 {panel[panel['is_treated']==0]['author_id'].nunique():,} 人")

# 验证：relative_year 分布（处理组）
treat_ry = (panel[panel['is_treated']==1]['relative_year']
            .value_counts().sort_index())
print(f"\n  处理组 relative_year 分布（应为整数 -3 ~ +3）：")
print(treat_ry.to_string())


# ════════════════════════════════════════════════════════════════
# Step 1: 构建事件研究虚拟变量
# ════════════════════════════════════════════════════════════════
def build_event_dummies(df, window=WINDOW, ref_year=REF_YEAR):
    """
    relative_year 精确整数匹配，对照组自动为 0
    """
    yr_range  = list(range(-window, window + 1))
    yr_no_ref = [y for y in yr_range if y != ref_year]

    df = df.copy()
    col_map = {}
    for y in yr_no_ref:
        col  = f'ry_{"m" if y < 0 else "p"}{abs(y)}'
        # 注意：用 == y 对 Int64 仍然有效
        df[col]    = ((df['is_treated'] == 1) &
                      (df['relative_year'] == y)).astype(int)
        col_map[y] = col

    dummy_cols = list(col_map.values())

    # 诊断每个虚拟变量的激活数
    print(f"\n  ── 虚拟变量激活行数诊断 ─────────────────────────────")
    zero_cols = []
    for y in yr_no_ref:
        col = col_map[y]
        n   = df[col].sum()
        print(f"    {col} (T={y:+d}) : {int(n):>6,} 行")
        if n == 0:
            zero_cols.append(col)

    if zero_cols:
        print(f"\n  ⚠️  移除全零虚拟变量：{zero_cols}")
        dummy_cols = [c for c in dummy_cols if c not in zero_cols]
        yr_no_ref  = [y for y in yr_no_ref if col_map[y] in dummy_cols]

    return df, dummy_cols, sorted(yr_no_ref), col_map


def run_event_study(df, title_suffix='', label='main',
                    window=WINDOW, ref_year=REF_YEAR):
    print(f"\n{'=' * 65}")
    print(f"  事件研究：{title_suffix if title_suffix else '主模型（全样本）'}")
    print('=' * 65)

    df = df.dropna(subset=['avg_logRCR', 'is_treated', 'year']).copy()
    df = df[~np.isinf(df['avg_logRCR'])].copy()
    df['relative_year'] = (pd.to_numeric(df['relative_year'], errors='coerce')
                           .round(0).astype('Int64'))

    df_es, dummy_cols, yr_no_ref, col_map = build_event_dummies(df, window, ref_year)

    if not dummy_cols:
        print("  ❌ 无有效虚拟变量，跳过")
        return None

    formula   = 'avg_logRCR ~ ' + ' + '.join(dummy_cols) + ' + EntityEffects + TimeEffects'
    panel_idx = df_es.set_index(['author_id', 'year'])

    if panel_idx.index.duplicated().any():
        panel_idx = panel_idx.groupby(level=['author_id', 'year']).first()

    try:
        res = PanelOLS.from_formula(
            formula,
            data=panel_idx,
            drop_absorbed=True          # ✅ 正确位置
        ).fit(
            cov_type='clustered',
            cluster_entity=True         # ✅ 无多余参数
        )
        print(f"\n  ✅ 回归成功（N={int(res.nobs):,} obs）")
    except Exception as e:
        print(f"  ❌ 回归失败：{e}")
        return None

    # ── 提取系数 ──────────────────────────────────────────────
    coefs, lo_ci, hi_ci, pvals, ses = {}, {}, {}, {}, {}
    for y in yr_no_ref:
        col = col_map[y]
        if col in res.params.index:
            coefs[y] = res.params[col]
            ci       = res.conf_int().loc[col]
            lo_ci[y] = ci['lower']
            hi_ci[y] = ci['upper']
            pvals[y] = res.pvalues[col]
            ses[y]   = res.std_errors[col]
        else:
            print(f"  ⚠️  T={y:+d} 被吸收，已自动跳过")

    coefs[ref_year] = 0.0
    lo_ci[ref_year] = 0.0
    hi_ci[ref_year] = 0.0

    all_yrs = sorted(coefs.keys())
    c_vals  = [coefs[y]  for y in all_yrs]
    lo_vals = [lo_ci[y]  for y in all_yrs]
    hi_vals = [hi_ci[y]  for y in all_yrs]

    # ── 平行趋势报告 ──────────────────────────────────────────
    print(f"\n  ── 基准期系数（T < 0，参照 T={ref_year}，期待不显著）─")
    pre_sig = []
    for y in sorted([yy for yy in all_yrs if yy < 0 and yy != ref_year]):
        c  = coefs.get(y, np.nan)
        p  = pvals.get(y, np.nan)
        se = ses.get(y, np.nan)
        ok = '✅' if (np.isnan(p) or p > 0.1) else '⚠️ '
        print(f"    T={y:>+3}  系数={c:>+.4f}  SE={se:.4f}  "
              f"p={p:.3f}  {stars(p)}  {ok}")
        if not np.isnan(p) and p < 0.1:
            pre_sig.append(y)

    if pre_sig:
        print(f"\n  ⚠️  平行趋势存疑：T={pre_sig} 显著（p<0.1）→ 建议补 Callaway-Sant'Anna")
    else:
        print(f"\n  ✅ 平行趋势成立：基准期系数均不显著，TWFE 估计可信")

    print(f"\n  ── 处理后系数（T ≥ 0）──────────────────────────────")
    for y in sorted([yy for yy in all_yrs if yy >= 0]):
        c = coefs.get(y, np.nan)
        p = pvals.get(y, np.nan)
        print(f"    T={y:>+3}  系数={c:>+.4f}  p={p:.3f}  {stars(p)}")

    # ── 画图 ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5.5))

    ax.axvspan(-window - 0.5, -0.5,  alpha=0.05, color='gray',  zorder=0)
    ax.axvspan(-0.5, window + 0.5,   alpha=0.07, color='green', zorder=0)
    ax.axhline(0, color='black', lw=0.8, ls='--', alpha=0.5)
    ax.axvline(-0.5, color='crimson', lw=1.8, ls='--',
               label='Treatment onset (T=0)', alpha=0.9, zorder=3)
    ax.axvline(ref_year, color='gray', lw=1.2, ls=':',
               label=f'Reference year (T={ref_year})', alpha=0.8, zorder=3)

    ax.fill_between(all_yrs, lo_vals, hi_vals,
                    alpha=0.20, color='steelblue', zorder=2)
    ax.plot(all_yrs, c_vals, 'o-', color='steelblue',
            lw=2.2, ms=7, zorder=5, label='Event-study coefficient (95% CI)')
    ax.plot(ref_year, 0, 's', color='gray', ms=9, zorder=6)
    ax.annotate('ref', xy=(ref_year, 0), xytext=(0, 10),
                textcoords='offset points', ha='center', fontsize=9, color='gray')

    for y in all_yrs:
        p = pvals.get(y, 1)
        if p < 0.05 and y != ref_year:
            ax.annotate(
                stars(p), xy=(y, coefs.get(y, 0)),
                xytext=(0, 9), textcoords='offset points',
                ha='center', fontsize=13, color='crimson', fontweight='bold')

    title_warn  = (f'\n⚠️  Pre-trend at T={pre_sig}'
                   if pre_sig else '')
    title_color = 'darkred' if pre_sig else 'black'
    ax.set_title(
        f'Event Study: Citation Impact of Cross-Boundary Collaboration'
        + (f'\n{title_suffix}' if title_suffix else '')
        + f'  (N={int(res.nobs):,})'
        + title_warn,
        fontsize=11, fontweight='bold', color=title_color)

    ax.set_xlabel('Years relative to first cross-boundary collaboration (T=0)',
                  fontsize=11)
    ax.set_ylabel('Δ log(1+RCR)  vs. reference year (T=−1)', fontsize=11)
    ax.set_xticks(all_yrs)
    ax.set_xticklabels([f'T={y:+d}' for y in all_yrs], fontsize=10)

    pre_patch  = mpatches.Patch(color='gray',  alpha=0.2, label='Pre-treatment window')
    post_patch = mpatches.Patch(color='green', alpha=0.2, label='Post-treatment window')
    ax.legend(handles=[pre_patch, post_patch] + ax.get_lines()[:3],
              fontsize=9, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    fname = f'event_study_{label}.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\n  ✅ 图已保存：{fname}")

    return {'coefs': coefs, 'pvals': pvals, 'ci_lo': lo_ci, 'ci_hi': hi_ci,
            'pre_sig': pre_sig, 'nobs': res.nobs}



# ════════════════════════════════════════════════════════════════
# Step 3: 异质性分组辅助函数
# ════════════════════════════════════════════════════════════════
def run_hetero(df, group_col, group_val, group_label, label_tag):
    """
    处理组按 group_col == group_val 筛选，对照组保持完整
    """
    treat_sub = df[(df['is_treated'] == 1) & (df[group_col] == group_val)].copy()
    ctrl_sub  = df[df['is_treated'] == 0].copy()

    if treat_sub['author_id'].nunique() < 30:
        print(f"\n  ⚠️  {group_label}：处理组作者过少（<30），跳过")
        return None

    sub = pd.concat([treat_sub, ctrl_sub], ignore_index=True)
    # 去重用 first()，保留整数列
    sub = (sub.sort_values(['author_id', 'year'])
              .groupby(['author_id', 'year'], as_index=False)
              .first())

    n = treat_sub['author_id'].nunique()
    print(f"\n  📌 分组：{group_label}  (处理组 N={n:,} 人)")

    return run_event_study(sub, title_suffix=group_label, label=label_tag)


# ════════════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════════════

# ── 1. 主事件研究（全样本）───────────────────────────────────
main_result = run_event_study(panel, title_suffix='', label='main')

# ── 2. 地域异质性：中美 vs 非中美（剔除无效CC）──────────────
print(f"\n{'=' * 65}")
print("  Step 3a: 地域异质性（中美 vs 其他地区）")
print('=' * 65)

run_hetero(panel, 'is_CN_US',     1, 'China & United States (CN + US)', 'hetero_CN_US')
run_hetero(panel, 'is_non_CN_US', 1, 'Non-CN/US Regions (GB/IN/KR/DE/CA/JP…)', 'hetero_non_CN_US')

# ── 3. 机构异质性：精英 vs 非精英 ────────────────────────────
if ELITE_COL:
    print(f"\n{'=' * 65}")
    print(f"  Step 3b: 机构异质性（{ELITE_COL}）")
    print('=' * 65)

    run_hetero(panel, ELITE_COL, 1, 'Elite Institutions',     'hetero_elite')
    run_hetero(panel, ELITE_COL, 0, 'Non-Elite Institutions', 'hetero_non_elite')

# ── 4. 汇总 ───────────────────────────────────────────────────
print(f"\n{'=' * 65}")
print("  ✅ 全部完成！")
print("  主图     → event_study_main.png")
print("  中美     → event_study_hetero_CN_US.png")
print("  非中美   → event_study_hetero_non_CN_US.png")
if ELITE_COL:
    print("  精英     → event_study_hetero_elite.png")
    print("  非精英   → event_study_hetero_non_elite.png")
if main_result and main_result.get('pre_sig'):
    print(f"\n  ⚠️  平行趋势存疑 T={main_result['pre_sig']}，建议补 Callaway-Sant'Anna")
else:
    print("\n  ✅ 平行趋势成立 → TWFE 结论完整可靠")
print("=" * 65)
