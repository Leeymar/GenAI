import io
import re
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════════════
# 配置
# ════════════════════════════════════════════════════════════════
PANEL_PATH = 'Ultimate_DID_Panel.csv'
YEAR_MIN   = 2014
YEAR_MAX   = 2024

# ════════════════════════════════════════════════════════════════
# Step 0: 加载 & 构建 CS-DID 所需变量
# ════════════════════════════════════════════════════════════════
print("=" * 65)
print("  Callaway-Sant'Anna DID 补充检验")
print("=" * 65)

raw = pd.read_csv(PANEL_PATH)
raw['author_id']     = raw['author_id'].astype(str).str.strip()
raw['year']          = raw['year'].astype(int)
raw['is_treated']    = pd.to_numeric(raw['is_treated'],    errors='coerce').fillna(0).astype(int)
raw['avg_logRCR']    = pd.to_numeric(raw['avg_logRCR'],    errors='coerce')
raw['relative_year'] = pd.to_numeric(raw['relative_year'], errors='coerce').round(0)

raw = raw[raw['year'].between(YEAR_MIN, YEAR_MAX)].copy()
raw = raw.dropna(subset=['avg_logRCR', 'is_treated', 'year']).copy()
raw = raw[~np.isinf(raw['avg_logRCR'])].copy()

# ── 构建队列变量 g ────────────────────────────────────────────
treat_cohort = raw[raw['is_treated'] == 1][['author_id', 'year', 'relative_year']].copy()
treat_cohort = treat_cohort[treat_cohort['relative_year'] == 0][['author_id', 'year']]
treat_cohort = treat_cohort.drop_duplicates('author_id').rename(columns={'year': 'g'})

ctrl_cohort = pd.DataFrame({
    'author_id': raw[raw['is_treated'] == 0]['author_id'].unique(),
    'g': 0
})

cohort_map = pd.concat([treat_cohort, ctrl_cohort], ignore_index=True)

raw_cs = raw[['author_id', 'year', 'avg_logRCR', 'is_treated']].copy()
raw_cs = raw_cs.merge(cohort_map, on='author_id', how='inner')
raw_cs = (raw_cs.sort_values(['author_id', 'year'])
                .groupby(['author_id', 'year'], as_index=False)
                .first())
raw_cs['id_num'] = raw_cs['author_id'].astype('category').cat.codes + 1

# ── 地域列 ────────────────────────────────────────────────────
raw['country_code'] = raw['country_code'].astype(str).str.strip().str.upper()
raw['is_CN_US']     = raw['country_code'].isin(['CN', 'US']).astype(int)
raw['is_non_CN_US'] = (1 - raw['is_CN_US'])

print(f"\n  样本：{len(raw_cs):,} 行")
print(f"  队列（g）分布：")
print(raw_cs[raw_cs['g'] > 0]['g'].value_counts().sort_index().to_string())
print(f"  未处理（g=0）：{(raw_cs['g'] == 0).sum():,} 行")


# ════════════════════════════════════════════════════════════════
# Step 1: 核心估计函数
# ════════════════════════════════════════════════════════════════
def run_attgt(df):
    from csdid.att_gt import ATTgt
    obj = ATTgt(
        yname                  = 'avg_logRCR',
        tname                  = 'year',
        idname                 = 'id_num',
        gname                  = 'g',
        data                   = df,
        control_group          = 'nevertreated',
        anticipation           = 0,
        panel                  = True,
        allow_unbalanced_panel = True,
        biters                 = 999,
        alp                    = 0.05,
        cband                  = False
    )
    if hasattr(obj, 'fit'):
        obj.fit()
    return obj


# ════════════════════════════════════════════════════════════════
# Step 2: 解析 Dynamic Effects 打印输出
# ════════════════════════════════════════════════════════════════
def _parse_dynamic_effects(text):
    """
    从 aggte 的打印输出中解析 Dynamic Effects 表格，返回 DataFrame。

    期望格式（示例）：
       Event time  Estimate  Std. Error  [95.0% Simult.   Conf. Band
    0          -2   -0.0338      0.0214          -0.0757      0.0080
    1          -1    0.0045      0.0169          -0.0285      0.0376
    """
    rows = []

    match_section = re.search(r'Dynamic Effects\s*:(.*?)(?:\n---|\Z)', text, re.DOTALL)
    if not match_section:
        return None

    section = match_section.group(1)

    pattern = re.compile(
        r'^\s*\d+\s+'           # 行序号（0, 1, 2...）
        r'(-?\d+)\s+'           # Event time（整数，可负）
        r'(-?[\d.]+)\s+'        # Estimate
        r'(-?[\d.]+)\s+'        # Std. Error
        r'(-?[\d.]+)\s+'        # CI lower
        r'(-?[\d.]+)',           # CI upper
        re.MULTILINE
    )

    for m in pattern.finditer(section):
        rows.append({
            'e'    : int(m.group(1)),
            'att'  : float(m.group(2)),
            'se'   : float(m.group(3)),
            'lower': float(m.group(4)),
            'upper': float(m.group(5)),
        })

    return pd.DataFrame(rows) if rows else None


# ════════════════════════════════════════════════════════════════
# Step 3: 聚合函数
# ════════════════════════════════════════════════════════════════
def get_event_study_df(cs_obj):
    """
    从 ATTgt 对象聚合事件研究系数，返回 DataFrame。
    方案A：捕获 aggte('dynamic') 的 stdout 并正则解析（含 SE / CI）。
    方案B：MP 手动聚合（无 SE 兜底）。
    """

    # ── 方案 A：捕获 stdout 并解析 ───────────────────────────
    old_stdout = sys.stdout
    try:
        buf = io.StringIO()
        sys.stdout = buf
        agg = cs_obj.aggte('dynamic', na_rm=True)
        sys.stdout = old_stdout
        output = buf.getvalue()
        print(output, end='')   # 保留控制台可见性

        df = _parse_dynamic_effects(output)
        if df is not None and len(df) > 0:
            print(f"    ✅ aggte('dynamic') 解析成功，{len(df)} 个时间点")
            return df
        else:
            print("    ⚠️  aggte('dynamic') 输出解析失败，尝试 MP 备用方案")

    except Exception as e:
        sys.stdout = old_stdout
        print(f"    aggte('dynamic') 失败：{e}")

    # ── 方案 B：MP 手动聚合 ───────────────────────────────────
    try:
        mp = cs_obj.MP
        d  = mp if isinstance(mp, dict) else (
             mp.__dict__ if hasattr(mp, '__dict__') else
             {a: getattr(mp, a) for a in dir(mp) if not a.startswith('__')})

        print(f"    [DEBUG] MP keys: {list(d.keys())}")

        def find_arr(candidates):
            for c in candidates:
                v = d.get(c) if isinstance(d, dict) else getattr(mp, c, None)
                if v is not None and isinstance(v, (list, np.ndarray)) and len(v) > 0:
                    return np.array(v)
            return None

        g_arr   = find_arr(['group', 'groups', 'g', 'G', 'cohort'])
        t_arr   = find_arr(['t', 'tlist', 'time', 'period'])
        att_arr = find_arr(['att', 'ATT', 'att_gt', 'coef', 'estimate'])
        se_arr  = find_arr(['se', 'SE', 'std_error', 'stderr'])

        if g_arr is None or t_arr is None or att_arr is None:
            print("    ❌ MP 关键列缺失")
            return None

        if se_arr is None:
            print("    ⚠️  MP 中未找到 SE，置信区间将不可用")

        df_gt = pd.DataFrame({'g': g_arr, 't': t_arr, 'att': att_arr})
        if se_arr is not None:
            df_gt['se'] = se_arr

        df_gt['e'] = df_gt['t'] - df_gt['g']
        df_gt = df_gt[np.isfinite(df_gt['att'])].copy()

        agg_funcs = {'att': lambda x: np.average(x['att'],
                                                   weights=np.abs(x['att']).clip(1e-9))}
        if 'se' in df_gt.columns:
            agg_funcs['se'] = lambda x: np.sqrt(np.mean(x['se'] ** 2))

        df_agg = (df_gt.groupby('e')
                       .apply(lambda x: pd.Series({k: fn(x) for k, fn in agg_funcs.items()}))
                       .reset_index())
        print(f"    ✅ MP 手动聚合成功，{len(df_agg)} 个时间点")
        print(f"    ⚠️  此路径无 SE，p 值不可用，请以控制台 Dynamic Effects 为准")
        return df_agg

    except Exception as e:
        print(f"    ❌ MP 方案失败：{e}")
        return None


# ════════════════════════════════════════════════════════════════
# Step 4: 列名标准化
# ════════════════════════════════════════════════════════════════
def standardize_cols(df):
    col_map = {}
    for col in df.columns:
        cl = col.lower().replace(' ', '').replace('_', '').replace('.', '')
        if cl in ['e', 'event', 'relativeyear', 'relyear', 'relyr',
                  'eventtime', 'eventyear', 'dynamic', 'egt']:
            col_map['e'] = col
        elif cl in ['att', 'estimate', 'coef', 'coefficient', 'betas',
                    'attegt', 'attegt']:
            col_map['att'] = col
        elif cl in ['se', 'stderr', 'stderror', 'std', 'sigma', 'seegt']:
            col_map['se'] = col
        elif cl in ['lower', 'cilower', 'lowci', 'lb', 'cil', 'ci_l',
                    'lowerci', 'lower95', 'll']:
            col_map['lower'] = col
        elif cl in ['upper', 'ciupper', 'upci', 'ub', 'ciu', 'ci_u',
                    'upperci', 'upper95', 'ul']:
            col_map['upper'] = col
    return df.rename(columns=col_map)


# ════════════════════════════════════════════════════════════════
# Step 5: 画图函数
# ════════════════════════════════════════════════════════════════
def plot_cs_event_study(cs_obj, title='', label='cs', window=3):
    from scipy import stats as scipy_stats

    if cs_obj is None:
        print("  ⚠️  无 CS 结果，跳过")
        return None

    df_raw = get_event_study_df(cs_obj)
    if df_raw is None:
        print("  ⚠️  聚合失败，跳过画图")
        return None

    print(f"\n  原始列名：{list(df_raw.columns)}")
    df_plot = standardize_cols(df_raw)

    if 'e' not in df_plot.columns or 'att' not in df_plot.columns:
        print(f"  ⚠️  列名识别失败，原始列：{list(df_raw.columns)}")
        return df_raw

    df_plot['e'] = pd.to_numeric(df_plot['e'], errors='coerce').astype(int)
    df_plot = df_plot[df_plot['e'].between(-window, window)].copy()

    # ── 构造 CI（优先同步置信带，其次 1.96×SE）──────────────
    if 'lower' not in df_plot.columns or 'upper' not in df_plot.columns:
        if 'se' in df_plot.columns:
            df_plot['lower'] = df_plot['att'] - 1.96 * df_plot['se']
            df_plot['upper'] = df_plot['att'] + 1.96 * df_plot['se']
        else:
            df_plot['lower'] = df_plot['att']
            df_plot['upper'] = df_plot['att']

    df_plot = df_plot.sort_values('e').reset_index(drop=True)

    # ── p 值 ────────────────────────────────────────────────
    if 'se' in df_plot.columns:
        df_plot['p'] = df_plot.apply(
            lambda r: 2 * (1 - scipy_stats.norm.cdf(abs(r['att'] / r['se'])))
            if pd.notna(r['se']) and r['se'] > 0 else np.nan, axis=1)
    else:
        df_plot['p'] = np.nan

    def star(p):
        if pd.isna(p): return 'n.s.'
        return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '†' if p < 0.1 else 'n.s.'

    df_plot['star'] = df_plot['p'].apply(star)

    # ── 平行趋势诊断 ─────────────────────────────────────────
    print(f"\n  ── CS-DID 基准期诊断（T < 0，期待不显著）─")
    pre_sig = []
    for _, row in df_plot[df_plot['e'] < 0].iterrows():
        ok     = '✅' if pd.isna(row['p']) or row['p'] > 0.1 else '⚠️ '
        se_str = f"{row['se']:.4f}" if 'se' in df_plot.columns and pd.notna(row.get('se')) else 'n/a'
        p_str  = f"{row['p']:.3f}"  if pd.notna(row['p']) else 'n/a'
        print(f"    T={int(row['e']):>+3}  ATT={row['att']:>+.4f}  "
              f"SE={se_str}  p={p_str}  {row['star']}  {ok}")
        if pd.notna(row['p']) and row['p'] < 0.1:
            pre_sig.append(int(row['e']))

    verdict = ("✅ CS-DID 平行趋势成立：TWFE 结论稳健"
               if not pre_sig
               else f"⚠️  预先趋势存在：T={pre_sig}，需审慎解读")
    print(f"\n  {verdict}")

    print(f"\n  ── 处理效应（T ≥ 0）──────────────────────────────")
    for _, row in df_plot[df_plot['e'] >= 0].iterrows():
        p_str = f"{row['p']:.3f}" if pd.notna(row['p']) else 'n/a'
        print(f"    T={int(row['e']):>+3}  ATT={row['att']:>+.4f}  "
              f"p={p_str}  {row['star']}")

    # ── 画图 ──────────────────────────────────────────────────
    ys     = df_plot['e'].tolist()
    coeffs = df_plot['att'].tolist()
    lo     = df_plot['lower'].tolist()
    hi     = df_plot['upper'].tolist()

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.axvspan(min(ys) - 0.5, -0.5,  alpha=0.06, color='gray',  zorder=0)
    ax.axvspan(-0.5, max(ys) + 0.5,  alpha=0.08, color='green', zorder=0)
    ax.axhline(0,     color='black',  lw=0.8,  ls='--', alpha=0.5)
    ax.axvline(-0.5,  color='crimson', lw=1.8, ls='--',
               label='Treatment onset (T=0)', alpha=0.9, zorder=3)

    ax.fill_between(ys, lo, hi, alpha=0.20, color='steelblue', zorder=2)
    ax.plot(ys, coeffs, 'o-', color='steelblue',
            lw=2.2, ms=7, zorder=5, label='CS-DID ATT(e) [95% CI]')

    for _, row in df_plot.iterrows():
        if pd.notna(row['p']) and row['p'] < 0.05:
            ax.annotate(row['star'],
                        xy=(int(row['e']), row['att']),
                        xytext=(0, 9), textcoords='offset points',
                        ha='center', fontsize=13, color='crimson', fontweight='bold')

    title_warn  = f'\n⚠️ Pre-trend at T={pre_sig}' if pre_sig else ''
    title_color = 'darkred' if pre_sig else 'black'
    ax.set_title(
        f"Callaway-Sant'Anna Event Study: Citation Impact of Cross-Boundary Collaboration"
        + (f'\n{title}' if title else '') + title_warn,
        fontsize=11, fontweight='bold', color=title_color)

    ax.set_xlabel('Years relative to first cross-boundary collaboration (T=0)', fontsize=11)
    ax.set_ylabel('ATT: Δ log(1+RCR)', fontsize=11)
    ax.set_xticks(ys)
    ax.set_xticklabels([f'T={y:+d}' for y in ys], fontsize=10)

    pre_p  = mpatches.Patch(color='gray',  alpha=0.2, label='Pre-treatment window')
    post_p = mpatches.Patch(color='green', alpha=0.2, label='Post-treatment window')
    ax.legend(handles=[pre_p, post_p], fontsize=9, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    fname = f'cs_event_study_{label}.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\n  ✅ 图已保存：{fname}")

    return {'df': df_plot, 'pre_sig': pre_sig}


# ════════════════════════════════════════════════════════════════
# Step 6: 主模型
# ════════════════════════════════════════════════════════════════
print(f"\n{'=' * 65}")
print("  CS-DID：全样本")
print('=' * 65)

try:
    cs_main = run_attgt(raw_cs)
    print("  ✅ CS-DID 估计完成")
    plot_cs_event_study(cs_main, title='', label='main')
except Exception as e:
    print(f"  ❌ CS-DID 失败：{e}")
    cs_main = None


# ════════════════════════════════════════════════════════════════
# Step 7: 异质性子组函数
# ════════════════════════════════════════════════════════════════
def run_cs_hetero(group_col, group_val, group_label, label_tag,
                  min_cohort_size=50):
    """
    运行 CS-DID 子组异质性分析

    Parameters
    ----------
    group_col       : str   分组列名（如 'is_CN_US'）
    group_val       : int   目标分组值（如 1）
    group_label     : str   图标题说明文字
    label_tag       : str   输出文件名标签
    min_cohort_size : int   队列最小作者数，低于此值的早期小队列被剔除（默认 50）
    """

    # ── 合并分组信息 ──────────────────────────────────────────
    grp_col_df = raw[['author_id', group_col]].drop_duplicates('author_id')
    raw_cs_mg  = raw_cs.merge(grp_col_df, on='author_id', how='left')

    treat_sub = raw_cs_mg[(raw_cs_mg['g'] > 0) & (raw_cs_mg[group_col] == group_val)]
    ctrl_sub  = raw_cs_mg[raw_cs_mg['g'] == 0]

    n = treat_sub['author_id'].nunique()
    print(f"\n{'=' * 65}")
    print(f"  CS-DID：{group_label}  (处理组 N={n:,} 人)")
    print('=' * 65)

    if n < 30:
        print(f"  ⚠️  处理组作者过少（{n} 人），跳过")
        return None

    # ── 拼合并去重 ────────────────────────────────────────────
    sub = pd.concat([treat_sub, ctrl_sub], ignore_index=True)
    sub = (sub.sort_values(['author_id', 'year'])
              .groupby(['author_id', 'year'], as_index=False)
              .first())

    # ── 剔除样本量过小的队列 ─────────────────────────────────
    cohort_sizes = (sub[sub['g'] > 0]
                    .groupby('g')['author_id']
                    .nunique()
                    .rename('cohort_n'))

    small = cohort_sizes[cohort_sizes < min_cohort_size].index.tolist()
    valid = cohort_sizes[cohort_sizes >= min_cohort_size].index.tolist()

    if small:
        print(f"  ℹ️  以下队列作者数 < {min_cohort_size}，已剔除：")
        for g in sorted(small):
            print(f"      g={g}  N={cohort_sizes[g]}")
        sub = sub[(sub['g'].isin(valid)) | (sub['g'] == 0)].copy()

    remaining_treat = sub[sub['g'] > 0]['author_id'].nunique()
    if remaining_treat < 30:
        print(f"  ⚠️  剔除小队列后处理组仅剩 {remaining_treat} 人，跳过")
        return None

    print(f"  ℹ️  有效队列：{sorted(valid)}  (处理组剩余 N={remaining_treat:,} 人)")

    # ── 重新生成数值型 id ─────────────────────────────────────
    sub['id_num'] = sub['author_id'].astype('category').cat.codes + 1

    # ── 估计 & 画图 ───────────────────────────────────────────
    try:
        cs_sub = run_attgt(sub)
        print(f"  ✅ CS-DID 估计完成")
        return plot_cs_event_study(cs_sub, title=group_label, label=label_tag)
    except Exception as e:
        print(f"  ❌ CS-DID 失败：{e}")
        return None


# ════════════════════════════════════════════════════════════════
# Step 8: 地域异质性
# ════════════════════════════════════════════════════════════════
print(f"\n{'=' * 65}")
print("  CS-DID 地域异质性")
print('=' * 65)

run_cs_hetero('is_CN_US',     1, 'China & United States (CN+US)', 'CN_US')
run_cs_hetero('is_non_CN_US', 1, 'Non-CN/US Regions',             'non_CN_US')


# ════════════════════════════════════════════════════════════════
# Step 9: 机构异质性
# ════════════════════════════════════════════════════════════════
ELITE_COL = 'is_elite_overall' if 'is_elite_overall' in raw.columns else None

if ELITE_COL:
    print(f"\n{'=' * 65}")
    print("  CS-DID 机构异质性")
    print('=' * 65)
    run_cs_hetero(ELITE_COL, 1, 'Elite Institutions',     'elite')
    run_cs_hetero(ELITE_COL, 0, 'Non-Elite Institutions', 'non_elite')
else:
    print("\n  ⚠️  未检测到 is_elite_overall 列，跳过机构异质性")


# ════════════════════════════════════════════════════════════════
# 完成汇总
# ════════════════════════════════════════════════════════════════
print(f"\n{'=' * 65}")
print("  ✅ CS-DID 全部完成！")
print()
print("  输出文件：")
print("    cs_event_study_main.png      ← CS-DID 全样本")
print("    cs_event_study_CN_US.png     ← 中美地区")
print("    cs_event_study_non_CN_US.png ← 非中美地区")
if ELITE_COL:
    print("    cs_event_study_elite.png     ← 精英机构")
    print("    cs_event_study_non_elite.png ← 非精英机构")
print()
print("  解读要点：")
print("    T<0 系数不显著 → 平行趋势成立，TWFE 可信")
print("    T≥0 系数方向与 TWFE 一致 → 主结论稳健")
print("=" * 65)
