import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ================================================================
# 0. 读取数据 & 合并学科信息
# ================================================================
def load_data():
    df = pd.read_csv('E:\PythonProject\GenAI\data_process\\test\\Ultimate_Regression_Base_with_Diversity.csv')
    df_field = pd.read_csv('work_id_to_field_all_stages.csv')

    df_field['work_id'] = df_field['work_id'].str.extract(r'(W\d+)$')
    df = df.merge(df_field[['work_id', 'primary_field']], on='work_id', how='left')
    df['primary_field'] = df['primary_field'].fillna('Other')

    print(f"✅ 数据加载完成，共 {len(df):,} 篇论文")
    print(df['primary_field'].value_counts())
    print()
    return df


# ================================================================
# 1. 特征工程
# ================================================================
def feature_engineering(df):
    df['HSS_Continuous'] = df['team_hss_dna_pct'] / 10.0

    bins = [-0.1, 0.1, 20.0, 40.0, 60.0, 100.1]
    labels = [
        '1_Pure_STEM(0%)',
        '2_Low_HSS(1-20%)',
        '3_Moderate_HSS(21-40%)',
        '4_High_HSS(41-60%)',
        '5_HSS_Dominant(>60%)'
    ]
    df['HSS_Bin'] = pd.cut(df['team_hss_dna_pct'], bins=bins, labels=labels)

    df['is_hss_label']        = df['arxiv_type'].apply(lambda x: 1 if 'Type 3' in str(x) else 0)
    df['log_team_size']       = np.log1p(df['total_authors'])
    df['log_prior_knowledge'] = np.log1p(df['Prior_Knowledge_Stock'])

    # 年份列自动识别
    year_col = 'year'
    if year_col not in df.columns:
        for alt in ['pub_year', 'publication_year', 'Year']:
            if alt in df.columns:
                df[year_col] = df[alt]
                print(f"✅ 年份列使用: {alt}")
                break
        else:
            raise ValueError("❌ 找不到年份列，请手动指定 year_col。")

    print(f"✅ 特征工程完成，年份范围: {df[year_col].min()} — {df[year_col].max()}\n")
    return df, year_col


# ================================================================
# 2. 原始 5 大模型 + 学科 FE
# ================================================================
def run_main_regressions(df):
    print("=" * 110)
    print("📈 Step 1：原始 5 大 OLS 模型 + 学科 FE 稳健性检验")
    print("=" * 110)

    ref_bin     = "1_Pure_STEM(0%)"
    controls    = "HSS_Diversity + log_team_size + log_prior_knowledge + C(stage)"
    controls_fe = controls + " + C(primary_field)"

    m1    = smf.ols(f"Log_RCR ~ C(HSS_Bin, Treatment('{ref_bin}')) + {controls}",                      data=df).fit()
    m2    = smf.ols(f"Log_RCR ~ is_hss_label + {controls}",                                            data=df).fit()
    m3    = smf.ols(f"Log_RCR ~ C(HSS_Bin, Treatment('{ref_bin}')) + is_hss_label + {controls}",       data=df).fit()
    m4    = smf.ols(f"Log_RCR ~ HSS_Continuous + {controls}",                                          data=df).fit()
    m5    = smf.ols(f"Log_RCR ~ HSS_Continuous + is_hss_label + {controls}",                           data=df).fit()
    m1_fe = smf.ols(f"Log_RCR ~ C(HSS_Bin, Treatment('{ref_bin}')) + {controls_fe}",                   data=df).fit()
    m3_fe = smf.ols(f"Log_RCR ~ C(HSS_Bin, Treatment('{ref_bin}')) + is_hss_label + {controls_fe}",    data=df).fit()
    m5_fe = smf.ols(f"Log_RCR ~ HSS_Continuous + is_hss_label + {controls_fe}",                        data=df).fit()

    regressor_order = [
        'HSS_Continuous',
        f"C(HSS_Bin, Treatment('{ref_bin}'))[T.2_Low_HSS(1-20%)]",
        f"C(HSS_Bin, Treatment('{ref_bin}'))[T.3_Moderate_HSS(21-40%)]",
        f"C(HSS_Bin, Treatment('{ref_bin}'))[T.4_High_HSS(41-60%)]",
        f"C(HSS_Bin, Treatment('{ref_bin}'))[T.5_HSS_Dominant(>60%)]",
        'is_hss_label',
        'HSS_Diversity',
        'log_team_size',
        'log_prior_knowledge',
    ]

    info_dict = {
        'N':        lambda x: f"{int(x.nobs):,}",
        'R²':       lambda x: f"{x.rsquared:.3f}",
        'Field FE': lambda x: 'Yes' if 'C(primary_field)' in x.model.formula else 'No',
    }

    print("\n🏆 Table 1：主回归（无学科 FE）")
    print(summary_col(
        [m1, m2, m3, m4, m5],
        model_names=['M1(Bin)', 'M2(Lab)', 'M3(Full)', 'M4(Cont)', 'M5(Full)'],
        stars=True, float_format='%0.3f',
        regressor_order=regressor_order, drop_omitted=True,
        info_dict=info_dict
    ))
    print("💡 Note: *** p<0.01, ** p<0.05, * p<0.1\n")

    print("🔬 Table 2：学科 FE 稳健性检验")
    print(summary_col(
        [m1, m1_fe, m3, m3_fe, m5, m5_fe],
        model_names=['M1', 'M1+FE', 'M3', 'M3+FE', 'M5', 'M5+FE'],
        stars=True, float_format='%0.3f',
        regressor_order=regressor_order, drop_omitted=True,
        info_dict=info_dict
    ))
    print("💡 Note: *** p<0.01, ** p<0.05, * p<0.1\n")

    return controls_fe


# ================================================================
# 3. DID
# ================================================================
def run_did(df, controls_fe):
    print("=" * 80)
    print("📊 Step 2：DID——文科基因 × GenAI 时代交叉项")
    print("=" * 80)

    m_did = smf.ols(
        f"Log_RCR ~ HSS_Continuous * C(stage) + is_hss_label + {controls_fe}",
        data=df
    ).fit()

    did_vars = [c for c in m_did.params.index if 'HSS_Continuous:' in c]
    print(f"\n  {'变量':<50} {'β':>8}  {'p值':>7}  显著性")
    print("-" * 75)
    for v in did_vars:
        coef = m_did.params[v]
        pval = m_did.pvalues[v]
        star = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
        print(f"  {v:<50} {coef:>+8.4f}  {pval:>7.4f}  {star}")
    print()


# ================================================================
# 4. 异质性分析
# ================================================================
def run_heterogeneity(df):
    print("=" * 80)
    print("🔬 Step 3：异质性分析——各学科中的文科基因溢价")
    print("=" * 80)
    print(f"\n  {'学科':<35} {'β':>8}  {'SE':>7}  {'p值':>7}  显著性  {'N':>8}")
    print("-" * 80)

    top_fields = [f for f in df['primary_field'].value_counts().index if f != 'Other']

    for field in top_fields:
        sub = df[df['primary_field'] == field].copy()
        if len(sub) < 200:
            print(f"  ⏭️  {field:<33} 样本不足（{len(sub)}），跳过")
            continue
        try:
            m_sub = smf.ols(
                "Log_RCR ~ HSS_Continuous + is_hss_label + "
                "HSS_Diversity + log_team_size + log_prior_knowledge + C(stage)",
                data=sub
            ).fit()
            coef = m_sub.params['HSS_Continuous']
            se   = m_sub.bse['HSS_Continuous']
            pval = m_sub.pvalues['HSS_Continuous']
            star = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else '  '))
            print(f"  {field:<35} {coef:>+8.4f}  {se:>7.4f}  {pval:>7.4f}  {star:<5}  {len(sub):>8,}")
        except Exception as e:
            print(f"  ❌ {field} 报错：{e}")
    print()


# ================================================================
# 5. 平行趋势检验
# ================================================================
def run_parallel_trends(df, year_col):
    print("=" * 80)
    print("📐 Step 4：平行趋势检验（Phase1 内部逐年交叉项）")
    print("=" * 80)

    df_phase1 = df[df['stage'] == 'Phase1'].copy()
    base_year = int(df_phase1[year_col].min())

    print(f"\n  Phase1 样本量: {len(df_phase1):,} 篇")
    print(f"  Phase1 年份范围: {df_phase1[year_col].min()} — {df_phase1[year_col].max()}")
    print(f"  基准年（参照组）: {base_year}\n")
    print(f"  {'年份':<10} {'β (HSS×Year)':>14}  {'SE':>8}  {'p值':>8}  显著性")
    print("-" * 60)

    pt_coefs = {}
    try:
        m_pt = smf.ols(
            f"Log_RCR ~ HSS_Continuous * C({year_col}, Treatment({base_year})) "
            f"+ is_hss_label + HSS_Diversity + log_team_size + log_prior_knowledge",
            data=df_phase1
        ).fit()

        for v in m_pt.params.index:
            if f'HSS_Continuous:C({year_col}' in v:
                yr_str = v.split('[T.')[-1].rstrip(']')
                try:
                    yr = int(float(yr_str))
                except ValueError:
                    continue
                coef = m_pt.params[v]
                se   = m_pt.bse[v]
                pval = m_pt.pvalues[v]
                star = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
                pt_coefs[yr] = (coef, se, pval)
                print(f"  {str(yr):<10} {coef:>+14.4f}  {se:>8.4f}  {pval:>8.4f}  {star}")

        sig_count = sum(1 for _, (c, s, p) in pt_coefs.items() if p < 0.1)
        total     = len(pt_coefs)
        print()
        if sig_count == 0:
            print(f"  ✅ 平行趋势成立：Phase1 内 {total} 个年份交叉项均不显著（p > 0.1）")
        elif sig_count / max(total, 1) <= 0.25:
            print(f"  ⚠️  基本满足平行趋势：{total} 个年份中仅 {sig_count} 个显著（建议在论文中注释说明）")
        else:
            print(f"  ❌ 平行趋势存疑：{total} 个年份中有 {sig_count} 个显著（p < 0.1），需进一步检查")

    except Exception as e:
        print(f"  ❌ 平行趋势回归报错：{e}")

    print()
    return pt_coefs, base_year


# ================================================================
# 6. Event Study 图
# ================================================================
def run_event_study(df, year_col, base_year=2016):
    print("=" * 80)
    print(f"📊 Step 5：Event Study（全时段逐年系数图，基准年 = {base_year}）")
    print("=" * 80)

    # 6.1 拟合逐年交叉项模型（全样本）
    m_event = smf.ols(
        f"Log_RCR ~ HSS_Continuous * C({year_col}, Treatment({base_year})) "
        f"+ is_hss_label + HSS_Diversity + log_team_size "
        f"+ log_prior_knowledge + C(primary_field)",
        data=df
    ).fit()

    # 6.2 提取交叉项系数
    records = []
    for v in m_event.params.index:
        if f"HSS_Continuous:C({year_col}" in v:
            yr_str = v.split('[T.')[-1].rstrip(']')
            try:
                yr = int(float(yr_str))
            except ValueError:
                continue
            records.append({
                'year': yr,
                'coef': m_event.params[v],
                'se':   m_event.bse[v],
                'pval': m_event.pvalues[v]
            })

    results = pd.DataFrame(records).sort_values('year').reset_index(drop=True)

    # 插入基准年（系数=0）
    base_row = pd.DataFrame([{'year': base_year, 'coef': 0.0, 'se': 0.0, 'pval': np.nan}])
    results  = pd.concat([results, base_row]).sort_values('year').reset_index(drop=True)
    results['ci_lo'] = results['coef'] - 1.96 * results['se']
    results['ci_hi'] = results['coef'] + 1.96 * results['se']

    # 6.3 打印系数表
    print(f"\n  {'年份':<8} {'β':>10}  {'SE':>8}  {'p值':>8}  显著性")
    print("-" * 55)
    for _, row in results.iterrows():
        yr   = int(row['year'])
        star = ''
        if not np.isnan(row['pval']):
            star = ('***' if row['pval'] < 0.01 else
                    ('**'  if row['pval'] < 0.05 else
                     ('*'   if row['pval'] < 0.1  else '')))
        base_marker = ' ← base' if yr == base_year else ''
        pval_str = f"{row['pval']:>8.4f}" if not np.isnan(row['pval']) else '     ---'
        print(f"  {yr:<8} {row['coef']:>+10.4f}  {row['se']:>8.4f}  {pval_str}  {star}{base_marker}")

    # 6.4 绘图
    fig, ax = plt.subplots(figsize=(13, 6))

    years = results['year'].values
    coefs = results['coef'].values
    ci_lo = results['ci_lo'].values
    ci_hi = results['ci_hi'].values
    pvals = results['pval'].values

    # 阶段背景色
    y_min_raw = min(ci_lo) - 0.02
    y_max_raw = max(ci_hi) + 0.05

    phase_regions = [
        (years.min(), 2016,       '#d0e8ff', 'Phase 1\n(Pre-Transformer)'),
        (2017,        2021,       '#fff3cd', 'Phase 2\n(Post-Transformer)'),
        (2022,        years.max(),'#d4edda', 'Phase 3\n(Post-ChatGPT)'),
    ]
    for x_start, x_end, color, label in phase_regions:
        ax.axvspan(x_start - 0.5, x_end + 0.5,
                   alpha=0.25, color=color, zorder=0)
        ax.text((x_start + x_end) / 2, y_max_raw,
                label, ha='center', va='bottom',
                fontsize=9, color='gray', style='italic')

    # 辅助线
    ax.axhline(0,         color='black',   linewidth=0.9, linestyle='--', alpha=0.5, zorder=1)
    ax.axvline(base_year, color='dimgray', linewidth=1.2, linestyle=':',  alpha=0.8, zorder=1)
    for boundary in [2016.5, 2021.5]:
        ax.axvline(boundary, color='gray', linewidth=0.8,
                   linestyle='--', alpha=0.4, zorder=1)

    # 置信区间带 & 折线
    ax.fill_between(years, ci_lo, ci_hi,
                    alpha=0.18, color='steelblue', zorder=2, label='95% CI')
    ax.plot(years, coefs,
            color='steelblue', linewidth=2.0,
            marker='o', markersize=7, zorder=3,
            label='β (HSS_Continuous × Year)')

    # 显著点标红
    sig_mask = np.array([not np.isnan(p) and p < 0.05 for p in pvals])
    if sig_mask.any():
        ax.scatter(years[sig_mask], coefs[sig_mask],
                   color='tomato', s=70, zorder=4,
                   label='Significant (p < 0.05)')

    # 基准年标注
    ax.annotate(f'Base: {base_year}',
                xy=(base_year, 0),
                xytext=(base_year + 0.3, y_min_raw + 0.01),
                fontsize=9, color='dimgray',
                arrowprops=dict(arrowstyle='->', color='dimgray', lw=0.8))

    ax.set_ylim(y_min_raw, y_max_raw + 0.04)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Coefficient of HSS_Continuous × Year\n(Relative to Base Year)', fontsize=11)
    ax.set_title(
        'Event Study: Effect of HSS DNA on Citation Impact Across GenAI Eras\n'
        '(OLS with Field FE; 95% CI)',
        fontsize=13, fontweight='bold', pad=14
    )
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(fontsize=10, loc='upper left')

    plt.tight_layout()
    out_path = 'event_study_hss_premium.png'
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.show()
    print(f"\n  ✅ Event Study 图已保存：{out_path}\n")

    return results


# ================================================================
# 主函数
# ================================================================
def main():
    print("🚀 启动完整分析...\n")

    # Step 0：加载数据
    df = load_data()

    # Step 1：特征工程
    df, year_col = feature_engineering(df)

    # Step 2：主回归
    controls_fe = run_main_regressions(df)

    # Step 3：DID
    run_did(df, controls_fe)

    # Step 4：异质性分析
    run_heterogeneity(df)

    # Step 5：平行趋势检验
    pt_coefs, pre_base_year = run_parallel_trends(df, year_col)

    # Step 6：Event Study（以 Phase1 末年 2016 为基准）
    event_results = run_event_study(df, year_col, base_year=2016)

    print("🎉 全部分析完成！")
    return event_results


if __name__ == "__main__":
    main()
