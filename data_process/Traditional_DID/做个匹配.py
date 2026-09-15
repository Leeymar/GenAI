import pandas as pd
import numpy as np
import json
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False
import warnings
warnings.filterwarnings('ignore')


def compute_smd(df, features, group_col='Treated'):
    rows = []
    for feat in features:
        mean_t = df[df[group_col] == 1][feat].mean()
        mean_c = df[df[group_col] == 0][feat].mean()
        std_t  = df[df[group_col] == 1][feat].std()
        std_c  = df[df[group_col] == 0][feat].std()
        pooled = np.sqrt((std_t**2 + std_c**2) / 2)
        rows.append({'Feature': feat, 'SMD': round(abs(mean_t - mean_c) / (pooled + 1e-9), 4)})
    return pd.DataFrame(rows)


def print_reg(results, keys, label, n, n_clusters):
    print("\n" + "=" * 65)
    print(label)
    print("=" * 65)
    tbl = results.summary2().tables[1]
    print(tbl.loc[[p for p in keys if p in tbl.index]].round(6))
    print(f"\n   R² = {results.rsquared:.4f} | N = {n} | 聚类数 = {n_clusters}")


def run_full_pipeline():

    # ==========================================================
    # 第一步：加载基准数据
    # ==========================================================
    print("⏳ 1. 加载基准数据...")
    data = []
    with open(r'E:\PythonProject\GenAI\data_process\DID_authors\Author_Baseline_Rebuilt_v2.jsonl',
              'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    df_base = pd.DataFrame(data)
    df_base['author_id'] = df_base['author_id'].astype(str).str.split('/').str[-1]

    match_cols = [
        'h_index',
        'first_pub_year',
        'pre_total_works',
        'pre_total_citations',
        'pre_growth_rate',
    ]
    for col in match_cols:
        df_base[col] = pd.to_numeric(df_base[col], errors='coerce').fillna(0)

    df_base['pre_growth_rate'] = df_base['pre_growth_rate'].clip(-1.0, 5.0)

    # ← 新增：对偏态变量做对数变换，降低离群值对 SMD 的影响
    df_base['log_pre_works']   = np.log1p(df_base['pre_total_works'])
    df_base['log_pre_cites']   = np.log1p(df_base['pre_total_citations'])
    df_base['log_h_index']     = np.log1p(df_base['h_index'])

    treated_df = df_base[df_base['is_treated'] == 1].copy()
    control_df = df_base[df_base['is_treated'] == 0].copy()
    pool_df    = pd.concat([treated_df, control_df], ignore_index=True)
    pool_df['Treated'] = pool_df['is_treated'].apply(lambda x: 1 if x == 1 else 0)

    print(f"📊 初始池：实验组 {len(treated_df)} 人 | 对照组 {len(control_df)} 人")

    # ==========================================================
    # 第二步：CEM 分箱
    # ==========================================================
    print("\n⏳ 2. 执行 CEM...")

    bin_config = {
        'h_index':             [0, 1, 2, 4, 7, 11, 16, 25, 40, 9999],
        'first_pub_year':      [1960, 1985, 1992, 1997, 2002, 2007, 2012, 2016, 2020, 9999],
        'pre_total_works':     [0, 1, 3, 6, 12, 25, 50, 100, 9999],
        'pre_total_citations': [0, 5, 20, 80, 250, 700, 9999],
        'pre_growth_rate':     [-999, -0.3, -0.1, 0.0, 0.1, 0.3, 0.6, 1.2, 9999],
    }

    df_cem = pool_df.copy()
    for col, bins in bin_config.items():
        df_cem[f'{col}_bin'] = pd.cut(
            df_cem[col], bins=bins, labels=False, right=False
        ).astype('Int64')

    bin_cols = [f'{c}_bin' for c in bin_config.keys()]
    df_cem['cem_stratum'] = df_cem[bin_cols].astype(str).agg('_'.join, axis=1)

    strata_counts = df_cem.groupby(['cem_stratum', 'Treated']).size().unstack(fill_value=0)
    strata_counts.columns = ['n_control', 'n_treated']
    valid_strata = strata_counts[
        (strata_counts['n_treated'] > 0) & (strata_counts['n_control'] > 0)
    ].index
    print(f"   总层数：{len(strata_counts)}，有效层数：{len(valid_strata)}")

    df_valid = df_cem[df_cem['cem_stratum'].isin(valid_strata)].copy()
    print(f"   有效层内：实验组 {(df_valid['Treated']==1).sum()} 人 | "
          f"对照组 {(df_valid['Treated']==0).sum()} 人")

    # ==========================================================
    # 第三步：层内 1:2 无放回抽样
    # ==========================================================
    np.random.seed(42)
    matched_treated_list = []
    matched_control_list = []

    for stratum in valid_strata:
        t_grp = df_valid[(df_valid['cem_stratum'] == stratum) & (df_valid['Treated'] == 1)]
        c_grp = df_valid[(df_valid['cem_stratum'] == stratum) & (df_valid['Treated'] == 0)]
        n_c   = min(len(t_grp) * 2, len(c_grp))
        matched_treated_list.append(t_grp)
        matched_control_list.append(c_grp.sample(n=n_c, replace=False))

    matched_treated = pd.concat(matched_treated_list, ignore_index=True)
    matched_control = pd.concat(matched_control_list, ignore_index=True)
    matched_all     = pd.concat([matched_treated, matched_control], ignore_index=True)

    matched_t_ids   = set(matched_treated['author_id'])
    matched_c_ids   = set(matched_control['author_id'])
    all_matched_ids = matched_t_ids | matched_c_ids

    print(f"\n✅ 匹配完成：实验组 {len(matched_treated)} 人 | 对照组 {len(matched_control)} 人")

    # ==========================================================
    # 第四步：匹配质量诊断
    # ==========================================================
    pool_df['Treated_flag']     = pool_df['Treated']
    matched_all['Treated_flag'] = matched_all['Treated']

    smd_before = compute_smd(pool_df,     match_cols, group_col='Treated_flag')
    smd_before.rename(columns={'SMD': 'SMD_匹配前'}, inplace=True)
    smd_after  = compute_smd(matched_all, match_cols, group_col='Treated_flag')
    smd_after.rename(columns={'SMD': 'SMD_匹配后'}, inplace=True)

    smd_report = smd_before.merge(smd_after, on='Feature')
    smd_report['均衡（<0.1）'] = smd_report['SMD_匹配后'].apply(
        lambda x: '✅' if x < 0.1 else '⚠️'
    )
    print("\n📋 匹配质量诊断：")
    print(smd_report.to_string(index=False))

    # ==========================================================
    # 第五步：加载面板数据 & 合并协变量
    # ==========================================================
    print("\n⏳ 3. 加载面板数据...")
    df_panel = pd.read_csv(
        r'E:\PythonProject\GenAI\data_process\DID_authors\Ultimate_DID_Panel.csv'
    )
    df_panel['author_id'] = df_panel['author_id'].astype(str).str.split('/').str[-1]

    df_lab = df_panel[
        df_panel['author_id'].isin(all_matched_ids) &
        df_panel['year'].between(2021, 2025)
    ].copy()

    # ← 新增：把协变量合并进面板（用于双重稳健回归）
    cov_cols = ['author_id', 'h_index', 'first_pub_year',
                'pre_total_works', 'pre_total_citations', 'pre_growth_rate',
                'log_h_index', 'log_pre_works', 'log_pre_cites']
    df_lab = df_lab.merge(
        df_base[cov_cols].drop_duplicates('author_id'),
        on='author_id', how='left'
    )

    n_authors = df_lab['author_id'].nunique()
    print(f"🔍 面板诊断：行数 = {len(df_lab)} | 作者数 = {n_authors}")

    df_lab['Treated']    = df_lab['author_id'].isin(matched_t_ids).astype(int)
    df_lab['Post']       = (df_lab['year'] >= 2023).astype(int)
    df_lab['year_trend'] = df_lab['year'] - 2021

    # ==========================================================
    # 第六步：主回归（规范A：标准 DID）
    # ==========================================================
    key_did = ['Intercept', 'Treated', 'Post', 'Treated:Post']

    results_a = smf.ols(
        'avg_logRCR ~ Treated + Post + Treated:Post', data=df_lab
    ).fit(cov_type='cluster', cov_kwds={'groups': df_lab['author_id']})
    print_reg(results_a, key_did,
              "🎓 规范A：标准 DID", len(df_lab), n_authors)

    # ==========================================================
    # 第七步：稳健性检验（规范B：加线性时间趋势）
    # ==========================================================
    results_b = smf.ols(
        'avg_logRCR ~ Treated + Post + Treated:Post + year_trend', data=df_lab
    ).fit(cov_type='cluster', cov_kwds={'groups': df_lab['author_id']})
    print_reg(results_b, key_did + ['year_trend'],
              "🔍 规范B：加线性时间趋势", len(df_lab), n_authors)

    # ==========================================================
    # 第八步：双重稳健回归（规范C：DID + 协变量控制）← 新增
    # ==========================================================
    formula_dr = (
        'avg_logRCR ~ Treated + Post + Treated:Post'
        ' + log_h_index + first_pub_year'
        ' + log_pre_works + log_pre_cites + pre_growth_rate'
    )
    results_c = smf.ols(formula_dr, data=df_lab).fit(
        cov_type='cluster', cov_kwds={'groups': df_lab['author_id']}
    )
    print_reg(results_c, key_did,
              "🛡️  规范C：双重稳健 DID（加协变量控制）", len(df_lab), n_authors)

    # 三个规范 Treated:Post 系数对比
    print("\n📊 核心系数对比（Treated:Post）：")
    print(f"   规范A（标准DID）       : {results_a.params['Treated:Post']:.4f}  "
          f"(p={results_a.pvalues['Treated:Post']:.4f})")
    print(f"   规范B（+时间趋势）     : {results_b.params['Treated:Post']:.4f}  "
          f"(p={results_b.pvalues['Treated:Post']:.4f})")
    print(f"   规范C（+协变量控制）   : {results_c.params['Treated:Post']:.4f}  "
          f"(p={results_c.pvalues['Treated:Post']:.4f})")

    # ==========================================================
    # 第九步：平行趋势检验（以 2022 为基准）
    # ==========================================================
    print("\n⏳ 4. 平行趋势检验...")

    event_years = [y for y in sorted(df_lab['year'].unique()) if y != 2022]
    for y in event_years:
        df_lab[f'T_x_{y}'] = (df_lab['year'] == y).astype(int) * df_lab['Treated']

    interaction_terms = ' + '.join([f'T_x_{y}' for y in event_years])

    # ← 修改：平行趋势检验也加入协变量控制，使结论更稳健
    formula_es = (
        f'avg_logRCR ~ Treated + Post + {interaction_terms}'
        ' + log_h_index + first_pub_year'
        ' + log_pre_works + log_pre_cites + pre_growth_rate'
    )
    results_es = smf.ols(formula_es, data=df_lab).fit(
        cov_type='cluster', cov_kwds={'groups': df_lab['author_id']}
    )

    tbl_es = results_es.summary2().tables[1]
    coef_rows = []
    for y in event_years:
        key_es = f'T_x_{y}'
        if key_es in tbl_es.index:
            r = tbl_es.loc[key_es]
            coef_rows.append({
                'year': y, 'coef': r['Coef.'],
                'ci_lo': r['[0.025'], 'ci_hi': r['0.975]']
            })

    df_es = pd.concat([
        pd.DataFrame(coef_rows),
        pd.DataFrame([{'year': 2022, 'coef': 0.0, 'ci_lo': 0.0, 'ci_hi': 0.0}])
    ]).sort_values('year').reset_index(drop=True)

    print(f"\n{'年份':>6} {'系数':>10} {'CI下界':>10} {'CI上界':>10} {'阶段':>8}")
    print("-" * 50)
    for _, r in df_es.iterrows():
        stage = '基准' if r['year'] == 2022 else ('处理前' if r['year'] < 2023 else '处理后')
        sig   = ' *' if (r['year'] != 2022 and
                         (r['ci_lo'] > 0 or r['ci_hi'] < 0)) else ''
        print(f"{int(r['year']):>6} {r['coef']:>10.4f} "
              f"{r['ci_lo']:>10.4f} {r['ci_hi']:>10.4f} {stage:>8}{sig}")

    # 绘图
    fig, ax = plt.subplots(figsize=(9, 5))
    pre  = df_es[df_es['year'] <  2023]
    post = df_es[df_es['year'] >= 2023]
    for subset, color, label in [
        (pre,  '#999999', '处理前'),
        (post, '#E76F51', '处理后')
    ]:
        ax.plot(subset['year'], subset['coef'], 'o-',
                color=color, label=label, linewidth=2, markersize=7)
        ax.fill_between(subset['year'], subset['ci_lo'], subset['ci_hi'],
                        alpha=0.15, color=color)

    ax.axhline(0,      color='black', linewidth=0.8)
    ax.axvline(2022.5, color='navy',  linewidth=1.2,
               linestyle='--', label='ChatGPT 发布（2023）')
    ax.set_xlabel('年份')
    ax.set_ylabel('DiD 系数（相对于 2022 基准）')
    ax.set_title('平行趋势检验：事件研究图（协变量控制后）')
    ax.set_xticks(sorted(df_lab['year'].unique()))
    ax.legend()
    plt.tight_layout()
    plt.savefig('parallel_trend_event_study.png', dpi=150)
    print("\n   📊 事件研究图已保存：parallel_trend_event_study.png")
    print("\n✅ 全部完成！")


if __name__ == "__main__":
    run_full_pipeline()
