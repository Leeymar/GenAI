import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col


def run_full_analysis():
    print("🚀 启动完整分析（原始5模型 + 学科FE + DID + 异质性）...\n")

    # ============================================================
    # 0. 读取并合并学科信息
    # ============================================================
    df = pd.read_csv('E:\PythonProject\GenAI\data_process\\test\\Ultimate_Regression_Base_with_Diversity.csv')
    df_field = pd.read_csv('work_id_to_field_all_stages.csv')

    # ✅ 学科表是完整 URL，提取短 ID
    df_field['work_id'] = df_field['work_id'].str.extract(r'(W\d+)$')

    df = df.merge(df_field[['work_id', 'primary_field']], on='work_id', how='left')
    df['primary_field'] = df['primary_field'].fillna('Other')

    print(f"✅ 合并完成，共 {len(df):,} 篇论文")
    print(df['primary_field'].value_counts())
    print()

    # ============================================================
    # 1. 特征工程（完整保留原始脚本逻辑）
    # ============================================================
    df['HSS_Continuous'] = df['team_hss_dna_pct'] / 10.0

    bins   = [-0.1, 0.1, 20.0, 40.0, 60.0, 100.1]
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

    ref_bin = "1_Pure_STEM(0%)"

    # ============================================================
    # 1.5 新增：计算 Hit Rate (同年份、同学科内部 Top 10% 爆款)
    # ============================================================
    print("🔥 正在计算爆款率 (Hit Rate Top 10%)...")
    # 过滤掉 RCR 为空的异常值（如果有的话）
    df['RCR'] = pd.to_numeric(df['RCR'], errors='coerce')

    # 核心逻辑：按 [年份, 学科] 分组，计算该组的 90 分位数，RCR 超过这个阈值的打 1，否则打 0
    # ✅ 推荐：用百分比排名，在组内严格取前10%，并处理空值
    df['Hit_Rate_10'] = df.groupby(['year', 'primary_field'])['RCR'].transform(
        lambda x: (x.rank(pct=True, method='min', na_option='keep') > 0.90).astype(int)
    )

    print(f"✅ Hit Rate 计算完成！全样本共产生 {df['Hit_Rate_10'].sum():,} 篇爆款神作。\n")

    # ============================================================
    # 跑一个 Hit Rate 的专属 LPM 模型 (线性概率模型)
    # ============================================================
    controls_fe = "HSS_Diversity + log_team_size + log_prior_knowledge + C(stage) + C(primary_field)"

    m_hit = smf.ols(f"Hit_Rate_10 ~ HSS_Continuous + is_hss_label + {controls_fe}", data=df).fit()

    print("=" * 80)
    print("🚀 补充 Table：颠覆性创新溢价 (Hit Rate Top 10% LPM 模型)")
    print("=" * 80)
    print(m_hit.summary().tables[1])
    print()

    # 1. 补充 Logit 边际效应，与 LPM 结论互相印证
    m_hit_logit = smf.logit(
        f"Hit_Rate_10 ~ HSS_Continuous + is_hss_label + {controls_fe}",
        data=df.dropna(subset=['Hit_Rate_10'])
    ).fit(maxiter=200)
    print(m_hit_logit.get_margeff().summary())

    # 2. 报告 LPM 预测值中超出 [0,1] 的比例（用于说明 LPM 合理性）
    fitted = m_hit.fittedvalues
    print(f"预测值 < 0 的比例: {(fitted < 0).mean():.2%}")
    print(f"预测值 > 1 的比例: {(fitted > 1).mean():.2%}")

    # ============================================================
    # 2. 原始 5 大模型（无学科 FE，完整保留）
    # ============================================================
    print("📈 拟合原始 5 大 OLS 模型...\n")
    controls    = "HSS_Diversity + log_team_size + log_prior_knowledge + C(stage)"
    controls_fe = controls + " + C(primary_field)"

    # ── 原始 5 个模型 ──────────────────────────────────────────
    m1 = smf.ols(f"Log_RCR ~ C(HSS_Bin, Treatment('{ref_bin}')) + {controls}",                          data=df).fit()
    m2 = smf.ols(f"Log_RCR ~ is_hss_label + {controls}",                                                data=df).fit()
    m3 = smf.ols(f"Log_RCR ~ C(HSS_Bin, Treatment('{ref_bin}')) + is_hss_label + {controls}",           data=df).fit()
    m4 = smf.ols(f"Log_RCR ~ HSS_Continuous + {controls}",                                              data=df).fit()
    m5 = smf.ols(f"Log_RCR ~ HSS_Continuous + is_hss_label + {controls}",                               data=df).fit()

    # ── 新增：加学科 FE 的对应版本（作为稳健性）──────────────
    m1_fe = smf.ols(f"Log_RCR ~ C(HSS_Bin, Treatment('{ref_bin}')) + {controls_fe}",                    data=df).fit()
    m3_fe = smf.ols(f"Log_RCR ~ C(HSS_Bin, Treatment('{ref_bin}')) + is_hss_label + {controls_fe}",     data=df).fit()
    m5_fe = smf.ols(f"Log_RCR ~ HSS_Continuous + is_hss_label + {controls_fe}",                         data=df).fit()

    # ============================================================
    # 3. 原始 5 模型三线表（完整还原）
    # ============================================================
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
        'N':           lambda x: f"{int(x.nobs):,}",
        'R²':          lambda x: f"{x.rsquared:.3f}",
        'Field FE':    lambda x: 'Yes' if 'C(primary_field)' in x.model.formula else 'No',
    }

    print("=" * 110)
    print("🏆 Table 1：原始 5 模型——跨界溢价主回归（无学科 FE）")
    print("=" * 110)
    t1 = summary_col(
        [m1, m2, m3, m4, m5],
        model_names=['M1(Bin)', 'M2(Lab)', 'M3(Full)', 'M4(Cont)', 'M5(Full)'],
        stars=True, float_format='%0.3f',
        regressor_order=regressor_order,
        drop_omitted=True,
        info_dict=info_dict
    )
    print(t1)
    print("💡 Note: *** p<0.01, ** p<0.05, * p<0.1\n")

    # ============================================================
    # 4. 学科 FE 稳健性三线表（新增，对应 M1/M3/M5）
    # ============================================================
    print("=" * 110)
    print("🔬 Table 2：学科固定效应稳健性检验（对应 M1/M3/M5 加入学科 FE）")
    print("=" * 110)
    t2 = summary_col(
        [m1, m1_fe, m3, m3_fe, m5, m5_fe],
        model_names=['M1', 'M1+FE', 'M3', 'M3+FE', 'M5', 'M5+FE'],
        stars=True, float_format='%0.3f',
        regressor_order=regressor_order,
        drop_omitted=True,
        info_dict=info_dict
    )
    print(t2)
    print("💡 Note: *** p<0.01, ** p<0.05, * p<0.1\n")

    # ============================================================
    # 5. DID：文科基因 × GenAI 阶段交叉项
    # ============================================================
    m_did = smf.ols(
        f"Log_RCR ~ HSS_Continuous * C(stage) + is_hss_label + {controls_fe}",
        data=df
    ).fit()

    print("=" * 80)
    print("📊 Table 3：DID——文科基因 × GenAI 时代交叉项")
    print("=" * 80)
    did_vars = [c for c in m_did.params.index if 'HSS_Continuous:' in c]
    print(f"  {'变量':<50} {'β':>8}  {'p值':>7}  显著性")
    print("-" * 75)
    for v in did_vars:
        coef = m_did.params[v]
        pval = m_did.pvalues[v]
        star = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
        print(f"  {v:<50} {coef:>+8.4f}  {pval:>7.4f}  {star}")
    print()

    # ============================================================
    # 6. 异质性分析：按学科跑子样本
    # ============================================================
    print("=" * 80)
    print("🔬 Table 4：异质性分析——各学科中的文科基因溢价（HSS_Continuous）")
    print("=" * 80)
    print(f"  {'学科':<35} {'β':>8}  {'SE':>7}  {'p值':>7}  显著性  {'N':>8}")
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


if __name__ == "__main__":
    run_full_analysis()
