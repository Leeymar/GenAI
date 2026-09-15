import pandas as pd
import pyfixest as pf
import warnings
warnings.filterwarnings('ignore')

def run_staggered_did():
    print("⏳ 加载数据...")
    df = pd.read_csv('Perfect_DID_Panel.csv')

    # --------------------------------------------------------
    # 必须有：每个作者的首次处理年份（cohort）
    # 如果原始数据里没有 event_year 列，用 relative_year 反推
    # --------------------------------------------------------
    df['author_id'] = df['author_id'].astype(str)

    # 反推处理队列年份：event_year = calendar_year - relative_year
    df['event_year'] = df['year'] - df['relative_year']

    # 对照组（从未处理）的 event_year 设为 9999（pyfixest 的约定）
    df.loc[df['is_treated'] == 0, 'event_year'] = 9999

    print(f"📊 处理队列分布：")
    print(df[df['is_treated'] == 1]['event_year']
          .value_counts().sort_index().to_string())

    # --------------------------------------------------------
    # Sun-Abraham 估计量
    # ref_cohort 指定从未处理组作为参照
    # --------------------------------------------------------
    print("\n🚀 运行 Sun-Abraham Staggered DID...")
    fit_sa = pf.feols(
        fml       = "avg_logRCR ~ sunab(event_year, year) | author_id + year",
        data      = df,
        vcov      = {"CRV1": "author_id"},   # 聚类稳健标准误
    )

    print("\n" + "=" * 60)
    print("🏆 Sun-Abraham 聚合处理效应（ATT）")
    print("=" * 60)
    fit_sa.summary()

    # --------------------------------------------------------
    # 提取聚合 ATT（所有处理期的加权平均）
    # --------------------------------------------------------
    iplot_data = pf.iplot(fit_sa, figsize=(10, 5), title="事件研究图（Sun-Abraham）")

    # --------------------------------------------------------
    # 打印简洁系数表
    # --------------------------------------------------------
    print("\n📋 各相对时间系数：")
    coef_df = fit_sa.tidy()
    print(coef_df[['Estimate', 'Std. Error', 'Pr(>|t|)']].round(4))

    # --------------------------------------------------------
    # 平行趋势检验：处理前系数是否联合不显著
    # --------------------------------------------------------
    pre_terms = [t for t in fit_sa.coefnames() if 'year::' in t and int(t.split('::')[1]) < 0]
    if pre_terms:
        joint_test = fit_sa.wald_test(restrictions=pre_terms)
        print(f"\n🧪 处理前平行趋势联合检验：F = {joint_test['statistic']:.4f}, "
              f"p = {joint_test['pvalue']:.4f}")
        print("   → " + ("✅ 平行趋势成立" if joint_test['pvalue'] > 0.05 else "⚠️ 平行趋势可能不成立"))


if __name__ == "__main__":
    run_staggered_did()
