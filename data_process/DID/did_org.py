import pandas as pd
import statsmodels.formula.api as smf


def run_ultimate_robust_did():
    print("🚀 启动【终极控制版】DID 回归分析...\n")

    # 1. 读取面板数据和刚打好标的精英数据
    try:
        df_panel = pd.read_csv('arXiv_Balanced_DID_Panel.csv')
        df_elite = pd.read_csv('Author_Institutions_Map_Top100.csv')
    except FileNotFoundError as e:
        print(f"❌ 找不到文件: {e}")
        return

    # 2. 合并数据 (Left Join)
    df_merged = df_panel.merge(df_elite[['author_id', 'is_elite_overall', 'is_industry_elite', 'is_academic_elite']],
                               on='author_id', how='left')

    # 填补缺失值：如果在精英表里没匹配上，说明是没有机构的独立学者，光环为 0
    df_merged['is_elite_overall'] = df_merged['is_elite_overall'].fillna(0)
    df_merged['is_industry_elite'] = df_merged['is_industry_elite'].fillna(0)
    df_merged['is_academic_elite'] = df_merged['is_academic_elite'].fillna(0)

    # 3. 大模型时代滤镜 (2022年之后)
    df_merged['T0_year'] = df_merged['year'] - df_merged['relative_time']
    df_llm = df_merged[df_merged['T0_year'] >= 2022].copy()
    df_llm['year_str'] = df_llm['year'].astype(str)

    # 4. 终极模型公式：因变量 ~ 核心交互项 + 年份固定效应 + 精英光环控制
    formula = "mean_log_rcr ~ is_treated * Post + C(year_str) + is_elite_overall"

    # 5. 拟合模型 (使用 HC1 稳健标准误)
    model = smf.ols(formula, data=df_llm).fit(cov_type='HC1')

    print("=========================================================")
    print("🎯 加入了【Top 100 + 工业界大厂光环】控制的最终 DID 结果")
    print("=========================================================")
    res_df = pd.DataFrame({
        'Coefficient': model.params,
        'Std.Error': model.bse,
        'P-value': model.pvalues
    })

    # 只提取我们最关心的几行展示
    print(res_df.loc[['is_treated:Post', 'is_elite_overall', 'is_treated', 'Post']])
    print("=========================================================")
    print(f"参与回归的总样本量 (Observations): {int(model.nobs):,}")
    print(f"模型 R-squared: {model.rsquared:.4f}")


run_ultimate_robust_did()