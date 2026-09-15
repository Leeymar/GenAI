import pandas as pd
import statsmodels.formula.api as smf


def run_ultimate_3d_robust_did():
    print("🚀 启动【时间 + 阶级 + 空间】三维防御版 DID 回归...\n")

    # 1. 读取面板数据和机构/国家数据
    try:
        df_panel = pd.read_csv('arXiv_Balanced_DID_Panel.csv')
        df_elite = pd.read_csv('Author_Institutions_Map_Top100.csv')  # 这个表里包含 country_code
    except FileNotFoundError as e:
        print(f"❌ 找不到文件: {e}")
        return

    # 2. 合并数据，把光环和国家代码一起带过来
    df_merged = df_panel.merge(
        df_elite[['author_id', 'is_elite_overall', 'country_code']],
        on='author_id', how='left'
    )

    # 填补缺失值
    df_merged['is_elite_overall'] = df_merged['is_elite_overall'].fillna(0)
    df_merged['country_code'] = df_merged['country_code'].fillna('Unknown')

    # 3. 构建地域控制变量 (中美双雄)
    df_merged['is_us'] = df_merged['country_code'].apply(lambda x: 1 if x == 'US' else 0)
    df_merged['is_cn'] = df_merged['country_code'].apply(lambda x: 1 if x == 'CN' else 0)

    # 4. 大模型时代滤镜
    df_merged['T0_year'] = df_merged['year'] - df_merged['relative_time']
    df_llm = df_merged[df_merged['T0_year'] >= 2022].copy()
    df_llm['year_str'] = df_llm['year'].astype(str)

    # 5. 终极模型公式：交互项 + 年份(时间) + 精英光环(阶级) + 中美地域(空间)
    formula = "mean_log_rcr ~ is_treated * Post + C(year_str) + is_elite_overall + is_us + is_cn"

    # 6. 拟合模型
    model = smf.ols(formula, data=df_llm).fit(cov_type='HC1')

    print("=========================================================")
    print("🎯 【全副武装版】DID 回归结果 (已控制名校、大厂与中美地域)")
    print("=========================================================")
    res_df = pd.DataFrame({
        'Coefficient': model.params,
        'Std.Error': model.bse,
        'P-value': model.pvalues
    })

    # 展示核心变量
    print(res_df.loc[['is_treated:Post', 'is_elite_overall', 'is_us', 'is_cn']])
    print("=========================================================")


run_ultimate_3d_robust_did()