import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


def get_stars(p_value):
    """根据 p 值自动生成顶刊标准的显著性星号"""
    if p_value < 0.01:
        return "***"
    elif p_value < 0.05:
        return "**"
    elif p_value < 0.1:
        return "*"
    else:
        return ""


def run_robustness_table_extraction():
    print("🚀 启动【Table 5 稳健性检验表格】参数全自动提取引擎...\n")

    # 1. 加载数据与预处理
    df = pd.read_csv('Ultimate_Regression_Base.csv')
    df = df.dropna(
        subset=['year', 'team_hss_dna_pct', 'Log_RCR', 'total_authors', 'Prior_Knowledge_Stock', 'arxiv_type'])

    df['year_str'] = df['year'].astype(str)
    # 缩放连续变量：每增加 10%
    df['hss_dna_decile'] = df['team_hss_dna_pct'] / 10.0

    # ==========================================
    # Model 4: 仅包含连续客观 DNA 与 控制变量
    # ==========================================
    formula_m4 = (
        "Log_RCR ~ hss_dna_decile + np.log(total_authors) + np.log(Prior_Knowledge_Stock + 1) + C(year_str)"
    )
    model4 = smf.ols(formula_m4, data=df).fit(cov_type='HC1')

    # ==========================================
    # Model 5: 连续客观 DNA + 主观 arXiv 标签 + 控制变量
    # 假设你的 Type 0 是基准组，它会自动作为 reference
    # ==========================================
    formula_m5 = (
        "Log_RCR ~ hss_dna_decile + C(arxiv_type) + np.log(total_authors) + np.log(Prior_Knowledge_Stock + 1) + C(year_str)"
    )
    model5 = smf.ols(formula_m5, data=df).fit(cov_type='HC1')

    # ==========================================
    # 提取与格式化输出 (精准对标 LaTeX 表格)
    # ==========================================
    obs_count = int(model4.nobs)

    # 安全提取函数的辅助闭包
    def extract_val(model, var_keyword):
        # 寻找匹配的变量名
        for idx in model.params.index:
            if var_keyword in idx:
                coef = model.params[idx]
                se = model.bse[idx]
                p = model.pvalues[idx]
                return f"{coef:.4f}{get_stars(p)}", f"({se:.4f})"
        return "N/A", "(N/A)"

    # 提取 Model 4 参数
    m4_hss_coef, m4_hss_se = extract_val(model4, 'hss_dna_decile')
    m4_team_coef, _ = extract_val(model4, 'total_authors')

    # 提取 Model 5 参数
    m5_hss_coef, m5_hss_se = extract_val(model5, 'hss_dna_decile')
    m5_team_coef, _ = extract_val(model5, 'total_authors')

    # 提取 Subjective Labels (注意：你的数据里标签名字可能略有不同，请匹配你真实的 arxiv_type 字符串)
    m5_type1_coef, _ = extract_val(model5, 'Type 1')
    m5_type2_coef, _ = extract_val(model5, 'Type 2')
    m5_type3_coef, _ = extract_val(model5, 'Type 3')

    print("=========================================================")
    print("📋 请将以下数据直接填入 LaTeX 的 Table 5 中：")
    print("=========================================================\n")

    print("HSS DNA (per 10% increase) 系数与星号:")
    print(f"Model 4: {m4_hss_coef}   |   Model 5: {m5_hss_coef}")
    print(f"Model 4: {m4_hss_se}      |   Model 5: {m5_hss_se}")
    print("-" * 50)

    print("Subjective arXiv Category (Model 5 专属):")
    print(f"Type 1 (Math/Eng)    : {m5_type1_coef}")
    print(f"Type 2 (Hard Science): {m5_type2_coef}")
    print(f"Type 3 (HSS Cross)   : {m5_type3_coef}")
    print("-" * 50)

    print("Controls:")
    print(f"Log(Team Size) 系数:")
    print(f"Model 4: {m4_team_coef}   |   Model 5: {m5_team_coef}")
    print("-" * 50)

    print(f"Observations (样本量): {obs_count:,}")
    print("=========================================================")


run_robustness_table_extraction()