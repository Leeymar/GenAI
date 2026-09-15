import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col


def run_ultimate_regressions_v2_5bins():
    print("🚀 启动【跨界溢价 V2.0：5档粒度探测与结构控制】终极实证引擎...\n")

    # ==========================================
    # 1. 读取包含 HSS_Diversity 的最新终极表
    # ==========================================
    data_file = 'Ultimate_Regression_Base_with_Diversity.csv'
    df = pd.read_csv(data_file)
    print(f"✅ 成功加载数据！总样本量: {len(df)}")

    # ==========================================
    # 2. 变量核对与特征工程 (基于截图的精准列名)
    # ==========================================

    # 2.1 核心自变量 1：连续型浓度 (1 unit = 10%)
    # 使用精准列名: team_hss_dna_pct
    df['HSS_Continuous'] = df['team_hss_dna_pct'] / 10.0

    # 2.2 核心自变量 2：阶梯型浓度 (5 档分箱)
    print("🪓 正在切分文科基因浓度阵营 (5档精细探测)...")
    bins = [-0.1, 0.1, 20.0, 40.0, 60.0, 100.1]
    labels = [
        '1_Pure_STEM(0%)',
        '2_Low_HSS(1-20%)',
        '3_Moderate_HSS(21-40%)',
        '4_High_HSS(41-60%)',
        '5_HSS_Dominant(>60%)'
    ]
    df['HSS_Bin'] = pd.cut(df['team_hss_dna_pct'], bins=bins, labels=labels)

    # 2.3 主观标签：是否贴了文科标签 (Type 3)
    # 使用精准列名: arxiv_type
    df['is_hss_label'] = df['arxiv_type'].apply(lambda x: 1 if 'Type 3' in str(x) else 0)

    # 2.4 控制变量：人数和资历取对数防极值
    # 使用精准列名: total_authors, Prior_Knowledge_Stock
    df['log_team_size'] = np.log1p(df['total_authors'])
    df['log_prior_knowledge'] = np.log1p(df['Prior_Knowledge_Stock'])

    # ==========================================
    # 3. 拟合 5 大 OLS 模型
    # ==========================================
    print("📈 正在拟合 OLS 模型群...\n")

    # 统一的控制变量群：加上了 HSS_Diversity 和精准命名的 log_prior_knowledge
    # 固定效应使用精准列名: stage
    controls = "HSS_Diversity + log_team_size + log_prior_knowledge + C(stage)"

    # 注意因变量使用精准列名: Log_RCR
    # Model 1: Baseline DNA
    m1 = smf.ols(f"Log_RCR ~ C(HSS_Bin, Treatment('1_Pure_STEM(0%)')) + {controls}", data=df).fit()

    # Model 2: Subjective Label
    m2 = smf.ols(f"Log_RCR ~ is_hss_label + {controls}", data=df).fit()

    # Model 3: Full Joint Model
    m3 = smf.ols(f"Log_RCR ~ C(HSS_Bin, Treatment('1_Pure_STEM(0%)')) + is_hss_label + {controls}", data=df).fit()

    # Model 4: Robustness Base (Continuous)
    m4 = smf.ols(f"Log_RCR ~ HSS_Continuous + {controls}", data=df).fit()

    # Model 5: Robustness Joint (Continuous + Label)
    m5 = smf.ols(f"Log_RCR ~ HSS_Continuous + is_hss_label + {controls}", data=df).fit()

    # ==========================================
    # 4. 生成顶级期刊标准三线表
    # ==========================================
    regressor_order = [
        'HSS_Continuous',
        "C(HSS_Bin, Treatment('1_Pure_STEM(0%)'))[T.2_Low_HSS(1-20%)]",
        "C(HSS_Bin, Treatment('1_Pure_STEM(0%)'))[T.3_Moderate_HSS(21-40%)]",
        "C(HSS_Bin, Treatment('1_Pure_STEM(0%)'))[T.4_High_HSS(41-60%)]",
        "C(HSS_Bin, Treatment('1_Pure_STEM(0%)'))[T.5_HSS_Dominant(>60%)]",
        'is_hss_label',
        'HSS_Diversity',
        'log_team_size',
        'log_prior_knowledge'
    ]

    res_table = summary_col(
        [m1, m2, m3, m4, m5],
        model_names=['M1(Bin)', 'M2(Lab)', 'M3(Full)', 'M4(Cont)', 'M5(Full)'],
        stars=True,
        float_format='%0.3f',
        regressor_order=regressor_order,
        drop_omitted=True
    )

    print("=" * 95)
    print("🏆 包含【认知结构】、【历史资历】与【5档浓度】的终极跨界溢价回归表")
    print("=" * 95)
    print(res_table)
    print("\n💡 Note: *** p<0.01, ** p<0.05, * p<0.1")


if __name__ == "__main__":
    run_ultimate_regressions_v2_5bins()