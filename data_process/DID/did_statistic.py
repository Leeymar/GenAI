import pandas as pd
import numpy as np


def generate_did_descriptive_stats():
    print("🚀 启动【DID 实验组 vs 对照组】描述性统计引擎...\n")

    # 1. 读取面板数据和刚打好标的精英数据 (和跑回归时一样的逻辑)
    try:
        df_panel = pd.read_csv('arXiv_Balanced_DID_Panel.csv')
        df_elite = pd.read_csv('Author_Institutions_Map_Top100.csv')
    except FileNotFoundError as e:
        print(f"❌ 找不到文件: {e}")
        return

    # 2. 合并数据，带入光环和地域特征
    df_merged = df_panel.merge(
        df_elite[['author_id', 'is_elite_overall', 'country_code']],
        on='author_id', how='left'
    )

    # 填补缺失值并生成虚拟变量
    df_merged['is_elite_overall'] = df_merged['is_elite_overall'].fillna(0)
    df_merged['country_code'] = df_merged['country_code'].fillna('Unknown')
    df_merged['is_us'] = df_merged['country_code'].apply(lambda x: 1 if x == 'US' else 0)
    df_merged['is_cn'] = df_merged['country_code'].apply(lambda x: 1 if x == 'CN' else 0)

    # 3. 大模型时代滤镜 (锁定参与最终回归的 2022 年后 LLM 时代核心样本)
    df_merged['T0_year'] = df_merged['year'] - df_merged['relative_time']
    df_llm = df_merged[df_merged['T0_year'] >= 2022].copy()

    # 4. 定义需要统计的核心变量
    # 这里列出了 DID 回归中用到的主要变量
    # 如果你的面板里记录了协变量（如 'team_size' 或 'cum_papers'），请在下表替换成你的实际列名
    vars_to_stat = ['mean_log_rcr', 'is_elite_overall', 'is_us', 'is_cn']

    # 尝试自动捕获潜在的控制变量列 (为了匹配前后的平衡性展示)
    potential_covariates = ['team_size', 'avg_coauthors', 'cum_papers', 'prior_knowledge']
    for col in potential_covariates:
        if col in df_llm.columns:
            vars_to_stat.append(col)

    # 5. 提取对照组 (0) 和实验组 (1) 进行对比统计
    control_df = df_llm[df_llm['is_treated'] == 0][vars_to_stat]
    treated_df = df_llm[df_llm['is_treated'] == 1][vars_to_stat]

    print("=========================================================")
    print("📊 DID 匹配后样本描述性统计 (Post-Matching Balance Table)")
    print("=========================================================")

    print("\n【对照组 / Control Group (Pure STEM)】")
    control_stats = control_df.describe().T[['count', 'mean', 'std', 'min', 'max']]
    print(control_stats.to_string(float_format="{:.4f}".format))

    print("\n【实验组 / Treated Group (HSS Cross)】")
    treated_stats = treated_df.describe().T[['count', 'mean', 'std', 'min', 'max']]
    print(treated_stats.to_string(float_format="{:.4f}".format))

    print("\n=========================================================")
    print("💡 审稿人看点提示：")
    print("请对比两组中 is_elite_overall, is_us, is_cn (以及其他协变量) 的 'mean' (均值)。")
    print("如果两组的这些均值非常接近，说明你的 PSM 匹配极其成功，两边势均力敌！")
    print("=========================================================")


generate_did_descriptive_stats()