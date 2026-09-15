import pandas as pd


def generate_full_did_table_data():
    print("🚀 启动【全周期 DID 面板】描述性统计生成引擎...\n")

    # 1. 读取数据
    try:
        df_panel = pd.read_csv('arXiv_Balanced_DID_Panel.csv')
        df_elite = pd.read_csv('Author_Institutions_Map_Top100.csv')
    except FileNotFoundError as e:
        print(f"❌ 找不到文件: {e}")
        return

    # 2. 合并光环和地域
    df_merged = df_panel.merge(
        df_elite[['author_id', 'is_elite_overall', 'country_code']],
        on='author_id', how='left'
    )
    df_merged['is_elite_overall'] = df_merged['is_elite_overall'].fillna(0)
    df_merged['country_code'] = df_merged['country_code'].fillna('Unknown')
    df_merged['is_us'] = df_merged['country_code'].apply(lambda x: 1 if x == 'US' else 0)
    df_merged['is_cn'] = df_merged['country_code'].apply(lambda x: 1 if x == 'CN' else 0)

    # 3. 划分 Period: Pre-2022 (冲击前) vs Post-2022 (冲击后)
    # 基于你相对时间或者实际发表年份来划分
    df_merged['T0_year'] = df_merged['year'] - df_merged['relative_time']
    df_merged['Period'] = df_merged['T0_year'].apply(lambda x: 'Post-2022' if x >= 2022 else 'Pre-2022')

    # 4. 定义你要填入表格的变量
    vars_to_stat = ['mean_log_rcr', 'is_elite_overall', 'is_us', 'is_cn']
    if 'team_size' in df_merged.columns: vars_to_stat.append('team_size')
    if 'prior_knowledge' in df_merged.columns: vars_to_stat.append('prior_knowledge')

    # 5. 嵌套分组统计并格式化输出
    for var in vars_to_stat:
        print(f"=========================================================")
        print(f"📌 变量: {var.upper()}")
        print(f"=========================================================")

        for treated_val, group_name in [(1, "Treated (HSS Cross)"), (0, "Control (Pure STEM)")]:
            for period_val in ['Pre-2022', 'Post-2022']:

                subset = df_merged[(df_merged['is_treated'] == treated_val) & (df_merged['Period'] == period_val)][var]

                if len(subset) > 0:
                    obs = len(subset)
                    mean = subset.mean()
                    std = subset.std()
                    min_val = subset.min()
                    max_val = subset.max()

                    print(
                        f"[{group_name}] | {period_val} | Obs: {obs:<6} | Mean: {mean:.4f} | Std: {std:.4f} | Min: {min_val:.2f} | Max: {max_val:.2f}")


generate_full_did_table_data()