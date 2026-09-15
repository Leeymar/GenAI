import pandas as pd
import numpy as np

# 1. 读取你刚刚合并好的最终版回归数据
df = pd.read_csv('Ultimate_Regression_Base_with_Diversity.csv')  # 请确保文件名对得上

# 2. 划定【控制组】和【实验组】
# 修正点 1：使用真实的列名 team_hss_dna_pct
df['Group'] = np.where(df['team_hss_dna_pct'] > 0,
                       'Experimental: HSS Cross',
                       'Control: Pure STEM')

# 3. 定义我们需要统计的核心变量
# 修正点 2：使用真实的列名 team_hss_dna_pct
variables_to_stat = {
    'team_hss_dna_pct': 'HSS_Concentration (文科浓度)',
    'HSS_Diversity': 'HSS_Diversity (认知多样性/标准差)'
}

print("📊 开始生成描述性统计数据...\n")

# 4. 循环计算并打印每个变量的结果
for col, display_name in variables_to_stat.items():
    print(f"======================================================")
    print(f"🌟 变量: {display_name}")
    print(f"======================================================")

    # 修正点 3：分组的列名改为 stage
    stats = df.groupby(['Group', 'stage'])[col].agg(
        Obs='count',
        Mean='mean',
        Std_Dev='std',
        Min='min',
        Max='max'
    ).fillna(0)  # 防止除以0产生NaN

    # 将浮点数保留三位小数，方便你直接填进 LaTeX
    stats['Mean'] = stats['Mean'].map('{:.3f}'.format)
    stats['Std_Dev'] = stats['Std_Dev'].map('{:.3f}'.format)
    stats['Min'] = stats['Min'].map('{:.2f}'.format)
    stats['Max'] = stats['Max'].map('{:.2f}'.format)

    print(stats)
    print("\n")

print("✅ 计算完成！请将上面打印出的小数直接填入你的 LaTeX 表格中。")