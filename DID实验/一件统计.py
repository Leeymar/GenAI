import pandas as pd

# 1. 读取超级面板
df = pd.read_csv('Ultimate_DID_Panel.csv')

# 2. 【关键修复】动态生成中美学者标签
df['country_code'] = df['country_code'].astype(str).str.strip().str.upper()
df['is_CN_US'] = df['country_code'].isin(['CN', 'US']).astype(int)

# 3. 选择需要展示在统计表中的核心变量
vars_to_summarize = [
    'avg_logRCR',           # 因变量：被引影响力
    'is_treated',           # 核心自变量：是否为跨界学者
    'Post',                 # 核心自变量：是否在跨界之后
    'yearly_paper_count',   # 控制变量：年度发文量
    'is_elite_overall',     # 异质性变量：是否为精英机构
    'is_CN_US'              # 异质性变量：是否为中美学者
]

# 4. 提取并格式化统计量
summary_stats = df[vars_to_summarize].describe().T
summary_stats = summary_stats[['count', 'mean', 'std', 'min', 'max']]

# 为了方便填入 LaTeX，保留 4 位小数
print("\n========== 描述性统计结果 ==========")
print(summary_stats.round(4).to_string())
print("====================================")