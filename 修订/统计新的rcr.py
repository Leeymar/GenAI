import pandas as pd
import numpy as np


# =========================
# 1. 数据路径
# =========================

input_csv = r"E:\PythonProject\GenAI\data_process\derived\genai_regression_enriched_v3_fixed.csv"


# =========================
# 2. 读取数据
# =========================

df = pd.read_csv(input_csv)


# =========================
# 3. 重新构造 Log_RCR
# =========================

# 原：
# Log_RCR = ln(RCR+1)

# 修改为：
# Log_RCR = ln(RCR+0.01)

df["Log_RCR"] = np.log(df["RCR"] + 0.01)



# =========================
# 4. 描述性统计变量
# =========================

# 根据你的表1/描述统计调整
variables = [
    "Log_RCR",
    "Citation_Field_Diversity",
    "HSS_Citing_Share",
    "Citation_Entropy",
    "team_hss_dna_pct",
    "log_team_size",
    "log_prior_knowledge"
]


# =========================
# 5. 生成统计表
# =========================

desc = []

for var in variables:

    if var not in df.columns:
        print(f"⚠️ 缺少变量: {var}")
        continue

    x = df[var].dropna()

    desc.append({
        "变量": var,
        "观测值": len(x),
        "均值": round(x.mean(), 4),
        "标准差": round(x.std(), 4),
        "最小值": round(x.min(), 4),
        "最大值": round(x.max(), 4)
    })


desc_df = pd.DataFrame(desc)


# =========================
# 6. 输出
# =========================

output = "描述性统计_Log_RCR_001版本.xlsx"

desc_df.to_excel(
    output,
    index=False
)


print("完成！")
print(f"结果保存：{output}")

print("\n描述性统计：")
print(desc_df)