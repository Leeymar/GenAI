import pandas as pd

df = pd.read_csv("E:\PythonProject\GenAI\data_process\derived\genai_regression_enriched_v3_fixed.csv")

# 显性 HSS 主题标签：二选一
df["explicit_hss_topic"] = df["Socially_Embedded_Topic"]
# 如果你最终模型实际用的是 has_hss_concepts，则改成：
# df["explicit_hss_topic"] = df["has_hss_concepts"]

descvars = [
    "Log_RCR",
    "Hit_Rate_10_year",
    "Citation_Field_Diversity",
    "HSS_Citing_Share",
    "team_hss_dna_pct",
    "hss_reference_share",
    "log_team_size",
    "log_prior_knowledge",
    "HSS_Diversity",
    "explicit_hss_topic",
]

table = df[descvars].agg(["count", "mean", "std", "min", "max"]).T
table.columns = ["观测值", "均值", "标准差", "最小值", "最大值"]
table = table.round(4)
print(table)

table.to_excel("table1_descriptive_stats.xlsx")