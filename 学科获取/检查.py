import pandas as pd

df_main  = pd.read_csv('E:\PythonProject\GenAI\data_process\\test\\Ultimate_Regression_Base_with_Diversity.csv')
df_field = pd.read_csv('work_id_to_field_all_stages.csv')

print("主表 work_id 样例：", df_main['work_id'].head(3).tolist())
print("学科表 work_id 样例：", df_field['work_id'].head(3).tolist())

# 测试正则提取
df_main['work_id_clean'] = df_main['work_id'].str.extract(r'(W\d+)$')
print("\n提取后样例：", df_main['work_id_clean'].head(3).tolist())

# 测试交集
overlap = set(df_main['work_id_clean']).intersection(set(df_field['work_id']))
print(f"\n交集数量：{len(overlap):,}（应该接近 138,829）")
