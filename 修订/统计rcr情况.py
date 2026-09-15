import pandas as pd

path = r"E:\PythonProject\GenAI\data_process\derived\genai_regression_enriched_v3_fixed.csv"

df = pd.read_csv(path)

print(df["RCR"].describe())

print("RCR=0比例:")
print((df["RCR"] == 0).mean())

print("RCR<1比例:")
print((df["RCR"] < 1).mean())

print("RCR分位数:")
print(df["RCR"].quantile([0,0.01,0.05,0.25,0.5,0.75,0.95,0.99,1]))