import pandas as pd
df = pd.read_csv('Ultimate_Regression_Base_with_Diversity.csv')
print(df['arxiv_primary_category'].value_counts().head(20))
