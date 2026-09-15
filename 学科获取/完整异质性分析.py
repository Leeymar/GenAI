import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col

# ============================================================
# 0. 读取并合并两张表
# ============================================================
df_main  = pd.read_csv('E:\PythonProject\GenAI\data_process\\test\\Ultimate_Regression_Base_with_Diversity.csv')
df_field = pd.read_csv('work_id_to_field_all_stages.csv')  # 上一步生成的
# ✅ 正确：提取的是学科表（学科表才是完整 URL）
df_field['work_id'] = df_field['work_id'].str.extract(r'(W\d+)$')

df = df_main.merge(df_field[['work_id', 'primary_field']], on='work_id', how='left')
df['primary_field'] = df['primary_field'].fillna('Other')

print(f"✅ 合并完成，共 {len(df):,} 篇论文")
print(df['primary_field'].value_counts())

# ============================================================
# 1. 特征工程（与你原代码保持一致）
# ============================================================
df['HSS_Continuous']      = df['team_hss_dna_pct'] / 10.0
df['log_team_size']       = np.log1p(df['total_authors'])
df['log_prior_knowledge'] = np.log1p(df['Prior_Knowledge_Stock'])
df['is_hss_label']        = df['arxiv_type'].apply(lambda x: 1 if 'Type 3' in str(x) else 0)

bins   = [-0.1, 0.1, 20.0, 40.0, 60.0, 100.1]
labels = ['1_Pure_STEM(0%)', '2_Low_HSS(1-20%)', '3_Moderate_HSS(21-40%)',
          '4_High_HSS(41-60%)', '5_HSS_Dominant(>60%)']
df['HSS_Bin'] = pd.cut(df['team_hss_dna_pct'], bins=bins, labels=labels)

# ============================================================
# 2. 主回归（复现你的原模型，加 primary_field 固定效应）
# ============================================================
controls_base = "HSS_Diversity + log_team_size + log_prior_knowledge + C(stage)"
controls_fe   = controls_base + " + C(primary_field)"  # 加学科固定效应

ref_bin = "1_Pure_STEM(0%)"

m1 = smf.ols(f"Log_RCR ~ C(HSS_Bin, Treatment('{ref_bin}')) + {controls_base}", data=df).fit()
m2 = smf.ols(f"Log_RCR ~ C(HSS_Bin, Treatment('{ref_bin}')) + {controls_fe}",  data=df).fit()
m3 = smf.ols(f"Log_RCR ~ HSS_Continuous + is_hss_label + {controls_fe}",        data=df).fit()

# ============================================================
# 3. DID 核心：文科基因 × Stage 交叉项
#    检验：跨界溢价在 GenAI 两个节点后是否增强？
# ============================================================
m4 = smf.ols(
    f"Log_RCR ~ HSS_Continuous * C(stage) + is_hss_label + {controls_fe}",
    data=df
).fit()

print("\n" + "="*80)
print("📊 DID 核心结果：文科基因 × GenAI 时代交叉项")
print("="*80)

# 提取关键交叉项系数
did_vars = [c for c in m4.params.index if 'HSS_Continuous:' in c or ':HSS_Continuous' in c]
print(m4.params[did_vars])
print(m4.pvalues[did_vars])

# ============================================================
# 4. 异质性分析：按学科分组跑子样本回归
# ============================================================
print("\n" + "="*80)
print("🔬 异质性分析：各学科中的文科基因溢价")
print("="*80)

top_fields = df['primary_field'].value_counts().head(8).index.tolist()
het_results = {}

for field in top_fields:
    sub = df[df['primary_field'] == field].copy()
    if len(sub) < 500:
        print(f"  ⏭️  {field} 样本不足（{len(sub)}），跳过")
        continue
    try:
        m_sub = smf.ols(
            f"Log_RCR ~ HSS_Continuous + is_hss_label + "
            f"HSS_Diversity + log_team_size + log_prior_knowledge + C(stage)",
            data=sub
        ).fit()
        het_results[field] = m_sub
        coef = m_sub.params['HSS_Continuous']
        pval = m_sub.pvalues['HSS_Continuous']
        star = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
        print(f"  {field:30s}  β={coef:+.4f}{star}  (N={len(sub):,})")
    except Exception as e:
        print(f"  ❌ {field} 报错：{e}")

# ============================================================
# 5. 主回归三线表输出
# ============================================================
regressor_order = [
    'HSS_Continuous',
    f"C(HSS_Bin, Treatment('{ref_bin}'))[T.2_Low_HSS(1-20%)]",
    f"C(HSS_Bin, Treatment('{ref_bin}'))[T.3_Moderate_HSS(21-40%)]",
    f"C(HSS_Bin, Treatment('{ref_bin}'))[T.4_High_HSS(41-60%)]",
    f"C(HSS_Bin, Treatment('{ref_bin}'))[T.5_HSS_Dominant(>60%)]",
    'is_hss_label',
    'HSS_Diversity',
    'log_team_size',
    'log_prior_knowledge',
]

table = summary_col(
    [m1, m2, m3],
    model_names=['M1(无学科FE)', 'M2(加学科FE)', 'M3(连续+学科FE)'],
    stars=True,
    float_format='%0.3f',
    regressor_order=regressor_order,
    drop_omitted=True,
    info_dict={'N': lambda x: f"{int(x.nobs):,}", 'R²': lambda x: f"{x.rsquared:.3f}"}
)
print("\n" + "="*80)
print(table)
print("*** p<0.01, ** p<0.05, * p<0.1")
