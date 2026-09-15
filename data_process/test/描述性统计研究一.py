import pandas as pd
import numpy as np

# ─── 0. 加载数据 ──────────────────────────────────────────────
df = pd.read_csv(
    r"E:\PythonProject\GenAI\data_process\test\Ultimate_Regression_Base_with_Diversity.csv"
)
print(f"原始样本量: {len(df):,}")

# ─── 1. 特征工程 ──────────────────────────────────────────────

# 1.1 截断 pct 至 [0, 100]
df['team_hss_dna_pct'] = df['team_hss_dna_pct'].clip(0, 100)
df['HSS_Continuous']   = df['team_hss_dna_pct'] / 10.0

# 1.2 HSS_Bin
bins   = [-0.1, 0.1, 20.0, 40.0, 60.0, 100.1]
labels = [
    '1_Pure_STEM(0%)',
    '2_Low_HSS(1-20%)',
    '3_Moderate_HSS(21-40%)',
    '4_High_HSS(41-60%)',
    '5_HSS_Dominant(>60%)'
]
df['HSS_Bin'] = pd.cut(df['team_hss_dna_pct'], bins=bins, labels=labels)

# 1.3 ✅ is_hss_label：Type 3 = 文科，其余 = 0
df['is_hss_label'] = df['arxiv_type'].apply(
    lambda x: 1 if 'Type 3' in str(x) else 0
)
print(f"arxiv_type 分布：")
print(df['arxiv_type'].value_counts())
print(f"\nis_hss_label=1（Type 3 文科）: {df['is_hss_label'].sum():,} 篇"
      f"（占比 {df['is_hss_label'].mean():.4f}）\n")

# 1.4 Log 控制变量
df['log_team_size']       = np.log1p(df['total_authors'])
df['log_prior_knowledge'] = np.log1p(df['Prior_Knowledge_Stock'])

# 1.5 Log_RCR 99百分位缩尾
p99 = df['Log_RCR'].quantile(0.99)
df['Log_RCR_wins'] = df['Log_RCR'].clip(upper=p99)
print(f"Log_RCR 99th pct 缩尾上限: {p99:.4f}")

# 1.6 ✅ Hit_Rate_10：按 year × arxiv_primary_category 分组，严格大于 p90
df['RCR'] = pd.to_numeric(df['RCR'], errors='coerce')
df['_p90'] = df.groupby(['year', 'arxiv_primary_category'])['RCR'] \
    .transform(lambda x: x.quantile(0.90))
df['Hit_Rate_10'] = (df['RCR'] > df['_p90']).astype(int)

hit_n   = df['Hit_Rate_10'].sum()
hit_pct = hit_n / len(df) * 100
print(f"Hit Rate: {int(hit_n):,} / {len(df):,}（{hit_pct:.2f}%）— 预期 6-8%\n")

# ─── 2. 核心描述性统计 ────────────────────────────────────────

core_vars = {
    'Log_RCR_wins':       'Log RCR (winsorized, Dep. Var.)',
    'Hit_Rate_10':        'Hit Rate Top-10% (Dummy)',
    'HSS_Continuous':     'HSS Continuous (per 10 pp)',
    'is_hss_label':       'is_hss_label (Type 3, HSS)',
    'HSS_Diversity':      'HSS Diversity',
    'log_team_size':      'Log(Team Size)',
    'log_prior_knowledge':'Log(Prior Knowledge Stock)',
}

rows = []
for col, label in core_vars.items():
    s = df[col].dropna()
    rows.append({
        'Variable':  label,
        'N':         f'{len(s):,}',
        'Mean':      f'{s.mean():.4f}',
        'Std. Dev.': f'{s.std():.4f}',
        'Min':       f'{s.min():.4f}',
        'Median':    f'{s.median():.4f}',
        'Max':       f'{s.max():.4f}',
    })

desc_df = pd.DataFrame(rows)
print('===== 核心变量描述性统计 =====')
print(desc_df.to_string(index=False))

# ─── 3. HSS_Bin 频数分布 ──────────────────────────────────────

bin_order = [
    '1_Pure_STEM(0%)',
    '2_Low_HSS(1-20%)',
    '3_Moderate_HSS(21-40%)',
    '4_High_HSS(41-60%)',
    '5_HSS_Dominant(>60%)'
]
bin_labels_display = {
    '1_Pure_STEM(0%)':          'Pure STEM (0%)',
    '2_Low_HSS(1-20%)':         'Low HSS (1-20%)',
    '3_Moderate_HSS(21-40%)':   'Moderate HSS (21-40%)',
    '4_High_HSS(41-60%)':       'High HSS (41-60%)',
    '5_HSS_Dominant(>60%)':     'HSS Dominant (>60%)',
}
bin_counts = df['HSS_Bin'].value_counts()
total = len(df)

print('\n===== HSS_Bin 频数分布 =====')
for b in bin_order:
    n = bin_counts.get(b, 0)
    print(f'  {bin_labels_display[b]:<28}: N = {n:>7,}  ({n/total*100:.2f}%)')

# ─── 4. 分阶段统计 ────────────────────────────────────────────

print('\n===== 各技术阶段样本量及 Log_RCR (winsorized) 均值 =====')
phase_stats = (
    df.groupby('stage')['Log_RCR_wins']
    .agg(N='count', Mean='mean', Std='std')
    .reset_index()
)
print(phase_stats.to_string(index=False))

# ─── 5. 导出 LaTeX 表格行 ─────────────────────────────────────

print('\n===== LaTeX 表格行（粘贴至 tab:desc_study1）=====')
print('\n% --- 连续变量 ---')
for col, label in core_vars.items():
    s = df[col].dropna()
    print(
        f'    {label:<42} & {len(s):>7,} '
        f'& {s.mean():.4f} & {s.std():.4f} '
        f'& {s.min():.4f} & {s.max():.4f} \\\\'
    )

print('\n% --- HSS Bin 频数分布 ---')
for b in bin_order:
    n = bin_counts.get(b, 0)
    print(
        f'    \\hspace{{0.4cm}} {bin_labels_display[b]:<28} '
        f'& {n:>7,} ({n/total*100:.1f}\\%) '
        f'& --- & --- & --- & --- \\\\'
    )

print(f'\n% 99th pct 缩尾上限   : {p99:.4f}')
print(f'% 爆款论文数          : {int(hit_n):,} / {len(df):,} ({hit_pct:.2f}%)')
print(f'% is_hss_label=1 比例 : {df["is_hss_label"].mean():.4f}')
