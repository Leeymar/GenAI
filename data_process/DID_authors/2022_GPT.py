import pandas as pd
import statsmodels.formula.api as smf
import warnings

warnings.filterwarnings('ignore')


def run_gpt_traditional_did():
    print("⏳ 正在打捞 GPT 浪潮下的‘定点实验组’与‘守旧对照组’...")
    df = pd.read_csv('Perfect_DID_Panel.csv')

    # 1. 识别：在 2023 或 2024 年‘刚刚跨界’的人 (relative_year=0)
    gpt_treated_ids = df[(df['relative_year'] == 0) & (df['year'] >= 2023)]['author_id'].unique()

    # 2. 识别：PSM 匹配出来的、自始至终没跨界的对照组
    control_ids = df[df['is_treated'] == 0]['author_id'].unique()

    # 3. 提取这两组人，并只保留 2021-2024 的数据（为了让对比更聚焦在冲击前后）
    df_lab = df[df['author_id'].isin(list(gpt_treated_ids) + list(control_ids))].copy()
    df_lab = df_lab[df_lab['year'] >= 2021]

    # 4. 构造经典 DID 变量
    df_lab['Post'] = (df_lab['year'] >= 2023).astype(int)
    df_lab['Treated'] = df_lab['author_id'].isin(gpt_treated_ids).astype(int)

    print(f"📊 实验样本规模：")
    print(f"   - 跨界‘变阵者’：{len(gpt_treated_ids)} 人")
    print(f"   - 硬核‘守旧者’：{len(control_ids)} 人")

    # 5. 运行传统 OLS 回归（带聚类稳健标准误）
    # 公式：影响力 ~ 是否跨界 + 是否在2022后 + 交互项
    model = smf.ols('avg_logRCR ~ Treated + Post + Treated:Post', data=df_lab)
    results = model.fit(cov_type='cluster', cov_kwds={'groups': df_lab['author_id']})

    print("\n" + "=" * 60)
    print("🤖 GPT 冲击定点实验：传统 DID 回归结果")
    print("=" * 60)
    print(results.summary())


if __name__ == "__main__":
    run_gpt_traditional_did()