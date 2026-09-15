import pandas as pd
import statsmodels.formula.api as smf
import warnings

warnings.filterwarnings('ignore')


def run_traditional_did():
    print("⏳ 正在加载完美匹配面板数据...")
    df = pd.read_csv('Perfect_DID_Panel.csv')
    df['is_elite_overall'] = (
                df['is_industry_elite'].fillna(0).astype(int) | df['is_academic_elite'].fillna(0).astype(int))

    # 提取我们最关心的子样本：普通机构组
    df_normal = df[df['is_elite_overall'] == 0].copy()

    # 1. 构造传统的 Post 变量 (基于相对年份)
    df_normal['Post'] = (df_normal['relative_year'] >= 0).astype(int)

    # 2. 构造传统的 Treated 变量 (其实就是 is_treated)
    df_normal['Treated'] = df_normal['is_treated']

    print("🚀 正在运行教科书版传统古典 DID (Pooled OLS)...")

    # 使用 R 语言风格的公式，Treated:Post 自动生成交互项
    # 我们依然要在作者层面聚类标准误，防止同一个人的误差自相关
    model = smf.ols('avg_logRCR ~ Treated + Post + Treated:Post', data=df_normal)
    results = model.fit(cov_type='cluster', cov_kwds={'groups': df_normal['author_id']})

    print("\n" + "=" * 60)
    print("🎓 教科书版传统 DID 回归结果 (普通组子样本)")
    print("=" * 60)
    print(results.summary())


if __name__ == "__main__":
    run_traditional_did()