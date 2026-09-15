import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
import warnings

warnings.filterwarnings('ignore')


def run_perfect_subsample_did(df, condition, group_name):
    """在完美匹配面板上跑子样本静态 DID"""
    sub_df = df[condition].copy()
    sub_df = sub_df.set_index(['author_id', 'year'])

    Y = sub_df['avg_logRCR']

    # 只要是在跨界后（relative_year >= 0），Post 就为 1
    # 因为已经是完美的处理组和对照组面板了，直接把 is_treated * Post 作为核心变量
    sub_df['Post'] = (sub_df['relative_year'] >= 0).astype(int)
    sub_df['DID_term'] = sub_df['is_treated'] * sub_df['Post']

    X = sub_df[['DID_term']]
    X = sm.add_constant(X)

    model = PanelOLS(Y, X, entity_effects=True, time_effects=True)
    results = model.fit(cov_type='clustered', cluster_entity=True)

    coef = results.params['DID_term']
    pval = results.pvalues['DID_term']

    print(f"📊 {group_name}")
    print(f"   ▶ DID 系数: {coef:.4f}")
    print(f"   ▶ P-value : {pval:.4f}")
    if pval < 0.05:
        print("   => 🌟 显著！有真实的平均跨界红利！")
    elif pval < 0.1:
        print("   => ⭐ 边缘显著。")
    else:
        print("   => ❌ 不显著（或者被衰退期平均掉了）。")
    print("-" * 50)


def main():
    print("⏳ 正在加载完美匹配面板数据 (Perfect Panel)...")
    df = pd.read_csv('Perfect_DID_Panel.csv')
    df['is_elite_overall'] = (
                df['is_industry_elite'].fillna(0).astype(int) | df['is_academic_elite'].fillna(0).astype(int))

    print("\n=======================================================")
    print(" 🏆 终极静态 DID 对决：谁才是跨学科红利的真正赢家？")
    print("=======================================================")

    run_perfect_subsample_did(df, df['is_elite_overall'] == 1, "🔴 精英组 (QS100 + 工业巨头)")
    run_perfect_subsample_did(df, df['is_elite_overall'] == 0, "🔵 普通组 (非顶尖机构)")


if __name__ == "__main__":
    main()