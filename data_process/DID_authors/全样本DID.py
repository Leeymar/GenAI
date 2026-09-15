import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
import warnings

warnings.filterwarnings('ignore')


def run_perfect_baseline_twfe():
    print("⏳ 正在加载完美匹配面板数据 (Perfect Panel)...")
    df = pd.read_csv('Perfect_DID_Panel.csv')

    # 设置面板多重索引
    df = df.set_index(['author_id', 'year'])

    # 生成标准的 Post 和 DID 交互项
    df['Post'] = (df['relative_year'] >= 0).astype(int)
    df['DID_term'] = df['is_treated'] * df['Post']

    Y = df['avg_logRCR']
    X = df[['DID_term']]
    X = sm.add_constant(X)

    print("🚀 正在运行全样本双向固定效应基准回归 (Baseline TWFE)...")
    model = PanelOLS(Y, X, entity_effects=True, time_effects=True)

    # 依然使用聚类稳健标准误
    results = model.fit(cov_type='clustered', cluster_entity=True)

    print("\n" + "=" * 60)
    print("🏆 论文表1：全样本基准回归结果 (Baseline TWFE DID)")
    print("=" * 60)
    print(results.summary)


if __name__ == "__main__":
    run_perfect_baseline_twfe()