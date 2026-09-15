import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from linearmodels.panel import PanelOLS
import statsmodels.api as sm
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def get_dynamic_plot_data(df_sub, window_min=-4, window_max=4):
    """为子样本跑动态面板，返回画图所需的点位数据"""
    df_sub = df_sub.copy()
    df_sub['rel_year_clipped'] = df_sub['relative_year'].clip(lower=window_min, upper=window_max)

    dummies = pd.get_dummies(df_sub['rel_year_clipped'], prefix='T')

    interaction_cols = []
    for col in dummies.columns:
        inter_name = f"DID_{col}"
        df_sub[inter_name] = df_sub['is_treated'] * dummies[col]
        interaction_cols.append(inter_name)

    baseline_col = 'DID_T_-1' if 'DID_T_-1' in interaction_cols else 'DID_T_-1.0'
    if baseline_col in interaction_cols:
        interaction_cols.remove(baseline_col)

    df_sub = df_sub.set_index(['author_id', 'year'])
    Y = df_sub['avg_logRCR']
    X = df_sub[interaction_cols]
    X = sm.add_constant(X)

    model = PanelOLS(Y, X, entity_effects=True, time_effects=True)
    results = model.fit(cov_type='clustered', cluster_entity=True)

    coefs = results.params.drop('const')
    conf_ints = results.conf_int().drop('const')

    coefs[baseline_col] = 0.0
    conf_ints.loc[baseline_col] = [0.0, 0.0]

    plot_data = pd.DataFrame({'coef': coefs, 'lower_ci': conf_ints['lower'], 'upper_ci': conf_ints['upper']})
    plot_data['time'] = [float(str(idx).replace('DID_T_', '')) for idx in plot_data.index]
    return plot_data.dropna(subset=['time']).sort_values('time')


def run_heterogeneous_event_study():
    print("⏳ 正在加载数据并构建精英标签...")
    df = pd.read_csv('Perfect_DID_Panel.csv')
    df['is_elite_overall'] = (
                df['is_industry_elite'].fillna(0).astype(int) | df['is_academic_elite'].fillna(0).astype(int))

    print("🚀 正在分别计算【精英组】和【普通组】的动态轨迹...")
    df_elite = df[df['is_elite_overall'] == 1]
    df_normal = df[df['is_elite_overall'] == 0]

    plot_elite = get_dynamic_plot_data(df_elite)
    plot_normal = get_dynamic_plot_data(df_normal)

    # ================= 开始画双线对比图 =================
    plt.figure(figsize=(11, 6), dpi=150)

    # 画精英组 (红色, 稍微错开一点防止遮挡)
    plt.errorbar(x=plot_elite['time'] - 0.05, y=plot_elite['coef'],
                 yerr=[plot_elite['coef'] - plot_elite['lower_ci'], plot_elite['upper_ci'] - plot_elite['coef']],
                 fmt='-o', color='crimson', label='精英组 (QS100 / 巨头)', capsize=4, elinewidth=2, markersize=7)

    # 画普通组 (蓝色)
    plt.errorbar(x=plot_normal['time'] + 0.05, y=plot_normal['coef'],
                 yerr=[plot_normal['coef'] - plot_normal['lower_ci'], plot_normal['upper_ci'] - plot_normal['coef']],
                 fmt='-s', color='steelblue', label='普通组', capsize=4, elinewidth=2, markersize=7, alpha=0.8)

    plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
    plt.axvline(x=-0.5, color='gray', linestyle='--', linewidth=1.5)

    plt.title('跨学科合作溢价的阶层分化：精英 vs 普通 (Dynamic DID)', fontsize=16, pad=15)
    plt.xlabel('距离首次跨界合作的相对年份', fontsize=12)
    plt.ylabel('对平均 logRCR 的净影响', fontsize=12)
    plt.xticks(plot_elite['time'])
    plt.legend(fontsize=11, loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig('Heterogeneity_Event_Study.png')
    print("🎉 双线对比图已保存为：Heterogeneity_Event_Study.png")
    plt.show()


if __name__ == "__main__":
    run_heterogeneous_event_study()