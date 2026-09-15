import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


def run_trajectory_psm():
    print("⏳ 1. 正在加载面板数据...")
    df_panel = pd.read_csv('Ultimate_DID_Panel.csv')

    # 动态构建精英标签 (防止之前漏掉)
    df_panel['is_elite_overall'] = (
                df_panel['is_industry_elite'].fillna(0).astype(int) | df_panel['is_academic_elite'].fillna(0).astype(
            int))

    print("✂️ 2. 正在提取跨界前 (T=-3, -2, -1) 的动态轨迹特征...")
    # 只取跨界前三年的数据
    df_pre = df_panel[df_panel['relative_year'].isin([-3, -2, -1])]

    # 将长面板变成宽面板 (每人一行，列是各年的 logRCR)
    df_pivot = df_pre.pivot_table(index='author_id', columns='relative_year', values='avg_logRCR',
                                  fill_value=0).reset_index()
    df_pivot.columns = ['author_id', 'pre_3', 'pre_2', 'pre_1']

    # 提取静态属性 (处理组标签、精英标签)
    df_static = df_panel[['author_id', 'is_treated', 'is_elite_overall']].drop_duplicates(subset=['author_id'])

    # 合并成专门用于 PSM 的特征矩阵
    df_psm = pd.merge(df_static, df_pivot, on='author_id', how='left')

    # 如果有人前三年根本没发论文，补 0
    df_psm[['pre_3', 'pre_2', 'pre_1']] = df_psm[['pre_3', 'pre_2', 'pre_1']].fillna(0)

    print(f"   => 提取完毕，准备为 {len(df_psm):,} 位作者计算倾向得分。")

    print("🧠 3. 正在运行 Logistic 倾向得分模型...")
    # 我们的自变量 (X) 现在包含了：精英光环 + 连续三年的精确影响力
    X = df_psm[['is_elite_overall', 'pre_3', 'pre_2', 'pre_1']]
    y = df_psm['is_treated']

    # 标准化特征 (提升逻辑回归收敛速度)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 拟合逻辑回归
    log_reg = LogisticRegression(solver='lbfgs', random_state=42)
    log_reg.fit(X_scaled, y)

    # 计算倾向得分 (Propensity Score)
    df_psm['pscore'] = log_reg.predict_proba(X_scaled)[:, 1]

    print("🤝 4. 正在执行 1:1 最近邻卡尺匹配 (Caliper Matching)...")
    # 分离处理组和对照组
    treated = df_psm[df_psm['is_treated'] == 1].reset_index(drop=True)
    control = df_psm[df_psm['is_treated'] == 0].reset_index(drop=True)

    # 设置极其严苛的卡尺（Caliper）: 0.01
    # 意味着只有 pscore 差距在 1% 以内的人才能牵手，宁缺毋滥！
    caliper = 0.01

    # 使用 KD-Tree 算法在对照组中找邻居
    nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')
    nn.fit(control[['pscore']].values)

    distances, indices = nn.kneighbors(treated[['pscore']].values)

    matched_treated_ids = []
    matched_control_ids = []

    # 遍历检查是否满足卡尺限制
    for i in range(len(treated)):
        dist = distances[i][0]
        if dist <= caliper:
            matched_treated_ids.append(treated.loc[i, 'author_id'])
            matched_control_ids.append(control.loc[indices[i][0], 'author_id'])

    matched_all_ids = set(matched_treated_ids + matched_control_ids)

    print(f"   => 匹配完成！")
    print(f"   => 原始处理组: {len(treated):,} 人")
    print(
        f"   => 成功牵手且符合轨迹卡尺的处理组: {len(matched_treated_ids):,} 人 (留存率: {len(matched_treated_ids) / len(treated) * 100:.1f}%)")
    print(f"   => 最终保留的作者总数 (1:1): {len(matched_all_ids):,} 人")

    print("💾 5. 正在生成全新的完美面板数据...")
    # 把这些幸存的“天选之子”从原面板里捞出来
    df_perfect_panel = df_panel[df_panel['author_id'].isin(matched_all_ids)]

    output_file = 'Perfect_DID_Panel.csv'
    df_perfect_panel.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"🎉 大功告成！基于轨迹匹配的无暇面板已保存至：{output_file}")

    print("\n💡 下一步行动提示：")
    print("请使用这份新生成的 'Perfect_DID_Panel.csv'，去跑之前的【动态平行趋势图】和【异质性分析】！")


if __name__ == "__main__":
    run_trajectory_psm()