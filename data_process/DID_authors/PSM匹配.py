import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# ================= 配置区 =================
INPUT_RAW_JSONL = r'Author_Global_Metadata_90k.jsonl'
OUTPUT_MATCHED_LIST = r'Matched_Author_IDs_Ultra_Strict.csv'

# 【严格控制参数】
CALIPER_MULTIPLIER = 0.05  # 极严卡尺 (常规为 0.2)
MAX_AGE_DIFF = 1  # 学术年龄差距最大允许 1 年


# ==========================================

def calculate_pre_features(counts_by_year, target_T):
    pre_works, pre_cites = 0, 0
    if isinstance(counts_by_year, list):
        for yr_data in counts_by_year:
            if target_T - 3 <= yr_data.get('year') <= target_T - 1:
                pre_works += yr_data.get('works_count', 0)
                pre_cites += yr_data.get('cited_by_count', 0)
    return pre_works, pre_cites


def run_ultra_strict_psm():
    print("1. ⏳ 正在加载 API 提取的全局特征数据...")
    records = []
    with open(INPUT_RAW_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))

    df_raw = pd.DataFrame(records)

    treated_pool = df_raw[df_raw['is_treated'] == 1].copy()
    control_pool = df_raw[df_raw['is_treated'] == 0].copy()

    cohort_years = sorted(treated_pool['event_year'].dropna().unique())
    print(f"📊 发现处理组存在 {len(cohort_years)} 个队列。准备执行【极严】匹配...")

    matched_results = []

    for T in cohort_years:
        T = int(T)
        print(f"\n   ▶ 处理 {T} 年跨界队列...")

        treat_c = treated_pool[treated_pool['event_year'] == T].copy()
        ctrl_c = control_pool.copy()
        ctrl_c['pseudo_event_year'] = T
        treat_c['pseudo_event_year'] = T

        df_cohort = pd.concat([treat_c, ctrl_c], ignore_index=True)

        # --- 计算协变量 ---
        df_cohort['academic_age_at_T'] = T - pd.to_numeric(df_cohort['first_pub_year'], errors='coerce')

        features = df_cohort['counts_by_year'].apply(lambda x: calculate_pre_features(x, T))
        df_cohort['pre_total_works'] = [f[0] for f in features]
        df_cohort['pre_total_citations'] = [f[1] for f in features]

        covariates = ['academic_age_at_T', 'pre_total_works', 'pre_total_citations', 'h_index']
        df_cohort = df_cohort.dropna(subset=covariates)

        # --- 计算倾向得分 ---
        X = df_cohort[covariates]
        y = df_cohort['is_treated'].values

        # 确保队列内有足够的样本跑回归
        if len(set(y)) < 2:
            print(f"     ⚠️ 队列 {T} 缺乏对照/处理样本，跳过。")
            continue

        X_scaled = StandardScaler().fit_transform(X)
        logit = LogisticRegression(solver='liblinear')
        logit.fit(X_scaled, y)
        df_cohort['propensity_score'] = logit.predict_proba(X_scaled)[:, 1]

        # 提取清理后的数据池并保留原始 DataFrame 的 Index 供追踪
        t_clean = df_cohort[df_cohort['is_treated'] == 1]
        c_clean = df_cohort[df_cohort['is_treated'] == 0]

        # --- 铁闸 1：计算极严卡尺 ---
        caliper = df_cohort['propensity_score'].std() * CALIPER_MULTIPLIER

        # --- 最近邻计算 ---
        nn = NearestNeighbors(n_neighbors=5, metric='euclidean', algorithm='kd_tree')  # 先找 5 个备胎
        nn.fit(c_clean[['propensity_score']])
        distances, indices = nn.kneighbors(t_clean[['propensity_score']])

        # --- 铁闸 2 & 3：筛选配对池，强制无放回 + 年龄限制 ---
        pair_records = []
        for i, t_idx in enumerate(t_clean.index):
            treat_age = t_clean.loc[t_idx, 'academic_age_at_T']

            for j in range(5):  # 遍历 5 个备选控制组
                c_idx = c_clean.index[indices[i, j]]
                dist = distances[i, j]
                ctrl_age = c_clean.loc[c_idx, 'academic_age_at_T']

                # 如果距离小于卡尺，且学术年龄差距小于等于 MAX_AGE_DIFF
                if dist <= caliper and abs(treat_age - ctrl_age) <= MAX_AGE_DIFF:
                    pair_records.append({
                        'treat_idx': t_idx,
                        'ctrl_idx': c_idx,
                        'distance': dist
                    })

        if not pair_records:
            print(f"     ❌ {T} 年队列匹配失败：无符合极严标准的样本。")
            continue

        pairs_df = pd.DataFrame(pair_records)

        # 贪心去重逻辑：按距离从小到大排序
        pairs_df = pairs_df.sort_values('distance')

        # 确保一个控制组只能用一次（无放回）
        pairs_df = pairs_df.drop_duplicates(subset='ctrl_idx', keep='first')
        # 确保一个处理组只能匹配一个控制组
        pairs_df = pairs_df.drop_duplicates(subset='treat_idx', keep='first')

        # 提取最终存活的对子
        matched_treat = t_clean.loc[pairs_df['treat_idx']].copy()
        matched_ctrl = c_clean.loc[pairs_df['ctrl_idx']].copy()

        matched_results.append(matched_treat)
        matched_results.append(matched_ctrl)
        print(f"     ✅ 极严匹配完成！经过残酷淘汰，成功配对 {len(matched_treat)} 对高质量样本。")

    if not matched_results:
        print("\n❌ 匹配条件过于苛刻，未能匹配到任何有效样本，请适当放宽条件！")
        return

    # 3. 汇总输出
    final_matched_df = pd.concat(matched_results, ignore_index=True)
    output_df = final_matched_df[['author_id', 'is_treated', 'pseudo_event_year', 'display_name']]
    output_df.rename(columns={'pseudo_event_year': 'event_year_aligned'}, inplace=True)

    output_df.to_csv(OUTPUT_MATCHED_LIST, index=False, encoding='utf-8-sig')

    print(f"\n🎉 大功告成！超严标准下的精英名单已生成至: {OUTPUT_MATCHED_LIST}")
    print(f"👉 存活名单共包含 {len(output_df)} 人。这批数据扔进回归，因果效力极强！")


if __name__ == "__main__":
    run_ultra_strict_psm()