import json
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# ================= 配置区 =================
INPUT_RAW_JSONL = r'Author_Global_Metadata_90k.jsonl'
OUTPUT_FINAL_PANEL = r'Final_Strict_Matched_Panel.csv'

# 全局宏观冲击年份 (GenAI 爆发并深刻影响学术界的年份)
EVENT_YEAR = 2023
# 基准期自动前推三年 (T-3 到 T-1)
BASELINE_YEARS = [2020, 2021, 2022]


# ==========================================

def run_strict_genai_psm():
    print("1. ⏳ 正在加载全量原始数据...")
    records = []
    with open(INPUT_RAW_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))

    df_raw = pd.DataFrame(records)
    df_raw['first_pub_year'] = pd.to_numeric(df_raw['first_pub_year'], errors='coerce')

    print(f"2. 🧮 正在计算统一基准期 {BASELINE_YEARS} 的特征...")
    df_raw['academic_age_at_shock'] = EVENT_YEAR - df_raw['first_pub_year']

    pre_works, pre_cites = [], []
    for counts in df_raw['counts_by_year']:
        w, c = 0, 0
        if isinstance(counts, list):
            for yr_data in counts:
                if yr_data.get('year') in BASELINE_YEARS:
                    w += yr_data.get('works_count', 0)
                    c += yr_data.get('cited_by_count', 0)
        pre_works.append(w)
        pre_cites.append(c)

    df_raw['pre_total_works'] = pre_works
    df_raw['pre_total_citations'] = pre_cites

    # 清理缺失值
    covariates = ['academic_age_at_shock', 'pre_total_works', 'pre_total_citations', 'h_index']
    df_clean = df_raw.dropna(subset=covariates + ['is_treated']).copy()

    print("3. 🤝 正在计算倾向得分并执行【严格卡尺匹配】...")
    X = df_clean[covariates]
    y = df_clean['is_treated'].values
    X_scaled = StandardScaler().fit_transform(X)

    logit = LogisticRegression(solver='liblinear')
    logit.fit(X_scaled, y)
    df_clean['propensity_score'] = logit.predict_proba(X_scaled)[:, 1]

    treat_clean = df_clean[df_clean['is_treated'] == 1].copy()
    ctrl_clean = df_clean[df_clean['is_treated'] == 0].copy()

    # --- 铁闸门 1：共同支撑 (Common Support) ---
    min_ps = max(treat_clean['propensity_score'].min(), ctrl_clean['propensity_score'].min())
    max_ps = min(treat_clean['propensity_score'].max(), ctrl_clean['propensity_score'].max())

    treat_clean = treat_clean[(treat_clean['propensity_score'] >= min_ps) & (treat_clean['propensity_score'] <= max_ps)]
    ctrl_clean = ctrl_clean[(ctrl_clean['propensity_score'] >= min_ps) & (ctrl_clean['propensity_score'] <= max_ps)]

    # --- 铁闸门 2：卡尺限制 (Caliper) ---
    # 业内标准：倾向得分标准差的 0.2 倍
    caliper = df_clean['propensity_score'].std() * 0.2
    print(f"📏 已设置匹配卡尺限制: 最大得分差距不能超过 {caliper:.5f}")

    # 使用处理过后的子集重新拟合树
    nn = NearestNeighbors(n_neighbors=1, metric='euclidean', algorithm='kd_tree')
    nn.fit(ctrl_clean[['propensity_score']])
    distances, indices = nn.kneighbors(treat_clean[['propensity_score']])

    # 核心过滤：距离大于卡尺的一律丢弃
    valid_matches = distances.flatten() <= caliper

    matched_treat = treat_clean[valid_matches]
    matched_ctrl_indices = indices.flatten()[valid_matches]
    matched_ctrl = ctrl_clean.iloc[matched_ctrl_indices].copy()

    final_matched_authors = pd.concat([matched_treat, matched_ctrl]).drop_duplicates(subset=['author_id'])

    print(f"✅ 严格匹配成功！淘汰劣质替身后，保留了高品质样本: {len(final_matched_authors)} 人。")

    print("4. 🔄 正在展开为长面板 (Long Panel)...")
    panel_records = []
    for row in final_matched_authors.itertuples(index=False):
        if isinstance(row.counts_by_year, list):
            for count_data in row.counts_by_year:
                calendar_year = count_data.get('year')
                # 保留 2014 至今足够画平行趋势即可
                if calendar_year and calendar_year >= 2014:
                    panel_records.append({
                        'author_id': row.author_id,
                        'is_treated': row.is_treated,
                        'calendar_year': calendar_year,
                        'relative_year': calendar_year - EVENT_YEAR,  # 以 2023 为 0 点
                        'works_count': count_data.get('works_count', 0),
                        'cited_by_count': count_data.get('cited_by_count', 0)
                    })

    df_panel = pd.DataFrame(panel_records)
    df_panel.to_csv(OUTPUT_FINAL_PANEL, index=False, encoding='utf-8-sig')
    print(f"🎉 严格版 GenAI 专属面板生成完毕！共 {len(df_panel)} 条记录，已保存至: {OUTPUT_FINAL_PANEL}")


if __name__ == "__main__":
    run_strict_genai_psm()