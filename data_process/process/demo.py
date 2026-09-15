import pandas as pd
import numpy as np
import json
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors


def run_hardcore_arxiv_psm():
    print("🚀 启动【arXiv 内部精英匹配】引擎：寻找最纯净的学术双胞胎...\n")

    # ==========================================
    # 1. 锁定 arXiv 活跃作者池
    # ==========================================
    print("📊 1. 正在从 OLS 底表锁定 arXiv 活跃作者...")
    df_ultimate = pd.read_csv(r'E:\PythonProject\GenAI\data_process\test\Ultimate_Regression_Base.csv')

    # 加载作者画像，确定“纯血理科”身份
    df_profiles = pd.read_csv('MASTER_AUTHOR_PROFILES_12YEARS.csv')
    hss_fields = {'Social Sciences', 'Arts and Humanities', 'Psychology',
                  'Business, Management and Accounting', 'Economics, Econometrics and Finance', 'Decision Sciences'}

    # 提取在 arXiv 发过文章的作者，并判断其 DNA
    # 我们只看那些主领域是理科，且历史 DNA 纯度 100% 的人
    pure_stem_pool = set()
    hss_sources = set()
    for _, row in df_profiles.iterrows():
        aid = row['author_id']
        primary = str(row['primary_origin_field']).strip()
        if primary in hss_fields:
            hss_sources.add(aid)
        elif primary in {'Computer Science', 'Mathematics', 'Physics and Astronomy', 'Engineering'}:
            # 严格检查 DNA 纯度
            try:
                dna = json.loads(row['scopus_dna_weights'].replace("'", '"'))
                if not any(f in hss_fields for f in dna.keys()):
                    pure_stem_pool.add(aid)
            except:
                continue

    print(f"✅ 锁定 arXiv 候选小白鼠: {len(pure_stem_pool):,} 人")

    # ==========================================
    # 2. 遍历 JSONL，提取这批人在 arXiv 内的轨迹
    # ==========================================
    print("📖 2. 正在提取 arXiv 小白鼠的跨界碰撞时刻 (T0)...")
    # 构建极速查询：只查这批小白鼠的论文
    arxiv_wids = set(df_ultimate['clean_work_id'].unique())
    author_stats = {}  # {aid: {year: {papers: 0, hss: 0, rcr_list: []}}}

    jsonl_files = [
        r'E:\PythonProject\GenAI\data_scrap_v2\stage_1\phase1_hybrid_cleaned.jsonl',
        r'E:\PythonProject\GenAI\data_scrap_v2\stage_2\phase2_hybrid_cleaned.jsonl',
        r'E:\PythonProject\GenAI\data_scrap_v2\stage_3\phase3_hybrid_cleaned.jsonl'
    ]

    # 为了匹配精准，我们要把 RCR 挂载上来
    rcr_lookup = dict(zip(df_ultimate['clean_work_id'], df_ultimate['Log_RCR']))

    for file_path in jsonl_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                paper = json.loads(line)
                wid = paper.get('work_id').split('/')[-1]
                if wid not in arxiv_wids: continue

                year = paper.get('publication_year')
                authors = paper.get('author_ids', [])
                has_hss = any(a in hss_sources for a in authors)

                for a in authors:
                    if a in pure_stem_pool:
                        if a not in author_stats: author_stats[a] = {}
                        if year not in author_stats[a]: author_stats[a][year] = {'p': 0, 'h': 0, 'r': []}
                        author_stats[a][year]['p'] += 1
                        if has_hss: author_stats[a][year]['h'] += 1
                        author_stats[a][year]['r'].append(rcr_lookup.get(wid, 0))

    # ==========================================
    # 3. PSM 匹配：在 arXiv 内部找双胞胎
    # ==========================================
    print("👯 3. 正在 arXiv 内部进行 1:1 精准 PSM 匹配...")
    rows = []
    for aid, years in author_stats.items():
        t0 = next((y for y in sorted(years.keys()) if years[y]['h'] > 0), None)
        rows.append({
            'author_id': aid,
            'is_treated': 1 if t0 else 0,
            't0': t0 if t0 else 0,
            'total_p': sum(y['p'] for y in years.values()),
            'avg_r': np.mean([r for y in years.values() for r in y['r']])
        })

    df_psm = pd.DataFrame(rows)
    # 逻辑回归算得分
    X = df_psm[['total_p', 'avg_r']]
    y = df_psm['is_treated']
    ps_model = LogisticRegression().fit(X, y)
    df_psm['pscore'] = ps_model.predict_proba(X)[:, 1]

    treated = df_psm[df_psm['is_treated'] == 1]
    control = df_psm[df_psm['is_treated'] == 0]

    # 1:1 匹配
    nn = NearestNeighbors(n_neighbors=1).fit(control[['pscore']])
    dist, idx = nn.kneighbors(treated[['pscore']])
    matched_control = control.iloc[idx.flatten()].copy()
    matched_control['pseudo_t0'] = treated['t0'].values

    # ==========================================
    # 4. 生成平衡后的最终面板
    # ==========================================
    print("✨ 4. 正在生成平衡后的 arXiv 专属 DID 面板...")
    final_dict = {row['author_id']: row['t0'] for _, row in treated.iterrows()}
    final_dict.update({row['author_id']: row['pseudo_t0'] for _, row in matched_control.iterrows()})

    final_panel = []
    for aid, t0 in final_dict.items():
        for year, stats in author_stats[aid].items():
            rel = year - t0
            if -3 <= rel <= 4:
                final_panel.append({
                    'author_id': aid, 'year': year, 'is_treated': 1 if aid in treated['author_id'].values else 0,
                    'relative_time': rel, 'Post': 1 if rel >= 0 else 0, 'mean_log_rcr': np.mean(stats['r'])
                })

    df_final = pd.DataFrame(final_panel)
    df_final.to_csv('arXiv_Balanced_DID_Panel.csv', index=False)
    print(f"✅ 匹配完成！最终保留了 {len(treated):,} 对 arXiv 学术双胞胎。")
    print(f"💾 文件已保存：arXiv_Balanced_DID_Panel.csv")


run_hardcore_arxiv_psm()