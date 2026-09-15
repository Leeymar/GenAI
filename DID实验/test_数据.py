import json
import ast
import os
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
# ─────────────────────────────────────────────────────────────────
# Step 4b: 活跃度筛选 + PSM 倾向得分匹配
# ─────────────────────────────────────────────────────────────────
from sklearn.neighbors import NearestNeighbors  # ← 在文件顶部 import 区补上这一行
# ─────────────────────────────────────────────────────────────────
# 全局配置
# ─────────────────────────────────────────────────────────────────
HSS_FIELDS = {
    'Social Sciences',
    'Arts and Humanities',
    'Psychology',
    'Business, Management and Accounting',
    'Economics, Econometrics and Finance',
    'Decision Sciences',
}

INPUT_JSONL_FILES = [
    r'E:\PythonProject\GenAI\data_scrap_v2\stage_1\phase1_hybrid_cleaned.jsonl',
    r'E:\PythonProject\GenAI\data_scrap_v2\stage_2\phase2_hybrid_cleaned.jsonl',
    r'E:\PythonProject\GenAI\data_scrap_v2\stage_3\phase3_hybrid_cleaned.jsonl',
]
AUTHOR_CSV_PATH   = 'MASTER_AUTHOR_PROFILES_12YEARS.csv'
CITATION_MAP_PATH = 'citation_mapping.json'
OUTPUT_PANEL_PATH = 'DID_Panel_Ready_1.csv'

WINDOW            = 3
MIN_PAPERS_PRE    = 1
LATEST_COHORT     = 2023
MIN_CAREER_PAPERS = 3
MATCH_RATIO       = 1
PSM_CALIPER       = 0.02   # 倾向得分差距上限（× 标准差），可调为 0.10


# ─────────────────────────────────────────────────────────────────
# Step 1: 加载并分类作者
# ─────────────────────────────────────────────────────────────────
def load_and_classify_authors(path: str) -> pd.DataFrame:
    print("=" * 60)
    print("Step 1: 加载并分类作者")
    print("=" * 60)

    sep = '\t' if path.endswith('.tsv') or path.endswith('.txt') else ','
    df  = pd.read_csv(path, sep=sep)

    df['author_id'] = (
        df['author_id']
        .astype(str)
        .str.replace('https://openalex.org/', '', regex=False)
        .str.strip()
    )

    df['is_hss'] = df['primary_origin_field'].isin(HSS_FIELDS)

    missing_mask = df['primary_origin_field'].isna()
    if missing_mask.any():
        def dna_top_field_is_hss(dna_str):
            try:
                dna = ast.literal_eval(dna_str) if isinstance(dna_str, str) else {}
                if not dna:
                    return False
                return max(dna, key=dna.get) in HSS_FIELDS
            except Exception:
                return False
        df.loc[missing_mask, 'is_hss'] = (
            df.loc[missing_mask, 'scopus_dna_weights'].apply(dna_top_field_is_hss)
        )

    print(f"✅ 作者总数：{len(df):,}")
    print(f"   HSS  作者：{df['is_hss'].sum():,}  ({df['is_hss'].mean()*100:.1f}%)")
    print(f"   STEM 作者：{(~df['is_hss']).sum():,}  ({(~df['is_hss']).mean()*100:.1f}%)")
    print(f"\n   primary_origin_field 全部取值（Top 30）：")
    print(df['primary_origin_field'].value_counts(dropna=False).head(30).to_string())

    return df


# ─────────────────────────────────────────────────────────────────
# Step 2: 扫描论文 → 识别跨界事件 + 建立作者-年-论文映射
# ─────────────────────────────────────────────────────────────────
def scan_papers(
    jsonl_files: list,
    author_df: pd.DataFrame,
) -> tuple:
    print("\n" + "=" * 60)
    print("Step 2: 扫描论文（识别跨界事件 + 建立作者-年-论文映射）")
    print("=" * 60)

    hss_ids  = set(author_df[author_df['is_hss'] == True ]['author_id'])
    stem_ids = set(author_df[author_df['is_hss'] == False]['author_id'])

    crossover_years    = defaultdict(list)
    author_year_papers = defaultdict(list)
    total_lines        = 0

    for fpath in jsonl_files:
        if not os.path.exists(fpath):
            print(f"⚠️  文件不存在，跳过：{fpath}")
            continue

        print(f"   📄 扫描：{os.path.basename(fpath)}")
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total_lines += 1

                try:
                    paper = json.loads(line)
                except json.JSONDecodeError:
                    continue

                year    = paper.get('publication_year')
                raw_ids = paper.get('author_ids', []) or []

                author_ids = {
                    aid.replace('https://openalex.org/', '').strip()
                    for aid in raw_ids
                    if aid is not None and isinstance(aid, str) and aid.strip()
                }

                raw_work_id = paper.get('work_id')
                if not raw_work_id or not isinstance(raw_work_id, str):
                    continue
                work_id = raw_work_id.replace('https://openalex.org/', '').strip()

                if not year or not work_id:
                    continue

                year = int(year)

                for aid in author_ids:
                    if aid in stem_ids or aid in hss_ids:
                        author_year_papers[(aid, year)].append(work_id)

                if len(author_ids) < 2:
                    continue

                paper_hss  = author_ids & hss_ids
                paper_stem = author_ids & stem_ids

                if paper_hss and paper_stem:
                    for sid in paper_stem:
                        crossover_years[sid].append(year)

    event_year_map = {
        sid: min(years)
        for sid, years in crossover_years.items()
        if sid in stem_ids
    }

    print(f"\n✅ 共扫描 {total_lines:,} 行")
    print(f"   曾发生跨界合作的 STEM 作者：{len(event_year_map):,} 人")
    print(f"   从未跨界的 STEM 作者（对照组候选）：{len(stem_ids) - len(event_year_map):,} 人")

    cohort_dist = pd.Series(event_year_map.values()).value_counts().sort_index()
    print(f"\n   跨界队列年份分布（含全部年份）：")
    print(cohort_dist.to_string())

    return event_year_map, author_year_papers


# ─────────────────────────────────────────────────────────────────
# Step 3: 计算 RCR → avg_logRCR 面板
# ─────────────────────────────────────────────────────────────────
def compute_avg_logRCR(
    jsonl_files: list,
    citation_map_path: str,
    author_year_papers: dict,
) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("Step 3: 计算 RCR → avg_logRCR 面板")
    print("=" * 60)

    print("   📥 加载 citation_mapping.json ...")
    with open(citation_map_path, 'r', encoding='utf-8') as f:
        cite_map = json.load(f)

    print("   📄 加载论文元数据（field, year）...")
    paper_meta = {}

    for fpath in jsonl_files:
        if not os.path.exists(fpath):
            continue
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    paper = json.loads(line)
                except json.JSONDecodeError:
                    continue

                raw_work_id = paper.get('work_id')
                if not raw_work_id or not isinstance(raw_work_id, str):
                    continue
                work_id  = raw_work_id.replace('https://openalex.org/', '').strip()
                year     = paper.get('publication_year')
                concepts = paper.get('concepts', []) or []

                if work_id and year:
                    paper_meta[work_id] = {
                        'pub_year': int(year),
                        'field'   : concepts[0] if concepts else 'Unknown',
                    }

    print(f"   ✅ 论文元数据：{len(paper_meta):,} 篇")

    print("   ⚙️  计算论文级 RCR ...")
    rows = []
    for work_id, meta in paper_meta.items():
        cite_info = cite_map.get(work_id, {})
        raw_cites = cite_info.get('citations', 0) or 0
        rows.append({
            'work_id'     : work_id,
            'pub_year'    : meta['pub_year'],
            'field'       : meta['field'],
            'raw_cites'   : raw_cites,
            'in_cite_map' : work_id in cite_map,
        })

    df_papers = pd.DataFrame(rows)

    coverage = df_papers['in_cite_map'].mean() * 100
    print(f"\n   📊 citation_mapping 覆盖率：{coverage:.1f}%"
          f"（{df_papers['in_cite_map'].sum():,} / {len(df_papers):,} 篇）")

    # field × pub_year 中位数基准
    baseline = (
        df_papers.groupby(['field', 'pub_year'])['raw_cites']
        .median()
        .reset_index()
        .rename(columns={'raw_cites': 'expected_cites'})
    )
    df_papers = df_papers.merge(baseline, on=['field', 'pub_year'], how='left')
    df_papers['expected_cites'] = df_papers['expected_cites'].replace(0, np.nan)

    df_papers['RCR']     = df_papers['raw_cites'] / df_papers['expected_cites']
    df_papers['log_RCR'] = np.log1p(df_papers['RCR'].fillna(0))

    # Winsorize 到 p99
    p99 = df_papers['log_RCR'].quantile(0.99)
    p95 = df_papers['log_RCR'].quantile(0.95)
    print(f"\n   📊 log_RCR 分布诊断：")
    print(f"      均值={df_papers['log_RCR'].mean():.3f}  "
          f"中位数={df_papers['log_RCR'].median():.3f}  "
          f"p95={p95:.3f}  p99={p99:.3f}  "
          f"最大值={df_papers['log_RCR'].max():.3f}")
    df_papers['log_RCR'] = df_papers['log_RCR'].clip(upper=p99)
    print(f"   ✅ 已截尾至 p99={p99:.3f}")

    logrcr_lookup  = df_papers.set_index('work_id')['log_RCR'].to_dict()
    valid_work_ids = set(df_papers[df_papers['in_cite_map']]['work_id'])

    print("   ⚙️  聚合到作者-年面板 ...")
    records = []
    for (author_id, year), work_ids in author_year_papers.items():
        valid_wids = [wid for wid in work_ids if wid in valid_work_ids]
        log_rcrs   = [logrcr_lookup[wid] for wid in valid_wids if wid in logrcr_lookup]

        records.append({
            'author_id'          : author_id,
            'year'               : year,
            'avg_logRCR'         : float(np.mean(log_rcrs)) if log_rcrs else np.nan,
            'yearly_paper_count' : len(log_rcrs),
            'total_papers_raw'   : len(work_ids),
        })

    panel = pd.DataFrame(records)
    print(f"✅ 面板：{len(panel):,} 行 | "
          f"作者数 {panel['author_id'].nunique():,} | "
          f"年份 {panel['year'].min()}~{panel['year'].max()}")

    return panel


# ─────────────────────────────────────────────────────────────────
# Step 4a: 过滤 citation_mapping 无记录的幽灵作者
# ─────────────────────────────────────────────────────────────────
def filter_valid_authors(panel: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("Step 4a: 过滤幽灵作者（citation_mapping 无记录）")
    print("=" * 60)

    before_rows = len(panel)
    panel       = panel.dropna(subset=['avg_logRCR']).copy()
    after_rows  = len(panel)
    print(f"   删除 avg_logRCR=NaN 的行：{before_rows:,} → {after_rows:,} 行")

    author_max_rcr = panel.groupby('author_id')['avg_logRCR'].max()
    valid_authors  = set(author_max_rcr[author_max_rcr > 0].index)

    before_authors = panel['author_id'].nunique()
    panel          = panel[panel['author_id'].isin(valid_authors)].copy()
    after_authors  = panel['author_id'].nunique()

    print(f"   剔除生涯全为零引用的作者：{before_authors:,} → {after_authors:,} 位")
    print(f"   ✅ 过滤后面板：{len(panel):,} 行")

    return panel

# ─────────────────────────────────────────────────────────────────
# Step 4b: 活跃度筛选 + PSM 倾向得分匹配（KD-Tree 加速版）
# ─────────────────────────────────────────────────────────────────
def match_control_group(
    panel: pd.DataFrame,
    author_df: pd.DataFrame,
    ratio: int = MATCH_RATIO,
    min_career_papers: int = MIN_CAREER_PAPERS,
    caliper: float = PSM_CALIPER,
) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("Step 4b: 活跃度筛选 + PSM 倾向得分匹配（修复版）")
    print("=" * 60)

    # ── 活跃度筛选 ────────────────────────────────────────────
    career_papers  = panel.groupby('author_id')['yearly_paper_count'].sum()
    active_authors = set(career_papers[career_papers >= min_career_papers].index)
    panel          = panel[panel['author_id'].isin(active_authors)].copy()

    n_treat = panel[panel['is_treated'] == 1]['author_id'].nunique()
    n_ctrl  = panel[panel['is_treated'] == 0]['author_id'].nunique()
    print(f"   活跃度筛选（生涯 ≥ {min_career_papers} 篇有效论文）：")
    print(f"   实验组：{n_treat:,} 人  |  对照组：{n_ctrl:,} 人")

    # ── 🔧 修复 4：控制组基线只使用 2021 年（固定时点）前的数据 ──
    treated_baseline = (
        panel[
            (panel['is_treated'] == 1) &
            (panel['relative_year'] < 0)       # 实验组：处理前
        ]
        .groupby('author_id')['avg_logRCR'].mean()
        .reset_index()
        .rename(columns={'avg_logRCR': 'baseline_rcr'})
    )
    control_baseline = (
        panel[
            (panel['is_treated'] == 0) &
            (panel['year'] <= 2021)             # 🔧 控制组：统一基准年前
        ]
        .groupby('author_id')['avg_logRCR'].mean()
        .reset_index()
        .rename(columns={'avg_logRCR': 'baseline_rcr'})
    )
    baseline_all = pd.concat([treated_baseline, control_baseline], ignore_index=True)

    # ── 构建 PSM 特征矩阵 ─────────────────────────────────────
    meta_cols = ['author_id', 'primary_origin_field']
    if 'T0_year' in author_df.columns:
        meta_cols.append('T0_year')
    else:
        # 🔧 修复 5：明确告知缺失字段
        print("\n   ⚠️  T0_year 字段不存在，倾向得分模型仅使用 baseline_rcr + field_encoded")
        print("      建议在 author_df 中补充学术年龄或首发年字段以提升匹配质量")

    author_summary = (
        panel.drop_duplicates('author_id')
        [['author_id', 'is_treated']]
        .merge(author_df[meta_cols], on='author_id', how='left')
        .merge(baseline_all, on='author_id', how='left')
    )

    # 填补缺失值
    author_summary['baseline_rcr'] = (
        author_summary['baseline_rcr']
        .fillna(author_summary['baseline_rcr'].median())
    )
    author_summary['primary_origin_field'] = (
        author_summary['primary_origin_field'].fillna('Unknown')
    )
    if 'T0_year' in author_summary.columns:
        author_summary['T0_year'] = (
            author_summary['T0_year']
            .fillna(author_summary['T0_year'].median())
            .astype(int)
        )

    le = LabelEncoder()
    author_summary['field_encoded'] = le.fit_transform(
        author_summary['primary_origin_field']
    )

    feature_cols = ['baseline_rcr', 'field_encoded']
    if 'T0_year' in author_summary.columns:
        feature_cols.append('T0_year')

    X = author_summary[feature_cols].values
    y = author_summary['is_treated'].values

    # ── 训练逻辑回归 → 倾向得分 ───────────────────────────────
    print(f"\n   ⚙️  训练倾向得分模型（特征：{feature_cols}）...")
    lr = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
    lr.fit(X, y)
    author_summary['pscore'] = lr.predict_proba(X)[:, 1]

    t_ps = author_summary[author_summary['is_treated'] == 1]['pscore']
    c_ps = author_summary[author_summary['is_treated'] == 0]['pscore']
    print(f"\n   📊 倾向得分分布：")
    print(f"   实验组：均值={t_ps.mean():.4f}  std={t_ps.std():.4f}  "
          f"min={t_ps.min():.4f}  max={t_ps.max():.4f}")
    print(f"   对照组：均值={c_ps.mean():.4f}  std={c_ps.std():.4f}  "
          f"min={c_ps.min():.4f}  max={c_ps.max():.4f}")

    # 🔧 修复 3：自适应 Caliper = caliper × std(pscore)
    pooled_std  = author_summary['pscore'].std()
    abs_caliper = caliper * pooled_std
    print(f"\n   Caliper（自适应）= {caliper} × std({pooled_std:.4f}) = {abs_caliper:.4f}")

    # ── KD-Tree 加速匹配 ──────────────────────────────────────
    treated_df = author_summary[author_summary['is_treated'] == 1].copy()
    control_df = author_summary[author_summary['is_treated'] == 0].copy()

    treated_df = treated_df.sort_values('pscore').reset_index(drop=True)
    control_df = control_df.sort_values('pscore').reset_index(drop=True)

    ctrl_ps_arr  = control_df['pscore'].values.reshape(-1, 1)
    trt_ps_arr   = treated_df['pscore'].values.reshape(-1, 1)
    ctrl_ids_arr = control_df['author_id'].values

    # 🔧 修复 6：更保险的 k_neighbors 上限
    k_neighbors = min(max(ratio * 50, 100), len(control_df))

    nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='kd_tree')
    nbrs.fit(ctrl_ps_arr)
    distances, indices = nbrs.kneighbors(trt_ps_arr)

    # 🔧 修复 1：同时记录成功匹配的实验组，未匹配的不保留
    used_control        = set()
    matched_treat_ids   = set()          # ← 新增：只记录匹配成功的实验组
    matched_control_ids = []
    unmatched_count     = 0

    for i, (dists, idxs) in enumerate(zip(distances, indices)):
        candidates = [
            ctrl_ids_arr[j]
            for d, j in zip(dists, idxs)
            if d <= abs_caliper and ctrl_ids_arr[j] not in used_control
        ]

        if not candidates:
            unmatched_count += 1
            continue                     # ← 未匹配的实验组跳过，不加入集合

        selected = candidates[:ratio]

        # ✅ 只有找到匹配对，才记录该实验组作者
        matched_treat_ids.add(treated_df.iloc[i]['author_id'])
        for cid in selected:
            matched_control_ids.append(cid)
            used_control.add(cid)

    # ── 打印匹配结果 ──────────────────────────────────────────
    print(f"\n   匹配结果：")
    print(f"   实验组总数              ：{len(treated_df):,} 人")
    print(f"   未找到匹配（已剔除）    ：{unmatched_count:,} 人"
          f"（{unmatched_count / len(treated_df) * 100:.1f}%）")
    print(f"   成功匹配实验组          ：{len(matched_treat_ids):,} 人")
    print(f"   成功匹配对照组          ：{len(matched_control_ids):,} 人")

    if unmatched_count > len(treated_df) * 0.1:
        print(f"\n   ⚠️  超过 10% 实验组未匹配，建议调整全局配置：")
        print(f"      PSM_CALIPER = 0.20  （放宽 caliper，当前为 {caliper}）")
        print(f"      或 MATCH_RATIO = 1  （确保 1:1 精确匹配）")

    # 🔧 修复 1：只保留匹配成功的实验组 + 匹配到的对照组
    matched_ids = matched_treat_ids | set(matched_control_ids)
    panel       = panel[panel['author_id'].isin(matched_ids)].copy()

    n_treat_after = panel[panel['is_treated'] == 1]['author_id'].nunique()
    n_ctrl_after  = panel[panel['is_treated'] == 0]['author_id'].nunique()
    print(f"\n   PSM 匹配后（1:{ratio}，caliper={caliper}×std={abs_caliper:.4f}）：")
    print(f"   实验组：{n_treat_after:,} 人  |  对照组：{n_ctrl_after:,} 人")

    # ── 匹配后平衡性检验（SMD）────────────────────────────────
    matched_summary = author_summary[author_summary['author_id'].isin(matched_ids)]
    t_matched = matched_summary[matched_summary['is_treated'] == 1]
    c_matched = matched_summary[matched_summary['is_treated'] == 0]

    print(f"\n   📊 匹配后平衡性检验（标准化均值差 SMD，< 0.1 为达标）：")
    for col in ['baseline_rcr'] + (['T0_year'] if 'T0_year' in matched_summary.columns else []):
        x1 = t_matched[col].dropna()
        x2 = c_matched[col].dropna()
        smd_val = (x1.mean() - x2.mean()) / np.sqrt(
            (x1.std() ** 2 + x2.std() ** 2) / 2 + 1e-9
        )
        flag = '✅' if abs(smd_val) < 0.1 else '⚠️ '
        print(f"   {flag} {col:20s}：SMD = {smd_val:+.4f}  "
              f"实验组均值={x1.mean():.4f}  对照组均值={x2.mean():.4f}")

    t_base = t_matched['baseline_rcr']
    c_base = c_matched['baseline_rcr']
    gap    = t_base.mean() - c_base.mean()
    print(f"\n   📊 PSM 后基线 RCR 对比：")
    print(f"   实验组：均值={t_base.mean():.4f}  中位数={t_base.median():.4f}")
    print(f"   对照组：均值={c_base.mean():.4f}  中位数={c_base.median():.4f}")
    print(f"   均值差距：{gap:.4f}  "
          f"{'✅ 达标（< 0.05）' if abs(gap) < 0.05 else '⚠️  仍超标，见建议'}")

    if abs(gap) >= 0.05:
        print(f"\n   💡 调参建议：")
        print(f"      1. 放宽 caliper：PSM_CALIPER = 0.20（当前 {caliper}）")
        print(f"      2. 或在回归时加入 baseline_rcr 作为协变量（RA-DID）")

    return panel




# ─────────────────────────────────────────────────────────────────
# Step 4: 构建 DID 面板（整合所有步骤）
# ─────────────────────────────────────────────────────────────────
def build_did_panel(
    panel: pd.DataFrame,
    author_df: pd.DataFrame,
    event_year_map: dict,
    window: int = WINDOW,
    min_papers_pre: int = MIN_PAPERS_PRE,
    latest_cohort: int = LATEST_COHORT,
) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("Step 4: 构建 DID 面板")
    print("=" * 60)

    # 只保留 STEM 作者
    stem_ids = set(author_df[author_df['is_hss'] == False]['author_id'])
    panel    = panel[panel['author_id'].isin(stem_ids)].copy()

    # 打标签
    panel['is_treated'] = panel['author_id'].apply(
        lambda x: 1 if x in event_year_map else 0
    )
    panel['event_year'] = panel['author_id'].map(event_year_map)
    panel['event_year'] = panel['event_year'].fillna(9999).astype(int)

    # 排除引用不成熟的晚期队列
    panel = panel[
        (panel['is_treated'] == 0) |
        (panel['event_year'] <= latest_cohort)
    ].copy()
    retained = panel[panel['is_treated'] == 1]['author_id'].nunique()
    print(f"   队列筛选（event_year ≤ {latest_cohort}）：实验组 {retained:,} 人")

    # relative_year + 窗口筛选
    panel['relative_year'] = panel.apply(
        lambda r: r['year'] - r['event_year']
        if r['is_treated'] == 1 else np.nan,
        axis=1
    )
    treated_in_window = (
        (panel['is_treated'] == 1) &
        (panel['relative_year'].between(-window, window))
    )
    panel = panel[treated_in_window | (panel['is_treated'] == 0)].copy()

    # 质量控制：处理前至少 min_papers_pre 篇
    if min_papers_pre > 0:
        pre_papers = (
            panel[
                (panel['is_treated'] == 1) &
                (panel['relative_year'] < 0)
            ]
            .groupby('author_id')['yearly_paper_count']
            .sum()
            .reset_index()
            .rename(columns={'yearly_paper_count': 'pre_papers'})
        )
        qualified = set(
            pre_papers[pre_papers['pre_papers'] >= min_papers_pre]['author_id']
        )
        panel = panel[
            (panel['is_treated'] == 0) |
            (panel['author_id'].isin(qualified))
        ].copy()
        print(f"   质量控制：实验组保留 {len(qualified):,} 位"
              f"（处理前至少 {min_papers_pre} 篇）")

    # Step 4a: 过滤幽灵作者
    panel = filter_valid_authors(panel)

    # Step 4b: 活跃度筛选 + PSM 匹配
    panel = match_control_group(panel, author_df)

    # DID 核心变量
    panel['Post']     = (panel['relative_year'] >= 0).astype(float)
    panel.loc[panel['is_treated'] == 0, 'Post'] = 0.0
    panel['DID_term'] = panel['is_treated'] * panel['Post']

    # 合并作者元数据
    meta_cols = ['author_id']
    optional  = [
        'is_industry_elite', 'is_academic_elite',
        'is_elite_overall', 'country_code',
        'primary_origin_field', 'T0_year',
    ]
    meta_cols += [c for c in optional if c in author_df.columns]
    panel = panel.merge(author_df[meta_cols], on='author_id', how='left')

    # 最终诊断输出
    n_treat = panel[panel['is_treated'] == 1]['author_id'].nunique()
    n_ctrl  = panel[panel['is_treated'] == 0]['author_id'].nunique()

    print(f"\n✅ DID 面板构建完成：")
    print(f"   总行数      ：{len(panel):,}")
    print(f"   实验组作者  ：{n_treat:,} 人")
    print(f"   对照组作者  ：{n_ctrl:,} 人")
    print(f"   年份范围    ：{panel['year'].min()} ~ {panel['year'].max()}")
    print(f"   avg_logRCR  ：均值={panel['avg_logRCR'].mean():.4f}  "
          f"中位数={panel['avg_logRCR'].median():.4f}  "
          f"标准差={panel['avg_logRCR'].std():.4f}")

    print(f"\n   实验组队列分布（筛选后）：")
    cohort_dist = (
        panel[panel['is_treated'] == 1]
        .drop_duplicates('author_id')['event_year']
        .value_counts()
        .sort_index()
    )
    print(cohort_dist.to_string())

    print(f"\n   【最终基线诊断】处理前 avg_logRCR 对比：")
    treat_pre = panel[
        (panel['is_treated'] == 1) & (panel['relative_year'] < 0)
    ]['avg_logRCR']
    ctrl_all  = panel[panel['is_treated'] == 0]['avg_logRCR']
    gap_final = treat_pre.mean() - ctrl_all.mean()
    print(f"   实验组（处理前）：均值={treat_pre.mean():.4f}  中位数={treat_pre.median():.4f}")
    print(f"   对照组（全部）  ：均值={ctrl_all.mean():.4f}  中位数={ctrl_all.median():.4f}")
    print(f"   均值差距        ：{gap_final:.4f}  "
          f"{'✅ 达标，可进入回归' if abs(gap_final) < 0.05 else '⚠️  建议调整 PSM_CALIPER 或使用 RA-DID'}")

    return panel


# ─────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────
def main():
    print("🚀 DID 面板构建流水线启动\n")

    # Step 1
    author_df = load_and_classify_authors(AUTHOR_CSV_PATH)

    # Step 2
    event_year_map, author_year_papers = scan_papers(
        INPUT_JSONL_FILES, author_df
    )

    # Step 3
    panel = compute_avg_logRCR(
        INPUT_JSONL_FILES, CITATION_MAP_PATH, author_year_papers
    )

    # Step 4
    did_panel = build_did_panel(panel, author_df, event_year_map)

    # 保存
    did_panel.to_csv(OUTPUT_PANEL_PATH, index=False)
    print(f"\n💾 已保存：{OUTPUT_PANEL_PATH}")
    print(f"\n前 5 行预览：")
    print(did_panel.head().to_string())


if __name__ == "__main__":
    main()
