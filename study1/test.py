# -*- coding: utf-8 -*-
"""
GenAI paper-level merge + variable construction (revised)

这版调整：
1. 删除 NON_HSS_BROAD / primary_field_proxy 逻辑
2. 不再用 concepts 强行试算 reference field diversity / entropy
3. 只做：
   - has_hss_concepts
   - Socially_Embedded_Topic
   - citing-side diffusion metrics
   - internal reference HSS trial
   - reference age / reference citation metrics from citation_mapping.json
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


# =========================
# 1. 文件路径
# =========================
MASTER_PATH = Path(r"E:\PythonProject\GenAI\data_process\test\Ultimate_Regression_Base_with_Diversity.csv")
CITING_PATH = Path(r"E:\PythonProject\GenAI\data_process\process\citing_disciplines_master.jsonl")
CITATION_MAP_PATH = Path(r"E:\PythonProject\GenAI\data_process\process\citation_mapping.json")

PHASE_FILES = [
    Path(r"E:\PythonProject\GenAI\data_scrap_v2\stage_1\phase1_hybrid_cleaned.jsonl"),
    Path(r"E:\PythonProject\GenAI\data_scrap_v2\stage_2\phase2_hybrid_cleaned.jsonl"),
    Path(r"E:\PythonProject\GenAI\data_scrap_v2\stage_3\phase3_hybrid_cleaned.jsonl"),
]

OUT_DIR = Path(r"E:\PythonProject\GenAI\data_process\derived")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_MAIN = OUT_DIR / "genai_regression_enriched_v2.csv"
OUT_REF = OUT_DIR / "genai_reference_trial_v2.csv"


# =========================
# 2. HSS 口径
# =========================
HSS_CANON = {
    "Social Sciences",
    "Arts and Humanities",
    "Psychology",
    "Business, Management and Accounting",
    "Economics, Econometrics and Finance",
    "Decision Sciences",
}

SOCIAL_RE = re.compile(
    r"(ethic|bias|fair|governance|policy|regulat|law|legal|copyright|privacy|"
    r"education|student|teacher|human|user|social|society|labor|worker|employment|"
    r"trust|safety|misinformation)",
    flags=re.I
)


# =========================
# 3. 工具函数
# =========================
def clean_openalex_id(x):
    """把 https://openalex.org/W123 -> W123"""
    if pd.isna(x):
        return np.nan
    s = str(x)
    m = re.search(r"(W\d+)", s)
    return m.group(1) if m else s.strip()


def json_dumps_safe(x):
    if isinstance(x, (dict, list)):
        return json.dumps(x, ensure_ascii=False)
    if pd.isna(x):
        return ""
    return str(x)


def normalize_label(x):
    """
    只负责把 HSS 同义项归并到统一口径；
    非 HSS 不做大规模分类，仅保留原值（或标准化后的原值）。
    """
    if pd.isna(x):
        return np.nan

    s_raw = str(x).strip()
    s = s_raw.lower().strip()
    s = s.replace("&", "and")

    # ---- HSS 归一化 ----
    if "psycholog" in s:
        return "Psychology"

    if s in {
        "social science", "social sciences", "sociology",
        "education", "law", "political science",
        "anthropology", "communication", "communications",
        "public administration"
    }:
        return "Social Sciences"

    if s in {"economics", "econometrics", "finance"}:
        return "Economics, Econometrics and Finance"

    if s in {
        "business", "management", "operations management",
        "marketing", "accounting"
    }:
        return "Business, Management and Accounting"

    if "decision" in s:
        return "Decision Sciences"

    if s in {
        "arts", "art", "humanities", "arts and humanities",
        "philosophy", "history", "linguistics", "literature"
    }:
        return "Arts and Humanities"

    # Computer science 单独统一，因为要算 NonCS share
    if s == "computer science":
        return "Computer Science"

    # 其他标签保留原值（做轻微规范化）
    return s_raw.strip()


def normalize_concepts(concepts):
    if not isinstance(concepts, list):
        return []
    out = []
    seen = set()
    for c in concepts:
        cc = normalize_label(c)
        if pd.isna(cc):
            continue
        if cc not in seen:
            seen.add(cc)
            out.append(cc)
    return out


def clean_ref_list(refs):
    if not isinstance(refs, list):
        return []
    return [clean_openalex_id(x) for x in refs if pd.notna(x)]


def top10_within_group(s):
    if s.notna().sum() == 0:
        return pd.Series(np.nan, index=s.index)
    cutoff = s.quantile(0.90)
    return (s >= cutoff).astype(int)


def has_hss_concepts_func(concepts_norm):
    if not isinstance(concepts_norm, list):
        return 0
    return int(any(x in HSS_CANON for x in concepts_norm))


def socially_embedded_topic_func(title, concepts_norm):
    title = "" if pd.isna(title) else str(title)
    concept_hit = any(x in HSS_CANON for x in concepts_norm) if isinstance(concepts_norm, list) else False
    title_hit = bool(SOCIAL_RE.search(title))
    return int(concept_hit or title_hit)


def summarize_citing_profile(citing_disciplines):
    """
    施引侧扩散指标：
    - HSS_Citing_Share：HSS / total
    - NonCS_Citation_Share：非 Computer Science / total
    - Citation_Field_Diversity：原始学科键数（经 HSS 同义归并后）
    - Citation_Entropy：同上
    """
    agg = {}
    for k, v in citing_disciplines.items():
        try:
            vv = float(v)
        except Exception:
            continue
        if vv <= 0:
            continue
        kk = normalize_label(k)
        agg[kk] = agg.get(kk, 0.0) + vv

    if len(agg) == 0:
        return {
            "Total_Citing_Profile_Count": 0.0,
            "Citation_Field_Diversity": 0,
            "Citation_Entropy": 0.0,
            "NonCS_Citation_Share": 0.0,
            "HSS_Citing_Share": 0.0,
        }

    total = float(sum(agg.values()))
    vals = np.array(list(agg.values()), dtype=float)
    shares = vals / total

    cs_count = agg.get("Computer Science", 0.0)
    hss_count = sum(v for k, v in agg.items() if k in HSS_CANON)

    return {
        "Total_Citing_Profile_Count": total,
        "Citation_Field_Diversity": int(sum(v > 0 for v in agg.values())),
        "Citation_Entropy": float(-(shares * np.log(shares)).sum()),
        "NonCS_Citation_Share": float((total - cs_count) / total) if total > 0 else 0.0,
        "HSS_Citing_Share": float(hss_count / total) if total > 0 else 0.0,
    }


def citing_metrics_from_row(row):
    """
    缺失处理规则：
    - citations == 0 且没有 citing_disciplines -> 扩散指标记 0
    - citations > 0 且没有 citing_disciplines -> 记缺失，并标记
    """
    d = row.get("citing_disciplines", np.nan)
    citations = row.get("citations", np.nan)

    has_profile = isinstance(d, dict) and len(d) > 0

    if has_profile:
        out = summarize_citing_profile(d)
        out["Citing_Profile_Present"] = 1
        out["citing_missing_flag"] = "has_profile"
        if pd.notna(citations) and citations > 0:
            out["Citing_Profile_Coverage"] = out["Total_Citing_Profile_Count"] / citations
        else:
            out["Citing_Profile_Coverage"] = np.nan
        return pd.Series(out)

    if pd.notna(citations) and citations == 0:
        return pd.Series({
            "Total_Citing_Profile_Count": 0.0,
            "Citation_Field_Diversity": 0,
            "Citation_Entropy": 0.0,
            "NonCS_Citation_Share": 0.0,
            "HSS_Citing_Share": 0.0,
            "Citing_Profile_Present": 0,
            "Citing_Profile_Coverage": np.nan,
            "citing_missing_flag": "zero_citations_no_profile",
        })

    flag = "positive_citations_missing_profile" if pd.notna(citations) and citations > 0 else "missing_profile"
    return pd.Series({
        "Total_Citing_Profile_Count": np.nan,
        "Citation_Field_Diversity": np.nan,
        "Citation_Entropy": np.nan,
        "NonCS_Citation_Share": np.nan,
        "HSS_Citing_Share": np.nan,
        "Citing_Profile_Present": 0,
        "Citing_Profile_Coverage": np.nan,
        "citing_missing_flag": flag,
    })


def load_phase_jsonl(paths):
    rows = []
    for p in paths:
        print(f"Loading phase file: {p}")
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                rows.append({
                    "clean_work_id": clean_openalex_id(obj.get("work_id")),
                    "title": obj.get("title"),
                    "publication_year_phase": obj.get("publication_year"),
                    "referenced_works": clean_ref_list(obj.get("referenced_works", [])),
                    "concepts": obj.get("concepts", []),
                })
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["clean_work_id"])
    return df


def load_citing_jsonl(path):
    rows = []
    print(f"Loading citing file: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append({
                "clean_work_id": clean_openalex_id(obj.get("work_id")),
                "citing_disciplines": obj.get("citing_disciplines", np.nan),
            })
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["clean_work_id"])
    return df


def load_citation_mapping(path):
    print(f"Loading citation mapping: {path}")
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    rows = []
    for k, v in obj.items():
        rows.append({
            "ref_work_id": clean_openalex_id(k),
            "ref_year": v.get("year"),
            "ref_citations": v.get("citations"),
        })
    return pd.DataFrame(rows)


def internal_reference_hss_metrics(refs, ref_hss_map):
    """
    只做“内部匹配版 HSS 参考占比”
    - refs 总数
    - phase 语料库内可匹配的 refs 数
    - 其中 HSS refs 数
    """
    if not isinstance(refs, list) or len(refs) == 0:
        return pd.Series({
            "InternalRef_Total": 0,
            "InternalRef_Matched": 0,
            "InternalRef_HSS_Matched": 0,
            "InternalRef_Coverage": np.nan,
            "HSS_Reference_Share_Internal_MatchedBase": np.nan,
            "HSS_Reference_Share_Internal_TotalBase": np.nan,
        })

    total_refs = len(refs)
    matched = 0
    hss_matched = 0

    for r in refs:
        if r in ref_hss_map:
            matched += 1
            hss_matched += int(ref_hss_map[r])

    coverage = matched / total_refs if total_refs > 0 else np.nan
    share_matched = hss_matched / matched if matched > 0 else np.nan
    share_total = hss_matched / total_refs if total_refs > 0 else np.nan

    return pd.Series({
        "InternalRef_Total": total_refs,
        "InternalRef_Matched": matched,
        "InternalRef_HSS_Matched": hss_matched,
        "InternalRef_Coverage": coverage,
        "HSS_Reference_Share_Internal_MatchedBase": share_matched,
        "HSS_Reference_Share_Internal_TotalBase": share_total,
    })


def reference_vintage_impact_metrics(refs, focal_year, citation_map_df_indexed):
    """
    利用 citation_mapping.json 计算：
    - 引用文献年份覆盖率
    - 平均参考年龄 / 中位参考年龄
    - 近3年参考占比
    - 参考文献平均 log(1+citations)
    """
    if not isinstance(refs, list) or len(refs) == 0 or pd.isna(focal_year):
        return pd.Series({
            "RefMeta_Total": 0,
            "RefMeta_Matched": 0,
            "RefMeta_Coverage": np.nan,
            "Mean_Reference_Age": np.nan,
            "Median_Reference_Age": np.nan,
            "Recent_Reference_Share_3y": np.nan,
            "Mean_Log_Ref_Citations": np.nan,
            "Median_Log_Ref_Citations": np.nan,
        })

    matched_years = []
    matched_citations = []

    for r in refs:
        if r in citation_map_df_indexed.index:
            row = citation_map_df_indexed.loc[r]
            ref_year = row["ref_year"]
            ref_cits = row["ref_citations"]

            if pd.notna(ref_year):
                age = focal_year - ref_year
                if pd.notna(age) and age >= 0:
                    matched_years.append(age)

            if pd.notna(ref_cits):
                matched_citations.append(np.log1p(ref_cits))

    total_refs = len(refs)
    matched = max(len(matched_years), len(matched_citations))
    coverage = matched / total_refs if total_refs > 0 else np.nan

    if len(matched_years) == 0:
        mean_age = np.nan
        median_age = np.nan
        recent_share = np.nan
    else:
        arr_age = np.array(matched_years, dtype=float)
        mean_age = float(arr_age.mean())
        median_age = float(np.median(arr_age))
        recent_share = float((arr_age <= 3).mean())

    if len(matched_citations) == 0:
        mean_log_ref_cits = np.nan
        median_log_ref_cits = np.nan
    else:
        arr_c = np.array(matched_citations, dtype=float)
        mean_log_ref_cits = float(arr_c.mean())
        median_log_ref_cits = float(np.median(arr_c))

    return pd.Series({
        "RefMeta_Total": total_refs,
        "RefMeta_Matched": matched,
        "RefMeta_Coverage": coverage,
        "Mean_Reference_Age": mean_age,
        "Median_Reference_Age": median_age,
        "Recent_Reference_Share_3y": recent_share,
        "Mean_Log_Ref_Citations": mean_log_ref_cits,
        "Median_Log_Ref_Citations": median_log_ref_cits,
    })


# =========================
# 4. 读取主表
# =========================
print(f"Loading master file: {MASTER_PATH}")
master = pd.read_csv(MASTER_PATH)

required_cols = [
    "clean_work_id", "stage", "arxiv_primary_category", "total_authors",
    "team_hss_dna_pct", "citations", "year", "RCR", "Log_RCR",
    "Reference_Count", "Prior_Knowledge_Stock", "HSS_Diversity"
]
missing_cols = [c for c in required_cols if c not in master.columns]
if missing_cols:
    raise ValueError(f"主表缺少字段: {missing_cols}")

master["clean_work_id"] = master["clean_work_id"].apply(clean_openalex_id)

# 主变量
master["HSS_Continuous"] = master["team_hss_dna_pct"] / 10.0
master["log_team_size"] = np.log1p(master["total_authors"])
master["log_prior_knowledge"] = np.log1p(master["Prior_Knowledge_Stock"])

master["HSS_Bin"] = pd.cut(
    master["team_hss_dna_pct"],
    bins=[-0.001, 0, 20, 40, 60, 100],
    labels=["Pure STEM", "Low HSS", "Medium HSS", "High HSS", "HSS-dominant"],
    right=True
)

# Top-10% 两种口径
master["Hit_Rate_10_year"] = master.groupby("year")["RCR"].transform(top10_within_group)
master["Hit_Rate_10_year_cat"] = master.groupby(["year", "arxiv_primary_category"])["RCR"].transform(top10_within_group)


# =========================
# 5. 读取 phase 文件并造主题变量
# =========================
phase = load_phase_jsonl(PHASE_FILES)

phase["concepts_norm"] = phase["concepts"].apply(normalize_concepts)
phase["concepts_norm_str"] = phase["concepts_norm"].apply(lambda xs: "; ".join(xs))
phase["has_hss_concepts"] = phase["concepts_norm"].apply(has_hss_concepts_func)
phase["Socially_Embedded_Topic"] = phase.apply(
    lambda r: socially_embedded_topic_func(r["title"], r["concepts_norm"]),
    axis=1
)

phase_keep = phase[
    [
        "clean_work_id",
        "title",
        "publication_year_phase",
        "referenced_works",
        "concepts_norm_str",
        "has_hss_concepts",
        "Socially_Embedded_Topic",
    ]
].copy()


# =========================
# 6. 读取 citing 文件并造扩散变量
# =========================
citing = load_citing_jsonl(CITING_PATH)

df = master.merge(phase_keep, on="clean_work_id", how="left")
df = df.merge(citing, on="clean_work_id", how="left")

df["phase_match_flag"] = df["title"].notna().astype(int)

# 没匹配上的先置 0
df["has_hss_concepts"] = df["has_hss_concepts"].fillna(0).astype(int)
df["Socially_Embedded_Topic"] = df["Socially_Embedded_Topic"].fillna(0).astype(int)

# 扩散变量
citing_metrics = df.apply(citing_metrics_from_row, axis=1)
df = pd.concat([df, citing_metrics], axis=1)


# =========================
# 7. 参考文献：内部 HSS 试算
# =========================
# 这里不再做“field diversity/entropy 试算”
# 只判断引用到的内部 refs 里有多少带 HSS concepts
ref_hss_map = phase.set_index("clean_work_id")["has_hss_concepts"].dropna().astype(int).to_dict()

ref_hss_metrics = df["referenced_works"].apply(lambda refs: internal_reference_hss_metrics(refs, ref_hss_map))
df = pd.concat([df, ref_hss_metrics], axis=1)


# =========================
# 8. 参考文献：年龄/被引强度
# =========================
citation_map_df = load_citation_mapping(CITATION_MAP_PATH)
citation_map_df = citation_map_df.drop_duplicates(subset=["ref_work_id"])
citation_map_df_indexed = citation_map_df.set_index("ref_work_id")

ref_meta_metrics = df.apply(
    lambda r: reference_vintage_impact_metrics(
        r["referenced_works"],
        r["year"],
        citation_map_df_indexed
    ),
    axis=1
)
df = pd.concat([df, ref_meta_metrics], axis=1)


# =========================
# 9. 诊断信息
# =========================
print("\n===== Basic diagnostics =====")
print(f"Master rows: {len(master):,}")
print(f"Phase rows: {len(phase):,}")
print(f"Citing rows: {len(citing):,}")
print(f"Final rows: {len(df):,}")

print("\nPhase match:")
print(df["phase_match_flag"].value_counts(dropna=False))

print("\nCiting missing flags:")
print(df["citing_missing_flag"].value_counts(dropna=False))

print("\nInternal reference HSS coverage summary:")
print(df["InternalRef_Coverage"].describe())

print("\nReference metadata coverage summary:")
print(df["RefMeta_Coverage"].describe())

problem_df = df[df["citing_missing_flag"] == "positive_citations_missing_profile"].copy()
print(f"\npositive_citations_missing_profile rows: {len(problem_df):,}")


# =========================
# 10. 导出
# =========================
df["citing_disciplines_json"] = df["citing_disciplines"].apply(json_dumps_safe)
df["referenced_works_json"] = df["referenced_works"].apply(json_dumps_safe)

save_df = df.drop(columns=["citing_disciplines", "referenced_works"], errors="ignore").copy()
save_df.to_csv(OUT_MAIN, index=False, encoding="utf-8-sig")

ref_out = save_df[
    [
        "clean_work_id", "stage", "year", "Reference_Count",
        "InternalRef_Total", "InternalRef_Matched", "InternalRef_HSS_Matched",
        "InternalRef_Coverage",
        "HSS_Reference_Share_Internal_MatchedBase",
        "HSS_Reference_Share_Internal_TotalBase",
        "RefMeta_Total", "RefMeta_Matched", "RefMeta_Coverage",
        "Mean_Reference_Age", "Median_Reference_Age", "Recent_Reference_Share_3y",
        "Mean_Log_Ref_Citations", "Median_Log_Ref_Citations",
    ]
].copy()
ref_out.to_csv(OUT_REF, index=False, encoding="utf-8-sig")

print(f"\nSaved main file to: {OUT_MAIN}")
print(f"Saved ref trial file to: {OUT_REF}")
print("\nDone.")
