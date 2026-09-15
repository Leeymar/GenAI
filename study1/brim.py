# -*- coding: utf-8 -*-
"""
brim.py

用途：
1. 读取现有的 enriched CSV
2. 将参考文献指标重定义为“OpenAlex 可观测参考文献”口径
3. 自动处理 status 列缺失 / 列名不一致
4. 区分：
   - 缺 enrichment（NOT_ENRICHED）
   - OpenAlex 可观测参考文献为 0（NO_OPENALEX_REFERENCES）
   - 只能做计数分析（OK_COUNT_ONLY）
   - 可做组成分析（OK_FOR_COMPOSITION）
5. 输出 analysis-ready CSV 和统计汇总

可直接在 PyCharm 运行，不需要 argparse。
"""

import os
import numpy as np
import pandas as pd


# =========================================================
# 1. 路径配置：这里只改你的输入文件路径
# =========================================================
INPUT_CSV = r"E:\PythonProject\GenAI\data_process\derived\genai_regression_enriched_v3.csv"

OUTPUT_DIR = r"openalex_reference_stats_output"
OUTPUT_ANALYSIS_READY_CSV = os.path.join(
    OUTPUT_DIR, "genai_regression_enriched_openalex_observable.csv"
)
OUTPUT_SUMMARY_TXT = os.path.join(
    OUTPUT_DIR, "openalex_reference_summary.txt"
)
OUTPUT_STATUS_COUNTS_CSV = os.path.join(
    OUTPUT_DIR, "status_counts.csv"
)
OUTPUT_MAIN_STATS_CSV = os.path.join(
    OUTPUT_DIR, "main_stats.csv"
)
OUTPUT_BY_YEAR_CSV = os.path.join(
    OUTPUT_DIR, "stats_by_year.csv"
)
OUTPUT_BY_TYPE_CSV = os.path.join(
    OUTPUT_DIR, "stats_by_type.csv"
)
OUTPUT_BY_SOURCE_CSV = os.path.join(
    OUTPUT_DIR, "stats_by_source.csv"
)
OUTPUT_DIAGNOSTIC_CSV = os.path.join(
    OUTPUT_DIR, "diagnostic_flags.csv"
)
OUTPUT_COLUMNS_TXT = os.path.join(
    OUTPUT_DIR, "detected_columns.txt"
)

# topic-based 组成指标只在已解析 topic 的引用数 >= 这个阈值时才纳入分析
MIN_TOPIC_RESOLVED_FOR_COMPOSITION = 5


# =========================================================
# 2. 辅助函数
# =========================================================
def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def to_numeric_safe(series):
    return pd.to_numeric(series, errors="coerce")


def safe_mean(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return np.nan
    return float(s.mean())


def safe_median(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return np.nan
    return float(s.median())


def safe_std(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) <= 1:
        return np.nan
    return float(s.std())


def safe_min(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return np.nan
    return float(s.min())


def safe_max(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return np.nan
    return float(s.max())


def safe_quantile(series, q):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return np.nan
    return float(s.quantile(q))


def format_num(x):
    if pd.isna(x):
        return "NA"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, float):
        if abs(x) >= 1000:
            return f"{x:,.2f}"
        return f"{x:.4f}"
    return str(x)


def summarize_numeric(series, name):
    s = pd.to_numeric(series, errors="coerce")
    return {
        "metric": name,
        "value": np.nan,
        "n_non_missing": int(s.notna().sum()),
        "mean": safe_mean(s),
        "median": safe_median(s),
        "std": safe_std(s),
        "min": safe_min(s),
        "p25": safe_quantile(s, 0.25),
        "p75": safe_quantile(s, 0.75),
        "max": safe_max(s),
    }


def value_counts_with_share(series, col_name="value"):
    vc = series.fillna("").astype(str).value_counts(dropna=False)
    total = vc.sum()
    rows = []
    for k, v in vc.items():
        rows.append({
            col_name: k,
            "count": int(v),
            "share": float(v / total) if total > 0 else np.nan
        })
    return pd.DataFrame(rows)


def pick_first_existing_column(df: pd.DataFrame, candidates):
    """
    先精确匹配，再忽略大小写匹配
    """
    cols = list(df.columns)
    lower_map = {str(c).lower(): c for c in cols}

    for c in candidates:
        if c in df.columns:
            return c

    for c in candidates:
        cl = str(c).lower()
        if cl in lower_map:
            return lower_map[cl]

    return None


def pick_first_contains_column(df: pd.DataFrame, include_keywords, exclude_keywords=None):
    """
    在列名里按关键词模糊匹配
    include_keywords: 所有关键词都要出现
    exclude_keywords: 任一出现则排除
    """
    exclude_keywords = exclude_keywords or []

    for c in df.columns:
        cl = str(c).lower()

        if all(k.lower() in cl for k in include_keywords):
            if not any(k.lower() in cl for k in exclude_keywords):
                return c
    return None


def build_group_stats(df_grouped, group_col_name):
    rows = []
    for key, g in df_grouped:
        rows.append({
            group_col_name: key,
            "n": int(len(g)),
            "share_rows_with_any_openalex_metrics": safe_mean(g["has_any_openalex_metrics"]),
            "share_has_openalex_references": safe_mean(g["has_openalex_references"]),
            "share_usable_for_count_analysis": safe_mean(g["usable_for_count_analysis"]),
            "share_usable_for_composition_analysis": safe_mean(g["usable_for_composition_analysis"]),
            "mean_openalex_matched_reference_count": safe_mean(g["openalex_matched_reference_count"]),
            "median_openalex_matched_reference_count": safe_median(g["openalex_matched_reference_count"]),
            "mean_openalex_topic_resolved_reference_count": safe_mean(g["openalex_topic_resolved_reference_count"]),
            "mean_openalex_reference_field_diversity": safe_mean(g["openalex_reference_field_diversity"]),
            "mean_openalex_reference_entropy": safe_mean(g["openalex_reference_entropy"]),
            "mean_hss_share_among_matched_refs": safe_mean(g["hss_share_among_matched_refs"]),
            "mean_cs_share_among_matched_refs": safe_mean(g["cs_share_among_matched_refs"]),
        })
    return pd.DataFrame(rows)


# =========================================================
# 3. 读取数据
# =========================================================
ensure_dir(OUTPUT_DIR)

if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"找不到输入文件: {INPUT_CSV}")

# low_memory=False 用来消掉你之前那个 DtypeWarning
df = pd.read_csv(INPUT_CSV, low_memory=False)

print(f"[INFO] Loaded rows: {len(df):,}")
print(f"[INFO] Loaded cols: {len(df.columns)}")


# =========================================================
# 4. 自动识别关键列
# =========================================================
status_col = pick_first_existing_column(
    df,
    ["status", "openalex_status", "oa_status", "reference_status"]
)
if status_col is None:
    status_col = pick_first_contains_column(df, ["status"])

year_col = pick_first_existing_column(df, ["publication_year", "year"])
if year_col is None:
    year_col = pick_first_contains_column(df, ["year"])

type_col = pick_first_existing_column(df, ["type", "work_type", "openalex_type"])
if type_col is None:
    type_col = pick_first_contains_column(df, ["type"], exclude_keywords=["prototype"])

source_col = pick_first_existing_column(
    df,
    ["source_display_name", "host_venue_display_name", "journal_name", "venue", "source"]
)
if source_col is None:
    source_col = pick_first_contains_column(df, ["source", "display", "name"])
if source_col is None:
    source_col = pick_first_contains_column(df, ["journal"])

work_id_col = pick_first_existing_column(df, ["work_id", "id", "openalex_id"])
if work_id_col is None:
    work_id_col = pick_first_contains_column(df, ["work", "id"])

reference_count_total_col = pick_first_existing_column(
    df,
    ["reference_count_total", "openalex_matched_reference_count"]
)
if reference_count_total_col is None:
    reference_count_total_col = pick_first_contains_column(df, ["reference", "count", "total"])

reference_count_with_primary_topic_col = pick_first_existing_column(
    df,
    ["reference_count_with_primary_topic", "openalex_topic_resolved_reference_count"]
)
if reference_count_with_primary_topic_col is None:
    reference_count_with_primary_topic_col = pick_first_contains_column(
        df, ["reference", "count", "primary", "topic"]
    )

reference_count_missing_primary_topic_col = pick_first_existing_column(
    df,
    ["reference_count_missing_primary_topic", "openalex_topic_missing_reference_count"]
)
if reference_count_missing_primary_topic_col is None:
    reference_count_missing_primary_topic_col = pick_first_contains_column(
        df, ["reference", "count", "missing", "primary", "topic"]
    )

reference_field_diversity_col = pick_first_existing_column(
    df,
    ["reference_field_diversity", "openalex_reference_field_diversity"]
)
if reference_field_diversity_col is None:
    reference_field_diversity_col = pick_first_contains_column(df, ["reference", "field", "diversity"])

reference_entropy_col = pick_first_existing_column(
    df,
    ["reference_entropy", "openalex_reference_entropy"]
)
if reference_entropy_col is None:
    reference_entropy_col = pick_first_contains_column(df, ["reference", "entropy"])

hss_reference_share_col = pick_first_existing_column(
    df,
    ["hss_reference_share", "hss_share_among_matched_refs"]
)
if hss_reference_share_col is None:
    hss_reference_share_col = pick_first_contains_column(df, ["hss", "share"])

cs_core_reference_share_col = pick_first_existing_column(
    df,
    ["cs_core_reference_share", "cs_share_among_matched_refs"]
)
if cs_core_reference_share_col is None:
    cs_core_reference_share_col = pick_first_contains_column(df, ["cs", "share"])

print("[INFO] Detected columns:")
print("  status_col =", status_col)
print("  year_col =", year_col)
print("  type_col =", type_col)
print("  source_col =", source_col)
print("  work_id_col =", work_id_col)
print("  reference_count_total_col =", reference_count_total_col)
print("  reference_count_with_primary_topic_col =", reference_count_with_primary_topic_col)
print("  reference_count_missing_primary_topic_col =", reference_count_missing_primary_topic_col)
print("  reference_field_diversity_col =", reference_field_diversity_col)
print("  reference_entropy_col =", reference_entropy_col)
print("  hss_reference_share_col =", hss_reference_share_col)
print("  cs_core_reference_share_col =", cs_core_reference_share_col)

with open(OUTPUT_COLUMNS_TXT, "w", encoding="utf-8") as f:
    f.write("Detected key columns\n")
    f.write("=" * 60 + "\n")
    f.write(f"status_col = {status_col}\n")
    f.write(f"year_col = {year_col}\n")
    f.write(f"type_col = {type_col}\n")
    f.write(f"source_col = {source_col}\n")
    f.write(f"work_id_col = {work_id_col}\n")
    f.write(f"reference_count_total_col = {reference_count_total_col}\n")
    f.write(f"reference_count_with_primary_topic_col = {reference_count_with_primary_topic_col}\n")
    f.write(f"reference_count_missing_primary_topic_col = {reference_count_missing_primary_topic_col}\n")
    f.write(f"reference_field_diversity_col = {reference_field_diversity_col}\n")
    f.write(f"reference_entropy_col = {reference_entropy_col}\n")
    f.write(f"hss_reference_share_col = {hss_reference_share_col}\n")
    f.write(f"cs_core_reference_share_col = {cs_core_reference_share_col}\n")
    f.write("\n\nAll columns\n")
    f.write("=" * 60 + "\n")
    for c in df.columns:
        f.write(str(c) + "\n")


# =========================================================
# 5. 标准化数值列
# =========================================================
if reference_count_total_col is None:
    raise ValueError("找不到 reference_count_total / openalex_matched_reference_count 这类列，脚本无法继续。")

df["openalex_matched_reference_count"] = to_numeric_safe(df[reference_count_total_col])

# 已解析 topic 的引用数
if reference_count_with_primary_topic_col is not None:
    df["openalex_topic_resolved_reference_count"] = to_numeric_safe(df[reference_count_with_primary_topic_col])
else:
    df["openalex_topic_resolved_reference_count"] = np.nan

# 缺 topic 的引用数
if reference_count_missing_primary_topic_col is not None:
    df["openalex_topic_missing_reference_count"] = to_numeric_safe(df[reference_count_missing_primary_topic_col])
else:
    df["openalex_topic_missing_reference_count"] = np.nan

# 尝试自动补齐 resolved / missing
if reference_count_with_primary_topic_col is None and reference_count_missing_primary_topic_col is not None:
    df["openalex_topic_resolved_reference_count"] = (
        df["openalex_matched_reference_count"] - df["openalex_topic_missing_reference_count"]
    )

if reference_count_missing_primary_topic_col is None and reference_count_with_primary_topic_col is not None:
    df["openalex_topic_missing_reference_count"] = (
        df["openalex_matched_reference_count"] - df["openalex_topic_resolved_reference_count"]
    )

# 防止出现负值
df["openalex_topic_resolved_reference_count"] = pd.to_numeric(
    df["openalex_topic_resolved_reference_count"], errors="coerce"
)
df["openalex_topic_missing_reference_count"] = pd.to_numeric(
    df["openalex_topic_missing_reference_count"], errors="coerce"
)

df.loc[df["openalex_topic_resolved_reference_count"] < 0, "openalex_topic_resolved_reference_count"] = np.nan
df.loc[df["openalex_topic_missing_reference_count"] < 0, "openalex_topic_missing_reference_count"] = np.nan

# 原始组成指标
if reference_field_diversity_col is not None:
    df["openalex_reference_field_diversity_raw"] = to_numeric_safe(df[reference_field_diversity_col])
else:
    df["openalex_reference_field_diversity_raw"] = np.nan

if reference_entropy_col is not None:
    df["openalex_reference_entropy_raw"] = to_numeric_safe(df[reference_entropy_col])
else:
    df["openalex_reference_entropy_raw"] = np.nan

if hss_reference_share_col is not None:
    df["hss_share_among_matched_refs_raw"] = to_numeric_safe(df[hss_reference_share_col])
else:
    df["hss_share_among_matched_refs_raw"] = np.nan

if cs_core_reference_share_col is not None:
    df["cs_share_among_matched_refs_raw"] = to_numeric_safe(df[cs_core_reference_share_col])
else:
    df["cs_share_among_matched_refs_raw"] = np.nan


# =========================================================
# 6. 生成 / 标准化 status
# =========================================================
if status_col is not None:
    df["status_std"] = (
        df[status_col]
        .astype(str)
        .fillna("")
        .replace("nan", "")
        .replace("None", "")
        .str.strip()
    )
else:
    df["status_std"] = np.select(
        [
            df["openalex_matched_reference_count"].isna(),
            df["openalex_matched_reference_count"].eq(0),
            (df["openalex_matched_reference_count"] > 0)
            & (df["openalex_topic_resolved_reference_count"].fillna(0) >= MIN_TOPIC_RESOLVED_FOR_COMPOSITION),
            (df["openalex_matched_reference_count"] > 0),
        ],
        [
            "NOT_ENRICHED",
            "NO_OPENALEX_REFERENCES",
            "OK_FOR_COMPOSITION",
            "OK_COUNT_ONLY",
        ],
        default=""
    )


# =========================================================
# 7. 重定义成“OpenAlex 可观测参考文献”口径
# =========================================================
# 7.1 是否有任何 OpenAlex enrichment
df["has_any_openalex_metrics"] = df["openalex_matched_reference_count"].notna().astype(int)

# 7.2 是否有 OpenAlex 可观测 references
df["has_openalex_references"] = np.nan
mask_non_missing_count = df["openalex_matched_reference_count"].notna()
df.loc[mask_non_missing_count, "has_openalex_references"] = (
    df.loc[mask_non_missing_count, "openalex_matched_reference_count"] > 0
).astype(int)

# 7.3 log count
df["log1p_openalex_matched_reference_count"] = np.where(
    df["openalex_matched_reference_count"].notna(),
    np.log1p(df["openalex_matched_reference_count"].clip(lower=0)),
    np.nan
)

# 7.4 是否可用于计数分析
df["usable_for_count_analysis"] = df["openalex_matched_reference_count"].notna().astype(int)

# 7.5 是否可用于组成分析
df["composition_metrics_eligible"] = (
    (df["openalex_matched_reference_count"] > 0)
    & (df["openalex_topic_resolved_reference_count"].fillna(0) >= MIN_TOPIC_RESOLVED_FOR_COMPOSITION)
).astype(int)

df["usable_for_composition_analysis"] = df["composition_metrics_eligible"]

# 7.6 真正用于分析的组成指标
# 对 reference_count == 0 或 topic_resolved_count 不足的行，组成指标记为 NA，而不是 0
df["openalex_reference_field_diversity"] = np.where(
    df["composition_metrics_eligible"] == 1,
    df["openalex_reference_field_diversity_raw"],
    np.nan
)

df["openalex_reference_entropy"] = np.where(
    df["composition_metrics_eligible"] == 1,
    df["openalex_reference_entropy_raw"],
    np.nan
)

df["hss_share_among_matched_refs"] = np.where(
    df["composition_metrics_eligible"] == 1,
    df["hss_share_among_matched_refs_raw"],
    np.nan
)

df["cs_share_among_matched_refs"] = np.where(
    df["composition_metrics_eligible"] == 1,
    df["cs_share_among_matched_refs_raw"],
    np.nan
)


# =========================================================
# 8. 诊断 flag
# =========================================================
df["flag_blank_status"] = (df["status_std"].astype(str).str.strip() == "").astype(int)
df["flag_status_inferred"] = 0 if status_col is not None else 1
df["flag_not_enriched"] = (df["status_std"] == "NOT_ENRICHED").astype(int)
df["flag_no_openalex_references"] = (df["status_std"] == "NO_OPENALEX_REFERENCES").astype(int)
df["flag_ok_count_only"] = (df["status_std"] == "OK_COUNT_ONLY").astype(int)
df["flag_ok_for_composition"] = (df["status_std"] == "OK_FOR_COMPOSITION").astype(int)

df["flag_count_missing"] = df["openalex_matched_reference_count"].isna().astype(int)
df["flag_count_zero"] = df["openalex_matched_reference_count"].fillna(-999999).eq(0).astype(int)
df["flag_count_positive"] = (df["openalex_matched_reference_count"] > 0).astype(int)

df["flag_zero_but_raw_composition_not_missing"] = np.where(
    (df["openalex_matched_reference_count"] == 0)
    & (
        df["openalex_reference_field_diversity_raw"].notna()
        | df["openalex_reference_entropy_raw"].notna()
        | df["hss_share_among_matched_refs_raw"].notna()
        | df["cs_share_among_matched_refs_raw"].notna()
    ),
    1,
    0
)

df["flag_positive_count_but_no_topic_resolved"] = np.where(
    (df["openalex_matched_reference_count"] > 0)
    & (df["openalex_topic_resolved_reference_count"].fillna(0) == 0),
    1,
    0
)

df["flag_topic_resolved_gt_total"] = np.where(
    df["openalex_topic_resolved_reference_count"] > df["openalex_matched_reference_count"],
    1,
    0
)

df["flag_topic_missing_gt_total"] = np.where(
    df["openalex_topic_missing_reference_count"] > df["openalex_matched_reference_count"],
    1,
    0
)


# =========================================================
# 9. 输出 analysis-ready CSV
# =========================================================
df.to_csv(OUTPUT_ANALYSIS_READY_CSV, index=False, encoding="utf-8-sig")
print(f"[OK] Saved analysis-ready CSV: {OUTPUT_ANALYSIS_READY_CSV}")


# =========================================================
# 10. 总体统计
# =========================================================
summary_rows = [
    {"metric": "n_total_rows", "value": int(len(df))},
    {"metric": "n_rows_with_any_openalex_metrics", "value": int(df["has_any_openalex_metrics"].sum())},
    {"metric": "share_rows_with_any_openalex_metrics", "value": safe_mean(df["has_any_openalex_metrics"])},
    {"metric": "n_rows_without_openalex_metrics", "value": int((df["has_any_openalex_metrics"] == 0).sum())},
    {"metric": "n_rows_has_openalex_references", "value": int((df["has_openalex_references"] == 1).sum())},
    {"metric": "share_rows_has_openalex_references", "value": safe_mean(df["has_openalex_references"])},
    {"metric": "n_rows_no_openalex_references", "value": int((df["has_openalex_references"] == 0).sum())},
    {"metric": "n_rows_usable_for_count_analysis", "value": int(df["usable_for_count_analysis"].sum())},
    {"metric": "share_rows_usable_for_count_analysis", "value": safe_mean(df["usable_for_count_analysis"])},
    {"metric": "n_rows_usable_for_composition_analysis", "value": int(df["usable_for_composition_analysis"].sum())},
    {"metric": "share_rows_usable_for_composition_analysis", "value": safe_mean(df["usable_for_composition_analysis"])},
    {"metric": "n_blank_status", "value": int(df["flag_blank_status"].sum())},
    {"metric": "n_not_enriched", "value": int(df["flag_not_enriched"].sum())},
    {"metric": "n_no_openalex_references_status", "value": int(df["flag_no_openalex_references"].sum())},
    {"metric": "n_ok_count_only_status", "value": int(df["flag_ok_count_only"].sum())},
    {"metric": "n_ok_for_composition_status", "value": int(df["flag_ok_for_composition"].sum())},
    {"metric": "n_zero_but_raw_composition_not_missing", "value": int(df["flag_zero_but_raw_composition_not_missing"].sum())},
    {"metric": "n_positive_count_but_no_topic_resolved", "value": int(df["flag_positive_count_but_no_topic_resolved"].sum())},
    {"metric": "n_topic_resolved_gt_total", "value": int(df["flag_topic_resolved_gt_total"].sum())},
    {"metric": "n_topic_missing_gt_total", "value": int(df["flag_topic_missing_gt_total"].sum())},
]

main_numeric_metrics = [
    "openalex_matched_reference_count",
    "log1p_openalex_matched_reference_count",
    "openalex_topic_resolved_reference_count",
    "openalex_topic_missing_reference_count",
    "openalex_reference_field_diversity",
    "openalex_reference_entropy",
    "hss_share_among_matched_refs",
    "cs_share_among_matched_refs",
]

numeric_summary_rows = [summarize_numeric(df[m], m) for m in main_numeric_metrics]

main_stats_df = pd.DataFrame(summary_rows + numeric_summary_rows)
main_stats_df.to_csv(OUTPUT_MAIN_STATS_CSV, index=False, encoding="utf-8-sig")
print(f"[OK] Saved main stats CSV: {OUTPUT_MAIN_STATS_CSV}")


# =========================================================
# 11. 状态分布统计
# =========================================================
status_counts_df = value_counts_with_share(df["status_std"], col_name="status")
status_counts_df.to_csv(OUTPUT_STATUS_COUNTS_CSV, index=False, encoding="utf-8-sig")
print(f"[OK] Saved status counts CSV: {OUTPUT_STATUS_COUNTS_CSV}")


# =========================================================
# 12. 分组统计：按年
# =========================================================
if year_col is not None:
    df["_year_for_stats"] = pd.to_numeric(df[year_col], errors="coerce")
    by_year_df = build_group_stats(
        df[df["_year_for_stats"].notna()].groupby("_year_for_stats"),
        "year"
    )
    if not by_year_df.empty:
        by_year_df["year"] = by_year_df["year"].astype(int)
        by_year_df = by_year_df.sort_values("year")
    by_year_df.to_csv(OUTPUT_BY_YEAR_CSV, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved by-year stats CSV: {OUTPUT_BY_YEAR_CSV}")
else:
    print("[WARN] 没找到 year/publication_year 列，跳过按年统计。")


# =========================================================
# 13. 分组统计：按类型
# =========================================================
if type_col is not None:
    df["_type_for_stats"] = df[type_col].fillna("").astype(str).str.strip()
    by_type_df = build_group_stats(
        df.groupby("_type_for_stats", dropna=False),
        "type"
    )
    by_type_df = by_type_df.sort_values("n", ascending=False)
    by_type_df.to_csv(OUTPUT_BY_TYPE_CSV, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved by-type stats CSV: {OUTPUT_BY_TYPE_CSV}")
else:
    print("[WARN] 没找到 type/work_type 列，跳过按类型统计。")


# =========================================================
# 14. 分组统计：按来源
# =========================================================
if source_col is not None:
    df["_source_for_stats"] = df[source_col].fillna("").astype(str).str.strip()
    by_source_df = build_group_stats(
        df.groupby("_source_for_stats", dropna=False),
        "source"
    )
    by_source_df = by_source_df.sort_values("n", ascending=False)
    by_source_df.to_csv(OUTPUT_BY_SOURCE_CSV, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved by-source stats CSV: {OUTPUT_BY_SOURCE_CSV}")
else:
    print("[WARN] 没找到 source 列，跳过按来源统计。")


# =========================================================
# 15. 诊断表
# =========================================================
diagnostic_cols = []

for c in [work_id_col, status_col, year_col, type_col, source_col]:
    if c is not None and c in df.columns and c not in diagnostic_cols:
        diagnostic_cols.append(c)

diagnostic_cols += [
    "status_std",
    "has_any_openalex_metrics",
    "has_openalex_references",
    "usable_for_count_analysis",
    "composition_metrics_eligible",
    "usable_for_composition_analysis",
    "openalex_matched_reference_count",
    "log1p_openalex_matched_reference_count",
    "openalex_topic_resolved_reference_count",
    "openalex_topic_missing_reference_count",
    "openalex_reference_field_diversity_raw",
    "openalex_reference_entropy_raw",
    "hss_share_among_matched_refs_raw",
    "cs_share_among_matched_refs_raw",
    "openalex_reference_field_diversity",
    "openalex_reference_entropy",
    "hss_share_among_matched_refs",
    "cs_share_among_matched_refs",
    "flag_blank_status",
    "flag_status_inferred",
    "flag_not_enriched",
    "flag_no_openalex_references",
    "flag_ok_count_only",
    "flag_ok_for_composition",
    "flag_count_missing",
    "flag_count_zero",
    "flag_count_positive",
    "flag_zero_but_raw_composition_not_missing",
    "flag_positive_count_but_no_topic_resolved",
    "flag_topic_resolved_gt_total",
    "flag_topic_missing_gt_total",
]

diagnostic_cols = [c for c in diagnostic_cols if c in df.columns]

diagnostic_df = df[diagnostic_cols].copy()

# 把更可疑的行排前面
sort_cols = [
    "flag_count_missing",
    "flag_zero_but_raw_composition_not_missing",
    "flag_positive_count_but_no_topic_resolved",
    "flag_topic_resolved_gt_total",
    "flag_topic_missing_gt_total",
]
sort_cols = [c for c in sort_cols if c in diagnostic_df.columns]
if sort_cols:
    diagnostic_df = diagnostic_df.sort_values(sort_cols, ascending=False)

diagnostic_df.to_csv(OUTPUT_DIAGNOSTIC_CSV, index=False, encoding="utf-8-sig")
print(f"[OK] Saved diagnostic CSV: {OUTPUT_DIAGNOSTIC_CSV}")


# =========================================================
# 16. 文本摘要
# =========================================================
lines = []
lines.append("OpenAlex observable reference metrics summary")
lines.append("=" * 70)
lines.append(f"Input file: {INPUT_CSV}")
lines.append(f"Total rows: {len(df):,}")
lines.append("")
lines.append("Detected columns")
lines.append("-" * 70)
lines.append(f"status_col = {status_col}")
lines.append(f"year_col = {year_col}")
lines.append(f"type_col = {type_col}")
lines.append(f"source_col = {source_col}")
lines.append(f"work_id_col = {work_id_col}")
lines.append(f"reference_count_total_col = {reference_count_total_col}")
lines.append(f"reference_count_with_primary_topic_col = {reference_count_with_primary_topic_col}")
lines.append(f"reference_count_missing_primary_topic_col = {reference_count_missing_primary_topic_col}")
lines.append("")

lines.append("Status distribution")
lines.append("-" * 70)
for _, r in status_counts_df.iterrows():
    lines.append(f"{str(r['status']):>28} : {int(r['count']):>10,} ({r['share']:.4%})")
lines.append("")

lines.append("Core diagnostics")
lines.append("-" * 70)
for row in summary_rows:
    lines.append(f"{row['metric']:<45} : {format_num(row['value'])}")
lines.append("")

lines.append("Main numeric stats")
lines.append("-" * 70)
for m in main_numeric_metrics:
    s = summarize_numeric(df[m], m)
    lines.append(
        f"{m}: "
        f"n={s['n_non_missing']:,}, "
        f"mean={format_num(s['mean'])}, "
        f"median={format_num(s['median'])}, "
        f"p25={format_num(s['p25'])}, "
        f"p75={format_num(s['p75'])}, "
        f"max={format_num(s['max'])}"
    )
lines.append("")

lines.append("Interpretation notes")
lines.append("-" * 70)
lines.append("1. 这里的 reference count 是 OpenAlex 可观测/可匹配参考文献数，不等于论文真实参考文献总数。")
lines.append("2. 对 count=0 或 topic_resolved_count 不足的行，entropy/diversity/share 这类组成指标记为 NA，而不是 0。")
lines.append(f"3. 组成指标只在 topic_resolved_reference_count >= {MIN_TOPIC_RESOLVED_FOR_COMPOSITION} 的样本里进入分析。")
lines.append("4. 回归建议拆成三层：")
lines.append("   - has_openalex_references")
lines.append("   - log1p_openalex_matched_reference_count")
lines.append("   - 组成型指标（只在 usable_for_composition_analysis == 1 的子样本里）")
lines.append("")

with open(OUTPUT_SUMMARY_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"[OK] Saved summary TXT: {OUTPUT_SUMMARY_TXT}")


# =========================================================
# 17. 控制台打印核心结果
# =========================================================
print("\n" + "=" * 80)
print("核心结果")
print("=" * 80)

print("\n[Status distribution]")
print(status_counts_df.to_string(index=False))

print("\n[Main numeric stats]")
print(pd.DataFrame(numeric_summary_rows).to_string(index=False))

print("\n[Done]")
print("输出文件：")
print(" -", OUTPUT_ANALYSIS_READY_CSV)
print(" -", OUTPUT_SUMMARY_TXT)
print(" -", OUTPUT_STATUS_COUNTS_CSV)
print(" -", OUTPUT_MAIN_STATS_CSV)
print(" -", OUTPUT_DIAGNOSTIC_CSV)
print(" -", OUTPUT_COLUMNS_TXT)
if year_col is not None:
    print(" -", OUTPUT_BY_YEAR_CSV)
if type_col is not None:
    print(" -", OUTPUT_BY_TYPE_CSV)
if source_col is not None:
    print(" -", OUTPUT_BY_SOURCE_CSV)
