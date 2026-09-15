# -*- coding: utf-8 -*-
"""
mechanism_regressions_openalex.py

用途：
1. 在 GenAI 论文样本内部，检验团队 HSS 浓度与“知识整合”机制变量的关系
2. 明确区分：
   - 全样本上的可观测性/进入机制样本选择
   - 机制样本（usable_for_composition_analysis == 1）上的组成型指标回归
3. 自动识别你这份数据里常见的 HSS 变量、控制变量和固定效应变量
4. 输出可直接写论文的系数表、模型信息和完整 summary

默认主解释变量：
- team_hss_dna_pct （优先）
- HSS_Continuous （备选）

默认机制因变量：
- openalex_reference_field_diversity
- openalex_reference_entropy
- hss_share_among_matched_refs
- cs_share_among_matched_refs

前置选择模型：
- has_any_openalex_metrics
- usable_for_composition_analysis

默认控制变量：
- log_team_size
- log_prior_knowledge

默认固定效应：
- year
- arxiv_primary_category（若没有则回退到 arxiv_type）

估计方法：
- OLS + HC1 robust SE
- 二元因变量也先用 LPM（便于解释；需要时可再加 Logit 稳健性）

使用方法：
- 直接改 INPUT_CSV 后运行
- 若想强制指定主解释变量，把 HSS_VAR_MANUAL 改成真实列名
"""

import json
import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm

warnings.filterwarnings("ignore", category=RuntimeWarning)


# =========================================================
# 1. 路径与开关
# =========================================================
INPUT_CSV = r"openalex_reference_stats_output\genai_regression_enriched_openalex_observable.csv"

OUTPUT_DIR = r"mechanism_regression_output"
OUTPUT_COEF_CSV = os.path.join(OUTPUT_DIR, "mechanism_main_coefficients.csv")
OUTPUT_MODEL_INFO_CSV = os.path.join(OUTPUT_DIR, "model_info.csv")
OUTPUT_DESCRIPTIVES_CSV = os.path.join(OUTPUT_DIR, "sample_descriptives.csv")
OUTPUT_SUMMARY_TXT = os.path.join(OUTPUT_DIR, "regression_summaries.txt")
OUTPUT_SETTINGS_JSON = os.path.join(OUTPUT_DIR, "detected_settings.json")
OUTPUT_USED_DATA_CSV = os.path.join(OUTPUT_DIR, "used_variable_snapshot.csv")

# 手动指定主解释变量；若为 None 则自动识别
HSS_VAR_MANUAL = None

# 是否额外跑“+ log ref count 控制”的稳健性规格
RUN_REFCOUNT_ROBUSTNESS = False

# 如果你想强制用 arxiv_type 代替 arxiv_primary_category，可设为 True
PREFER_ARXIV_TYPE_FE = False

# 机制样本最小阈值；低于这个阈值就报错
MIN_MECH_SAMPLE = 200


# =========================================================
# 2. 候选列名
# =========================================================
HSS_VAR_CANDIDATES = [
    "team_hss_dna_pct",
    "HSS_Continuous",
]

HSS_BIN_CANDIDATES = [
    "HSS_Bin",
]

CONTROL_CANDIDATES = [
    "log_team_size",
    "log_prior_knowledge",
]

YEAR_FE_CANDIDATES = [
    "year",
    "publication_year_phase",
]

FIELD_FE_CANDIDATES_PRIMARY = [
    "arxiv_primary_category",
    "arxiv_type",
]

FIELD_FE_CANDIDATES_COARSE = [
    "arxiv_type",
    "arxiv_primary_category",
]

HAS_METRICS_CANDIDATES = [
    "has_any_openalex_metrics",
]

MECH_SAMPLE_CANDIDATES = [
    "usable_for_composition_analysis",
    "composition_metrics_eligible",
]

REFCOUNT_CONTROL_CANDIDATES = [
    "log1p_openalex_matched_reference_count",
    "openalex_matched_reference_count",
    "reference_count_total",
    "Reference_Count",
]

MECHANISM_OUTCOME_CANDIDATES = {
    "reference_field_diversity": [
        "openalex_reference_field_diversity",
        "reference_field_diversity",
    ],
    "reference_entropy": [
        "openalex_reference_entropy",
        "reference_entropy",
    ],
    "hss_reference_share": [
        "hss_share_among_matched_refs",
        "hss_reference_share",
    ],
    "cs_core_reference_share": [
        "cs_share_among_matched_refs",
        "cs_core_reference_share",
    ],
}


# =========================================================
# 3. 工具函数
# =========================================================
def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def pick_first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = list(df.columns)
    lower_map = {str(c).lower(): c for c in cols}

    for c in candidates:
        if c in df.columns:
            return c
    for c in candidates:
        lc = c.lower()
        if lc in lower_map:
            return lower_map[lc]
    return None


def to_numeric_safe(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def scale_share_variable(series: pd.Series) -> Tuple[pd.Series, str]:
    """
    将 HSS 浓度变量尽量统一到 [0,1] 区间，便于解释“10 个百分点”的效应。
    返回：缩放后的 series, unit_note
    """
    s = to_numeric_safe(series)
    nonmissing = s.dropna()
    if len(nonmissing) == 0:
        return s, "all_missing"

    min_v = float(nonmissing.min())
    max_v = float(nonmissing.max())

    if min_v >= 0 and max_v <= 1.000001:
        return s, "already_share_0_1"
    if min_v >= 0 and max_v <= 100.000001:
        return s / 100.0, "converted_from_percent_div100"
    return s, "kept_raw_scale"


def describe_series(series: pd.Series, name: str) -> Dict:
    s = to_numeric_safe(series).dropna()
    if len(s) == 0:
        return {
            "variable": name,
            "n_non_missing": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "p25": np.nan,
            "median": np.nan,
            "p75": np.nan,
            "max": np.nan,
        }
    return {
        "variable": name,
        "n_non_missing": int(s.notna().sum()),
        "mean": float(s.mean()),
        "std": float(s.std()) if len(s) > 1 else np.nan,
        "min": float(s.min()),
        "p25": float(s.quantile(0.25)),
        "median": float(s.median()),
        "p75": float(s.quantile(0.75)),
        "max": float(s.max()),
    }


def safe_unique_nonmissing(series: pd.Series) -> int:
    return int(series.dropna().nunique())


def build_formula(dv: str, controls: List[str], add_refcount: bool) -> str:
    terms = ["_hss_main"]
    terms.extend(controls)

    if add_refcount and "_log_refcount" in WORK_COLS:
        terms.append("_log_refcount")

    if "_year_fe" in WORK_COLS:
        terms.append("C(_year_fe)")
    if "_field_fe" in WORK_COLS:
        terms.append("C(_field_fe)")

    rhs = " + ".join(terms) if terms else "1"
    return f"{dv} ~ {rhs}"


def fit_ols_hc1(formula: str, data: pd.DataFrame):
    y, X = patsy.dmatrices(formula, data=data, return_type="dataframe")
    model = sm.OLS(y, X)
    res = model.fit(cov_type="HC1")
    return res, y, X


def extract_main_effect(
    result,
    dv_name: str,
    model_label: str,
    sample_label: str,
    formula: str,
    used_df: pd.DataFrame,
    add_refcount: bool,
    hss_scale_note: str,
) -> Dict:
    term = "_hss_main"

    coef = result.params.get(term, np.nan)
    se = result.bse.get(term, np.nan)
    tval = result.tvalues.get(term, np.nan)
    pval = result.pvalues.get(term, np.nan)

    ci = result.conf_int()
    if term in ci.index:
        ci_low = float(ci.loc[term, 0])
        ci_high = float(ci.loc[term, 1])
    else:
        ci_low = np.nan
        ci_high = np.nan

    y_sd = float(used_df[dv_name].std()) if used_df[dv_name].notna().sum() > 1 else np.nan
    x_sd = float(used_df[term].std()) if used_df[term].notna().sum() > 1 else np.nan
    std_beta = coef * x_sd / y_sd if pd.notna(coef) and pd.notna(x_sd) and pd.notna(y_sd) and y_sd != 0 else np.nan

    effect_10pp = coef * 0.10 if hss_scale_note in {"already_share_0_1", "converted_from_percent_div100"} else np.nan

    return {
        "model_label": model_label,
        "sample": sample_label,
        "dependent_variable": dv_name,
        "formula": formula,
        "hss_variable_used": HSS_VAR_USED,
        "hss_scale_note": hss_scale_note,
        "coef_hss": float(coef) if pd.notna(coef) else np.nan,
        "se_hss": float(se) if pd.notna(se) else np.nan,
        "t_hss": float(tval) if pd.notna(tval) else np.nan,
        "p_hss": float(pval) if pd.notna(pval) else np.nan,
        "ci_low_hss": ci_low,
        "ci_high_hss": ci_high,
        "effect_per_10pp_hss": float(effect_10pp) if pd.notna(effect_10pp) else np.nan,
        "std_beta_hss": float(std_beta) if pd.notna(std_beta) else np.nan,
        "nobs": int(result.nobs),
        "r2": float(result.rsquared) if hasattr(result, "rsquared") else np.nan,
        "dv_mean": float(used_df[dv_name].mean()) if used_df[dv_name].notna().sum() > 0 else np.nan,
        "dv_sd": float(used_df[dv_name].std()) if used_df[dv_name].notna().sum() > 1 else np.nan,
        "x_mean": float(used_df[term].mean()) if used_df[term].notna().sum() > 0 else np.nan,
        "x_sd": float(used_df[term].std()) if used_df[term].notna().sum() > 1 else np.nan,
        "includes_log_refcount_control": int(add_refcount),
    }


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    return str(obj)


# 用于 build_formula 检查列是否存在
WORK_COLS: List[str] = []
HSS_VAR_USED: Optional[str] = None


# =========================================================
# 4. 主流程
# =========================================================
def main():
    global WORK_COLS, HSS_VAR_USED

    ensure_dir(OUTPUT_DIR)

    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"找不到输入文件: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV, low_memory=False)
    print(f"[INFO] Loaded rows: {len(df):,}")
    print(f"[INFO] Loaded cols: {len(df.columns)}")

    # -------------------------
    # 自动识别列
    # -------------------------
    if HSS_VAR_MANUAL is not None:
        hss_col = HSS_VAR_MANUAL if HSS_VAR_MANUAL in df.columns else pick_first_existing_column(df, [HSS_VAR_MANUAL])
    else:
        hss_col = pick_first_existing_column(df, HSS_VAR_CANDIDATES)

    if hss_col is None:
        raise ValueError(
            "找不到团队 HSS 浓度变量。\n"
            f"请把 HSS_VAR_MANUAL 改成真实列名。候选默认值: {HSS_VAR_CANDIDATES}"
        )

    year_col = pick_first_existing_column(df, YEAR_FE_CANDIDATES)

    field_candidates = FIELD_FE_CANDIDATES_COARSE if PREFER_ARXIV_TYPE_FE else FIELD_FE_CANDIDATES_PRIMARY
    field_col = pick_first_existing_column(df, field_candidates)

    control_cols = [c for c in CONTROL_CANDIDATES if pick_first_existing_column(df, [c]) is not None]
    control_cols = [pick_first_existing_column(df, [c]) for c in control_cols]

    has_metrics_col = pick_first_existing_column(df, HAS_METRICS_CANDIDATES)
    mech_sample_col = pick_first_existing_column(df, MECH_SAMPLE_CANDIDATES)
    refcount_col = pick_first_existing_column(df, REFCOUNT_CONTROL_CANDIDATES)

    mechanism_cols = {}
    for short_name, candidates in MECHANISM_OUTCOME_CANDIDATES.items():
        found = pick_first_existing_column(df, candidates)
        if found is not None:
            mechanism_cols[short_name] = found

    required_mechs = [
        "reference_field_diversity",
        "reference_entropy",
        "hss_reference_share",
        "cs_core_reference_share",
    ]
    missing_required_mechs = [k for k in required_mechs if k not in mechanism_cols]
    if missing_required_mechs:
        raise ValueError(f"缺少机制变量列: {missing_required_mechs}")

    if mech_sample_col is None:
        raise ValueError("找不到 usable_for_composition_analysis / composition_metrics_eligible 列。")

    # -------------------------
    # 构造工作列
    # -------------------------
    work = df.copy()

    HSS_VAR_USED = hss_col
    work["_hss_main_raw"] = to_numeric_safe(work[hss_col])
    work["_hss_main"], hss_scale_note = scale_share_variable(work["_hss_main_raw"])

    if safe_unique_nonmissing(work["_hss_main"]) < 2:
        raise ValueError(f"HSS 变量 {hss_col} 在非缺失样本中几乎没有变化，无法回归。")

    if year_col is not None:
        work["_year_fe"] = work[year_col].astype(str).fillna("MISSING")
    if field_col is not None:
        work["_field_fe"] = work[field_col].astype(str).fillna("MISSING")

    for c in control_cols:
        work[c] = to_numeric_safe(work[c])

    if has_metrics_col is not None:
        work["has_any_openalex_metrics"] = to_numeric_safe(work[has_metrics_col])

    work["usable_for_composition_analysis"] = to_numeric_safe(work[mech_sample_col])

    # refcount control
    if refcount_col is not None:
        ref_s = to_numeric_safe(work[refcount_col])
        if str(refcount_col).lower().startswith("log1p_"):
            work["_log_refcount"] = ref_s
        else:
            work["_log_refcount"] = np.where(ref_s.notna(), np.log1p(ref_s.clip(lower=0)), np.nan)

    # outcomes numeric化
    for _, c in mechanism_cols.items():
        work[c] = to_numeric_safe(work[c])

    mech_df = work[work["usable_for_composition_analysis"] == 1].copy()
    if len(mech_df) < MIN_MECH_SAMPLE:
        raise ValueError(
            f"机制样本 usable_for_composition_analysis==1 只有 {len(mech_df)} 行，"
            f"低于阈值 {MIN_MECH_SAMPLE}。"
        )

    WORK_COLS = list(work.columns)

    # -------------------------
    # 描述统计
    # -------------------------
    desc_rows = []
    desc_rows.append({"variable": "N_total", "n_non_missing": len(work), "mean": len(work), "std": np.nan, "min": np.nan, "p25": np.nan, "median": np.nan, "p75": np.nan, "max": np.nan})
    desc_rows.append({"variable": "N_mechanism_sample", "n_non_missing": len(mech_df), "mean": len(mech_df), "std": np.nan, "min": np.nan, "p25": np.nan, "median": np.nan, "p75": np.nan, "max": np.nan})

    for v in ["_hss_main", "usable_for_composition_analysis"]:
        desc_rows.append(describe_series(work[v], v))

    if has_metrics_col is not None:
        desc_rows.append(describe_series(work["has_any_openalex_metrics"], "has_any_openalex_metrics"))

    for c in control_cols:
        desc_rows.append(describe_series(work[c], c))

    if "_log_refcount" in work.columns:
        desc_rows.append(describe_series(work["_log_refcount"], "_log_refcount"))

    for short_name, c in mechanism_cols.items():
        desc_rows.append(describe_series(mech_df[c], c))

    desc_df = pd.DataFrame(desc_rows)
    desc_df.to_csv(OUTPUT_DESCRIPTIVES_CSV, index=False, encoding="utf-8-sig")

    # 变量快照，方便人工检查
    snapshot_cols = [
        c for c in [
            "work_id",
            hss_col,
            "_hss_main",
            year_col,
            field_col,
            *control_cols,
            has_metrics_col,
            mech_sample_col,
            refcount_col,
            *mechanism_cols.values(),
        ] if c is not None and c in work.columns
    ]
    snapshot_cols = list(dict.fromkeys(snapshot_cols))
    work[snapshot_cols].head(200).to_csv(OUTPUT_USED_DATA_CSV, index=False, encoding="utf-8-sig")

    # -------------------------
    # 跑模型
    # -------------------------
    coef_rows = []
    model_info_rows = []
    summary_blocks = []

    model_specs = []

    # 0. 前置选择模型
    if has_metrics_col is not None:
        model_specs.append({
            "model_label": "M0_has_any_openalex_metrics",
            "sample_label": "full_sample",
            "dv": "has_any_openalex_metrics",
            "data": work,
            "add_refcount": False,
        })

    model_specs.append({
        "model_label": "M1_usable_for_composition_analysis",
        "sample_label": "full_sample",
        "dv": "usable_for_composition_analysis",
        "data": work,
        "add_refcount": False,
    })

    # 1. 机制主回归
    mech_label_map = {
        "reference_field_diversity": "M2_reference_field_diversity",
        "reference_entropy": "M3_reference_entropy",
        "hss_reference_share": "M4_hss_reference_share",
        "cs_core_reference_share": "M5_cs_core_reference_share",
    }

    for short_name, dv_col in mechanism_cols.items():
        if short_name not in mech_label_map:
            continue
        model_specs.append({
            "model_label": mech_label_map[short_name],
            "sample_label": "mechanism_sample",
            "dv": dv_col,
            "data": mech_df,
            "add_refcount": False,
        })

        if RUN_REFCOUNT_ROBUSTNESS and "_log_refcount" in work.columns:
            model_specs.append({
                "model_label": mech_label_map[short_name] + "_plus_log_refcount",
                "sample_label": "mechanism_sample",
                "dv": dv_col,
                "data": mech_df,
                "add_refcount": True,
            })

    for spec in model_specs:
        formula = build_formula(spec["dv"], control_cols, spec["add_refcount"])
        try:
            result, y, X = fit_ols_hc1(formula, spec["data"])
            used_df = spec["data"].loc[y.index].copy()

            coef_rows.append(
                extract_main_effect(
                    result=result,
                    dv_name=spec["dv"],
                    model_label=spec["model_label"],
                    sample_label=spec["sample_label"],
                    formula=formula,
                    used_df=used_df,
                    add_refcount=spec["add_refcount"],
                    hss_scale_note=hss_scale_note,
                )
            )

            model_info_rows.append({
                "model_label": spec["model_label"],
                "sample": spec["sample_label"],
                "dependent_variable": spec["dv"],
                "formula": formula,
                "nobs": int(result.nobs),
                "df_model": float(result.df_model),
                "df_resid": float(result.df_resid),
                "r2": float(result.rsquared) if hasattr(result, "rsquared") else np.nan,
                "adj_r2": float(result.rsquared_adj) if hasattr(result, "rsquared_adj") else np.nan,
                "fvalue": float(result.fvalue) if result.fvalue is not None else np.nan,
                "f_pvalue": float(result.f_pvalue) if result.f_pvalue is not None else np.nan,
                "cov_type": str(result.cov_type),
                "controls_used": ", ".join(control_cols),
                "year_fe_used": int("_year_fe" in WORK_COLS),
                "field_fe_used": int("_field_fe" in WORK_COLS),
                "log_refcount_control_used": int(spec["add_refcount"]),
            })

            summary_blocks.append("=" * 100)
            summary_blocks.append(spec["model_label"])
            summary_blocks.append(f"DV: {spec['dv']}")
            summary_blocks.append(f"Sample: {spec['sample_label']}")
            summary_blocks.append(f"Formula: {formula}")
            summary_blocks.append(str(result.summary()))
            summary_blocks.append("")

            print(f"[OK] {spec['model_label']} | n={int(result.nobs):,}")

        except Exception as e:
            model_info_rows.append({
                "model_label": spec["model_label"],
                "sample": spec["sample_label"],
                "dependent_variable": spec["dv"],
                "formula": formula,
                "nobs": np.nan,
                "df_model": np.nan,
                "df_resid": np.nan,
                "r2": np.nan,
                "adj_r2": np.nan,
                "fvalue": np.nan,
                "f_pvalue": np.nan,
                "cov_type": "HC1",
                "controls_used": ", ".join(control_cols),
                "year_fe_used": int("_year_fe" in WORK_COLS),
                "field_fe_used": int("_field_fe" in WORK_COLS),
                "log_refcount_control_used": int(spec["add_refcount"]),
                "error": str(e),
            })
            print(f"[FAIL] {spec['model_label']} -> {e}")

    coef_df = pd.DataFrame(coef_rows)
    model_info_df = pd.DataFrame(model_info_rows)

    coef_df.to_csv(OUTPUT_COEF_CSV, index=False, encoding="utf-8-sig")
    model_info_df.to_csv(OUTPUT_MODEL_INFO_CSV, index=False, encoding="utf-8-sig")

    with open(OUTPUT_SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_blocks))

    settings = {
        "input_csv": INPUT_CSV,
        "n_total_rows": int(len(work)),
        "n_mechanism_sample": int(len(mech_df)),
        "hss_var_manual": HSS_VAR_MANUAL,
        "hss_var_used": hss_col,
        "hss_scale_note": hss_scale_note,
        "year_fe_col": year_col,
        "field_fe_col": field_col,
        "control_cols": control_cols,
        "has_metrics_col": has_metrics_col,
        "mechanism_sample_col": mech_sample_col,
        "refcount_control_col": refcount_col,
        "mechanism_outcome_cols": mechanism_cols,
        "run_refcount_robustness": RUN_REFCOUNT_ROBUSTNESS,
        "prefer_arxiv_type_fe": PREFER_ARXIV_TYPE_FE,
    }
    with open(OUTPUT_SETTINGS_JSON, "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2, default=json_default)

    # -------------------------
    # 控制台打印核心结果
    # -------------------------
    print("\n" + "=" * 90)
    print("核心结果：团队 HSS 浓度 的主系数")
    print("=" * 90)

    if coef_df.empty:
        print("[WARN] 没有成功估计的模型，请查看 model_info.csv / regression_summaries.txt")
    else:
        show_cols = [
            "model_label",
            "sample",
            "dependent_variable",
            "coef_hss",
            "se_hss",
            "p_hss",
            "effect_per_10pp_hss",
            "nobs",
            "r2",
        ]
        show_cols = [c for c in show_cols if c in coef_df.columns]
        print(coef_df[show_cols].to_string(index=False))

    print("\n[Done] 输出文件：")
    print(" -", OUTPUT_COEF_CSV)
    print(" -", OUTPUT_MODEL_INFO_CSV)
    print(" -", OUTPUT_DESCRIPTIVES_CSV)
    print(" -", OUTPUT_SUMMARY_TXT)
    print(" -", OUTPUT_SETTINGS_JSON)
    print(" -", OUTPUT_USED_DATA_CSV)


if __name__ == "__main__":
    main()
