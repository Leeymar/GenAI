# -*- coding: utf-8 -*-
"""
Section 3.5 moderation regression script for the GenAI paper project.

估计模型：
    Y = α + β1*HSS_Continuous_p + β2*Integration_p
        + β3*(HSS_Continuous_p * Integration_p)
        + γ'X_p + τ_t + η_f + ε_p

其中：
    Y ∈ {Log_RCR, Hit_Rate_10_year_cat}
    Integration_p ∈ {
        reference_field_diversity,
        reference_entropy,
        hss_reference_share,
        cs_core_reference_share
    }
    X_p = {log_team_size, log_prior_knowledge}
    τ_t = 年份固定效应
    η_f = arXiv 主分类固定效应

说明：
1. 不修改你原文中的模型说法，代码严格按该式估计
2. HSS_Continuous 已经是 /10 后的变量，系数解释为团队 HSS 占比每增加 10 个百分点的边际变化
3. 标准误使用 HC1 robust SE
4. 参考文献样本筛选：
   - reference_count_total > 0
   - reference_topic_coverage >= 0.30
   若 reference_topic_coverage 缺失，则用
   reference_count_with_primary_topic / reference_count_total 现场补算
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")


# =========================================================
# 0. 路径设置
# =========================================================
DATA_PATH = Path(r"E:\PythonProject\GenAI\data_process\derived\genai_regression_enriched_v3_fixed.csv")
OUT_DIR = Path(r"E:\PythonProject\GenAI\data_process\reg_results_section35_moderation")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 1. 全局设置
# =========================================================
MIN_REFERENCE_TOPIC_COVERAGE = 0.30

NUMERIC_COLS = [
    "HSS_Continuous",
    "Log_RCR",
    "Hit_Rate_10_year_cat",
    "log_team_size",
    "log_prior_knowledge",
    "year",
    "reference_count_total",
    "reference_count_with_primary_topic",
    "reference_count_missing_primary_topic",
    "reference_count_unresolved",
    "reference_topic_coverage",
    "reference_field_diversity",
    "reference_entropy",
    "hss_reference_share",
    "cs_core_reference_share",
]

INTEGRATION_VAR_MAP = {
    "RefFieldDiversity": "reference_field_diversity",
    "RefEntropy": "reference_entropy",
    "HSSRefShare": "hss_reference_share",
    "CSCoreRefShare": "cs_core_reference_share",
}

OUTCOME_MAP = {
    "LogRCR": "Log_RCR",
    "Hit10YearCat": "Hit_Rate_10_year_cat",
}


# =========================================================
# 2. 工具函数
# =========================================================
def check_columns(df, required_cols):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要字段: {missing}")


def load_and_prepare_data(path: Path) -> pd.DataFrame:
    print(f"Loading data: {path}")
    df = pd.read_csv(path, low_memory=False)

    required = [
        "clean_work_id",
        "HSS_Continuous",
        "Log_RCR",
        "Hit_Rate_10_year_cat",
        "log_team_size",
        "log_prior_knowledge",
        "year",
        "arxiv_primary_category",
        "reference_count_total",
        "reference_count_with_primary_topic",
        "reference_field_diversity",
        "reference_entropy",
        "hss_reference_share",
        "cs_core_reference_share",
    ]
    check_columns(df, required)

    # 数值列转换
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # HSS_Continuous 合理区间限制
    if "HSS_Continuous" in df.columns:
        df = df[
            (df["HSS_Continuous"].isna()) |
            ((df["HSS_Continuous"] >= 0) & (df["HSS_Continuous"] <= 10))
        ].copy()

    # 二元因变量清理
    if "Hit_Rate_10_year_cat" in df.columns:
        df["Hit_Rate_10_year_cat"] = pd.to_numeric(df["Hit_Rate_10_year_cat"], errors="coerce")
        df.loc[~df["Hit_Rate_10_year_cat"].isin([0, 1]), "Hit_Rate_10_year_cat"] = np.nan

    # -----------------------------------------------------
    # 关键修正：
    # reference_topic_coverage 如果为空，就用计数字段现场补算
    # -----------------------------------------------------
    if {"reference_count_total", "reference_count_with_primary_topic"}.issubset(df.columns):
        calc_cov = np.where(
            df["reference_count_total"] > 0,
            df["reference_count_with_primary_topic"] / df["reference_count_total"],
            np.nan
        )

        if "reference_topic_coverage" not in df.columns:
            df["reference_topic_coverage"] = calc_cov
        else:
            df["reference_topic_coverage"] = pd.to_numeric(df["reference_topic_coverage"], errors="coerce")
            df["reference_topic_coverage"] = df["reference_topic_coverage"].fillna(pd.Series(calc_cov, index=df.index))

    # 限制 coverage 在 [0, 1]
    if "reference_topic_coverage" in df.columns:
        df.loc[
            ~df["reference_topic_coverage"].between(0, 1, inclusive="both"),
            "reference_topic_coverage"
        ] = np.nan

    return df


def get_integration_base_sample(df: pd.DataFrame) -> pd.DataFrame:
    """
    参考文献侧基础样本：
    - 至少有参考文献
    - topic coverage >= 0.30
    """
    sample = df.copy()

    sample = sample[
        (sample["reference_count_total"] > 0) &
        (sample["reference_topic_coverage"] >= MIN_REFERENCE_TOPIC_COVERAGE)
    ].copy()

    return sample


def get_estimation_df(df: pd.DataFrame, outcome: str, moderator: str) -> pd.DataFrame:
    needed = [
        outcome,
        "HSS_Continuous",
        moderator,
        "log_team_size",
        "log_prior_knowledge",
        "year",
        "arxiv_primary_category",
    ]
    d = df[needed].dropna().copy()
    return d


def fit_model(formula: str, data: pd.DataFrame):
    if len(data) == 0:
        raise ValueError("estimation sample is empty after filtering and dropna")
    return smf.ols(formula=formula, data=data, missing="drop").fit(cov_type="HC1")


def tidy_result(result, model_name: str, formula: str, moderator: str, outcome: str, sample_label: str) -> pd.DataFrame:
    ci = result.conf_int()
    ci.columns = ["ci_low", "ci_high"]

    out = pd.DataFrame({
        "model_name": model_name,
        "sample_name": sample_label,
        "outcome": outcome,
        "moderator": moderator,
        "formula": formula,
        "term": result.params.index,
        "coef": result.params.values,
        "std_err": result.bse.values,
        "stat": result.tvalues.values if hasattr(result, "tvalues") else np.nan,
        "p_value": result.pvalues.values,
        "ci_low": ci["ci_low"].values,
        "ci_high": ci["ci_high"].values,
        "nobs": float(result.nobs),
        "rsquared": getattr(result, "rsquared", np.nan),
        "rsquared_adj": getattr(result, "rsquared_adj", np.nan),
        "aic": getattr(result, "aic", np.nan),
        "bic": getattr(result, "bic", np.nan),
    })
    return out


def save_summary_text(result, model_name: str, out_dir: Path):
    out_path = out_dir / f"{model_name}_summary.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result.summary().as_text())


def _scalar_from_test_attr(obj):
    arr = np.asarray(obj)
    if arr.size == 0:
        return np.nan
    return float(arr.reshape(-1)[0])


def linear_combo_test(result, hypothesis: str, label: str, model_name: str, moderator: str, outcome: str) -> dict:
    test_res = result.t_test(hypothesis)

    estimate = _scalar_from_test_attr(test_res.effect)
    se = _scalar_from_test_attr(test_res.sd)
    stat = _scalar_from_test_attr(getattr(test_res, "tvalue", np.nan))
    p_value = _scalar_from_test_attr(test_res.pvalue)

    ci = np.asarray(test_res.conf_int())
    if ci.size >= 2:
        ci_low = float(ci.reshape(-1, 2)[0][0])
        ci_high = float(ci.reshape(-1, 2)[0][1])
    else:
        ci_low = np.nan
        ci_high = np.nan

    return {
        "model_name": model_name,
        "outcome": outcome,
        "moderator": moderator,
        "label": label,
        "hypothesis": hypothesis,
        "estimate": estimate,
        "std_err": se,
        "stat": stat,
        "p_value": p_value,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "nobs": float(result.nobs),
    }


def build_compact_key_table(all_coef_df: pd.DataFrame, out_dir: Path):
    keep_patterns = [
        "HSS_Continuous",
        "reference_field_diversity",
        "reference_entropy",
        "hss_reference_share",
        "cs_core_reference_share",
        "HSS_Continuous:reference_field_diversity",
        "HSS_Continuous:reference_entropy",
        "HSS_Continuous:hss_reference_share",
        "HSS_Continuous:cs_core_reference_share",
    ]

    mask = all_coef_df["term"].apply(lambda x: any(p in x for p in keep_patterns))
    key_df = all_coef_df[mask].copy()

    def fmt_row(r):
        return f"{r['coef']:.4f} ({r['std_err']:.4f})"

    key_df["coef_se"] = key_df.apply(fmt_row, axis=1)

    compact = key_df.pivot_table(
        index="term",
        columns="model_name",
        values="coef_se",
        aggfunc="first"
    )
    compact.to_csv(out_dir / "compact_key_coefficients.csv", encoding="utf-8-sig")


def make_descriptive_tables(df_all: pd.DataFrame, df_base: pd.DataFrame, out_dir: Path):
    base_summary = pd.DataFrame({
        "metric": [
            "rows_all",
            "rows_integration_base",
            "nonmissing_reference_field_diversity_all",
            "nonmissing_reference_entropy_all",
            "nonmissing_hss_reference_share_all",
            "nonmissing_cs_core_reference_share_all",
            "nonmissing_reference_field_diversity_base",
            "nonmissing_reference_entropy_base",
            "nonmissing_hss_reference_share_base",
            "nonmissing_cs_core_reference_share_base",
        ],
        "value": [
            len(df_all),
            len(df_base),
            df_all["reference_field_diversity"].notna().sum(),
            df_all["reference_entropy"].notna().sum(),
            df_all["hss_reference_share"].notna().sum(),
            df_all["cs_core_reference_share"].notna().sum(),
            df_base["reference_field_diversity"].notna().sum(),
            df_base["reference_entropy"].notna().sum(),
            df_base["hss_reference_share"].notna().sum(),
            df_base["cs_core_reference_share"].notna().sum(),
        ]
    })
    base_summary.to_csv(out_dir / "integration_base_sample_summary.csv", index=False, encoding="utf-8-sig")

    desc_cols = [
        "HSS_Continuous",
        "Log_RCR",
        "Hit_Rate_10_year_cat",
        "log_team_size",
        "log_prior_knowledge",
        "reference_topic_coverage",
        "reference_field_diversity",
        "reference_entropy",
        "hss_reference_share",
        "cs_core_reference_share",
    ]
    desc_cols = [c for c in desc_cols if c in df_base.columns]
    if desc_cols:
        df_base[desc_cols].describe().T.to_csv(
            out_dir / "integration_base_descriptive_stats.csv",
            encoding="utf-8-sig"
        )


def run_and_save_model(model_name: str, outcome: str, moderator: str, base_df: pd.DataFrame, out_dir: Path):
    formula = (
        f"{outcome} ~ HSS_Continuous * {moderator} + "
        "log_team_size + log_prior_knowledge + C(year) + C(arxiv_primary_category)"
    )

    estimation_df = get_estimation_df(base_df, outcome=outcome, moderator=moderator)

    if len(estimation_df) == 0:
        raise ValueError(f"no usable rows for outcome={outcome}, moderator={moderator}")

    result = fit_model(formula=formula, data=estimation_df)
    coef_df = tidy_result(
        result=result,
        model_name=model_name,
        formula=formula,
        moderator=moderator,
        outcome=outcome,
        sample_label="integration_base"
    )

    coef_df.to_csv(out_dir / f"{model_name}_coef.csv", index=False, encoding="utf-8-sig")
    save_summary_text(result, model_name, out_dir)

    interaction_term = f"HSS_Continuous:{moderator}"
    print("\n" + "=" * 90)
    print(f"Model: {model_name}")
    print(f"Outcome: {outcome}")
    print(f"Moderator: {moderator}")
    print(f"N = {int(result.nobs):,}")
    print(f"R^2 = {getattr(result, 'rsquared', np.nan):.4f}")

    for t in ["HSS_Continuous", moderator, interaction_term]:
        if t in result.params.index:
            print(f"  {t}: coef={result.params[t]:.4f}, se={result.bse[t]:.4f}, p={result.pvalues[t]:.4g}")

    return result, coef_df, estimation_df


def run_simple_slopes(result, model_name: str, outcome: str, moderator: str, estimation_df: pd.DataFrame) -> pd.DataFrame:
    qs = estimation_df[moderator].quantile([0.25, 0.50, 0.75]).to_dict()
    interaction_term = f"HSS_Continuous:{moderator}"

    tests = [
        linear_combo_test(
            result,
            "HSS_Continuous = 0",
            "Slope_at_Moderator_0",
            model_name,
            moderator,
            outcome
        )
    ]

    for q, val in qs.items():
        label = f"Slope_at_{int(q * 100)}th_percentile"
        hyp = f"HSS_Continuous + ({val})*{interaction_term} = 0"
        tests.append(
            linear_combo_test(
                result,
                hyp,
                label,
                model_name,
                moderator,
                outcome
            )
        )

    out = pd.DataFrame(tests)
    out["moderator_p25"] = qs.get(0.25, np.nan)
    out["moderator_p50"] = qs.get(0.50, np.nan)
    out["moderator_p75"] = qs.get(0.75, np.nan)
    return out


# =========================================================
# 3. 读取并准备数据
# =========================================================
df = load_and_prepare_data(DATA_PATH)
df_base = get_integration_base_sample(df)

print("\nBasic data snapshot:")
print(f"Rows (all): {len(df):,}")
print(f"Rows (integration base sample): {len(df_base):,}")

print("\nIntegration variable check:")
for short_name, var in INTEGRATION_VAR_MAP.items():
    print(f"  {short_name:<18} -> {var:<28} | non-missing(all) = {df[var].notna().sum():,} | non-missing(base) = {df_base[var].notna().sum():,}")

print("\nIntegration base sample summary:")
print(
    df_base[
        [
            "reference_topic_coverage",
            "reference_field_diversity",
            "reference_entropy",
            "hss_reference_share",
            "cs_core_reference_share",
        ]
    ].describe().T
)

make_descriptive_tables(df, df_base, OUT_DIR)


# =========================================================
# 4. 设定模型
# =========================================================
model_specs = []

for outcome_short, outcome_var in OUTCOME_MAP.items():
    for mod_short, mod_var in INTEGRATION_VAR_MAP.items():
        model_specs.append({
            "model_name": f"M35_{outcome_short}_{mod_short}_Moderation",
            "outcome": outcome_var,
            "moderator": mod_var,
        })


# =========================================================
# 5. 执行模型
# =========================================================
all_coef_tables = []
all_simple_slopes = []
model_registry = []

for spec in model_specs:
    try:
        result, coef_df, estimation_df = run_and_save_model(
            model_name=spec["model_name"],
            outcome=spec["outcome"],
            moderator=spec["moderator"],
            base_df=df_base,
            out_dir=OUT_DIR
        )
        all_coef_tables.append(coef_df)

        slope_df = run_simple_slopes(
            result=result,
            model_name=spec["model_name"],
            outcome=spec["outcome"],
            moderator=spec["moderator"],
            estimation_df=estimation_df
        )
        slope_df.to_csv(OUT_DIR / f"{spec['model_name']}_simple_slopes.csv", index=False, encoding="utf-8-sig")
        all_simple_slopes.append(slope_df)

        model_registry.append({
            "model_name": spec["model_name"],
            "outcome": spec["outcome"],
            "moderator": spec["moderator"],
            "base_sample_rows": len(df_base),
            "estimation_rows": len(estimation_df),
            "nobs": float(result.nobs),
            "rsquared": getattr(result, "rsquared", np.nan),
            "rsquared_adj": getattr(result, "rsquared_adj", np.nan),
            "aic": getattr(result, "aic", np.nan),
            "bic": getattr(result, "bic", np.nan),
            "error": "",
        })

    except Exception as e:
        print("\n" + "!" * 90)
        print(f"Model failed: {spec['model_name']}")
        print(f"Reason: {e}")
        print("!" * 90)

        model_registry.append({
            "model_name": spec["model_name"],
            "outcome": spec["outcome"],
            "moderator": spec["moderator"],
            "base_sample_rows": len(df_base),
            "estimation_rows": np.nan,
            "nobs": np.nan,
            "rsquared": np.nan,
            "rsquared_adj": np.nan,
            "aic": np.nan,
            "bic": np.nan,
            "error": str(e),
        })


# =========================================================
# 6. 汇总输出
# =========================================================
model_registry_df = pd.DataFrame(model_registry)
model_registry_df.to_csv(OUT_DIR / "model_registry.csv", index=False, encoding="utf-8-sig")

if all_coef_tables:
    all_coef_df = pd.concat(all_coef_tables, ignore_index=True)
    all_coef_df.to_csv(OUT_DIR / "all_model_coefficients_long.csv", index=False, encoding="utf-8-sig")
    build_compact_key_table(all_coef_df, OUT_DIR)

if all_simple_slopes:
    pd.concat(all_simple_slopes, ignore_index=True).to_csv(
        OUT_DIR / "all_simple_slopes.csv",
        index=False,
        encoding="utf-8-sig"
    )

print("\n" + "=" * 90)
print("Section 3.5 moderation regressions complete.")
print(f"Results saved to: {OUT_DIR}")
print("Key files to inspect:")
print("  - model_registry.csv")
print("  - integration_base_sample_summary.csv")
print("  - integration_base_descriptive_stats.csv")
print("  - all_model_coefficients_long.csv")
print("  - compact_key_coefficients.csv")
print("  - all_simple_slopes.csv")
for outcome_short in OUTCOME_MAP.keys():
    for mod_short in INTEGRATION_VAR_MAP.keys():
        print(f"  - M35_{outcome_short}_{mod_short}_Moderation_coef.csv")
print("=" * 90)
