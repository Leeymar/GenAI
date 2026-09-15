# -*- coding: utf-8 -*-
r"""
Supplementary regression script for the GenAI paper project.

Purpose
-------
This script adds the most important follow-up checks after the core models:
1. Threshold model with year fixed effects.
2. Additional outcomes: Any_Citation and Log_Citations.
3. Post-estimation total-slope tests for stage and topic interactions.

Input
-----
E:\PythonProject\GenAI\data_process\derived\genai_regression_enriched_v2.csv

Outputs
-------
E:\PythonProject\GenAI\data_process\reg_results_supplement
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")


# =========================================================
# 0. Paths and global settings
# =========================================================
DATA_PATH = Path(r"E:\PythonProject\GenAI\data_process\derived\genai_regression_enriched_v2.csv")
OUT_DIR = Path(r"E:\PythonProject\GenAI\data_process\reg_results_supplement")
OUT_DIR.mkdir(parents=True, exist_ok=True)

STAGE_ORDER = ["Phase1", "Phase2", "Phase3"]
HSS_BIN_ORDER = ["Pure STEM", "Low HSS", "Medium HSS", "High HSS", "HSS-dominant"]

MAIN_TOPIC_VAR = "has_hss_concepts"
BROAD_TOPIC_VAR = "social_topic_title_concepts"

NUMERIC_COLS = [
    "team_hss_dna_pct",
    "HSS_Continuous",
    "total_authors",
    "log_team_size",
    "Prior_Knowledge_Stock",
    "log_prior_knowledge",
    "HSS_Diversity",
    "citations",
    "year",
    "RCR",
    "Log_RCR",
    "Hit_Rate_10_year",
    "Hit_Rate_10_year_cat",
    "Citation_Field_Diversity",
    "Citation_Entropy",
    "NonCS_Citation_Share",
    "HSS_Citing_Share",
    "Total_Citing_Profile_Count",
    "Reference_Count",
    "RefMeta_Matched",
    "RefMeta_Coverage",
    "Mean_Reference_Age",
    "Mean_Log_Ref_Citations",
    "Recent_Reference_Share_3y",
]


# =========================================================
# 1. Helpers
# =========================================================
def check_columns(df: pd.DataFrame, required_cols):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")



def load_and_prepare_data(path: Path) -> pd.DataFrame:
    print(f"Loading data: {path}")
    df = pd.read_csv(path, low_memory=False)

    required = [
        "clean_work_id", "stage", "arxiv_primary_category",
        "HSS_Continuous", "HSS_Bin", "Log_RCR",
        "has_hss_concepts", "HSS_Diversity",
        "log_team_size", "log_prior_knowledge",
        "citations", "year"
    ]
    check_columns(df, required)

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["stage"] = pd.Categorical(df["stage"], categories=STAGE_ORDER, ordered=True)
    df["HSS_Bin"] = pd.Categorical(df["HSS_Bin"], categories=HSS_BIN_ORDER, ordered=True)

    df[MAIN_TOPIC_VAR] = pd.to_numeric(df[MAIN_TOPIC_VAR], errors="coerce").fillna(0).astype(int)

    if "social_topic_title_concepts" in df.columns:
        df[BROAD_TOPIC_VAR] = pd.to_numeric(df["social_topic_title_concepts"], errors="coerce").fillna(0).astype(int)
    elif "Socially_Embedded_Topic" in df.columns:
        df[BROAD_TOPIC_VAR] = pd.to_numeric(df["Socially_Embedded_Topic"], errors="coerce").fillna(0).astype(int)
    else:
        df[BROAD_TOPIC_VAR] = 0

    # Additional outcomes
    df["Any_Citation"] = (df["citations"].fillna(0) > 0).astype(int)
    df["Log_Citations"] = np.log1p(df["citations"].fillna(0))

    if "HSS_Citing_Share" in df.columns:
        df["Any_HSS_Citer"] = ((df["citations"].fillna(0) > 0) & (df["HSS_Citing_Share"].fillna(0) > 0)).astype(int)
    else:
        df["Any_HSS_Citer"] = np.nan

    if "NonCS_Citation_Share" in df.columns:
        df["Any_NonCS_Citer"] = ((df["citations"].fillna(0) > 0) & (df["NonCS_Citation_Share"].fillna(0) > 0)).astype(int)
    else:
        df["Any_NonCS_Citer"] = np.nan

    # Safety restriction
    df = df[(df["HSS_Continuous"].isna()) | ((df["HSS_Continuous"] >= 0) & (df["HSS_Continuous"] <= 10))].copy()

    return df



def get_core_controls(topic_var: str, fe_mode: str = "stage") -> str:
    if fe_mode == "stage":
        fe_part = "C(stage) + C(arxiv_primary_category)"
    elif fe_mode == "year":
        fe_part = "C(year) + C(arxiv_primary_category)"
    else:
        raise ValueError("fe_mode must be 'stage' or 'year'")

    return f"{topic_var} + log_team_size + log_prior_knowledge + HSS_Diversity + {fe_part}"



def get_sample(df: pd.DataFrame, sample_name: str) -> pd.DataFrame:
    if sample_name == "main":
        return df.copy()

    if sample_name == "cited_only":
        return df[df["citations"] > 0].copy()

    raise ValueError(f"Unknown sample_name: {sample_name}")



def fit_model(formula: str, data: pd.DataFrame, model_type: str = "ols"):
    if model_type == "ols":
        return smf.ols(formula=formula, data=data, missing="drop").fit(cov_type="HC1")

    if model_type == "glm_binom":
        return smf.glm(
            formula=formula,
            data=data,
            family=sm.families.Binomial(),
            missing="drop"
        ).fit(cov_type="HC1")

    raise ValueError(f"Unknown model_type: {model_type}")



def tidy_result(result, model_name: str, model_type: str, sample_name: str, formula: str) -> pd.DataFrame:
    ci = result.conf_int()
    ci.columns = ["ci_low", "ci_high"]

    if hasattr(result, "tvalues"):
        stat_vals = result.tvalues
    elif hasattr(result, "zvalues"):
        stat_vals = result.zvalues
    else:
        stat_vals = pd.Series(np.nan, index=result.params.index)

    out = pd.DataFrame({
        "model_name": model_name,
        "model_type": model_type,
        "sample_name": sample_name,
        "formula": formula,
        "term": result.params.index,
        "coef": result.params.values,
        "std_err": result.bse.values,
        "stat": stat_vals.values,
        "p_value": result.pvalues.values,
        "ci_low": ci["ci_low"].values,
        "ci_high": ci["ci_high"].values,
        "nobs": float(result.nobs),
        "aic": getattr(result, "aic", np.nan),
        "bic": getattr(result, "bic", np.nan),
        "rsquared": getattr(result, "rsquared", np.nan),
        "rsquared_adj": getattr(result, "rsquared_adj", np.nan),
    })
    return out



def save_summary_text(result, model_name: str, out_dir: Path):
    out_path = out_dir / f"{model_name}_summary.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        try:
            f.write(result.summary().as_text())
        except Exception:
            f.write(str(result.summary()))



def run_and_save_model(model_name: str, formula: str, df: pd.DataFrame, sample_name: str, model_type: str, out_dir: Path):
    sample_df = get_sample(df, sample_name)
    result = fit_model(formula=formula, data=sample_df, model_type=model_type)
    coef_df = tidy_result(result, model_name, model_type, sample_name, formula)

    coef_df.to_csv(out_dir / f"{model_name}_coef.csv", index=False, encoding="utf-8-sig")
    save_summary_text(result, model_name, out_dir)

    key_terms = [t for t in result.params.index if ("HSS_Continuous" in t or "HSS_Bin" in t)]
    print("\n" + "=" * 80)
    print(f"Model: {model_name}")
    print(f"Type: {model_type}")
    print(f"Sample: {sample_name}")
    print(f"N = {int(result.nobs):,}")
    if hasattr(result, "rsquared"):
        print(f"R^2 = {result.rsquared:.4f}")
    print("Key terms:")
    for t in key_terms[:12]:
        print(f"  {t}: coef={result.params[t]:.4f}, se={result.bse[t]:.4f}, p={result.pvalues[t]:.4g}")

    return result, coef_df



def _scalar_from_test_attr(obj):
    arr = np.asarray(obj)
    if arr.size == 0:
        return np.nan
    return float(arr.reshape(-1)[0])



def linear_combo_test(result, hypothesis: str, label: str, group: str) -> dict:
    test_res = result.t_test(hypothesis)
    estimate = _scalar_from_test_attr(test_res.effect)
    se = _scalar_from_test_attr(test_res.sd)
    stat = _scalar_from_test_attr(getattr(test_res, "tvalue", getattr(test_res, "zvalue", np.nan)))
    p_value = _scalar_from_test_attr(test_res.pvalue)
    ci = np.asarray(test_res.conf_int())
    if ci.size >= 2:
        ci_low = float(ci.reshape(-1, 2)[0][0])
        ci_high = float(ci.reshape(-1, 2)[0][1])
    else:
        ci_low = np.nan
        ci_high = np.nan

    return {
        "group": group,
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



def run_slope_tests(df: pd.DataFrame, out_dir: Path):
    print("\nRunning post-estimation slope tests...")

    # Stage interaction model (same design as core M3)
    stage_formula = (
        "Log_RCR ~ HSS_Continuous * C(stage) + "
        f"{MAIN_TOPIC_VAR} + log_team_size + log_prior_knowledge + HSS_Diversity + C(arxiv_primary_category)"
    )
    stage_result = fit_model(stage_formula, df, model_type="ols")
    save_summary_text(stage_result, "SLOPE_Base_StageInteraction", out_dir)
    tidy_result(stage_result, "SLOPE_Base_StageInteraction", "ols", "main", stage_formula).to_csv(
        out_dir / "SLOPE_Base_StageInteraction_coef.csv", index=False, encoding="utf-8-sig"
    )

    stage_tests = [
        linear_combo_test(stage_result, "HSS_Continuous = 0", "Phase1_total_slope", "stage_total_slope"),
        linear_combo_test(
            stage_result,
            "HSS_Continuous + HSS_Continuous:C(stage)[T.Phase2] = 0",
            "Phase2_total_slope",
            "stage_total_slope",
        ),
        linear_combo_test(
            stage_result,
            "HSS_Continuous + HSS_Continuous:C(stage)[T.Phase3] = 0",
            "Phase3_total_slope",
            "stage_total_slope",
        ),
        linear_combo_test(stage_result, "HSS_Continuous:C(stage)[T.Phase2] = 0", "Phase2_minus_Phase1", "stage_difference"),
        linear_combo_test(stage_result, "HSS_Continuous:C(stage)[T.Phase3] = 0", "Phase3_minus_Phase1", "stage_difference"),
        linear_combo_test(
            stage_result,
            "HSS_Continuous:C(stage)[T.Phase3] - HSS_Continuous:C(stage)[T.Phase2] = 0",
            "Phase3_minus_Phase2",
            "stage_difference",
        ),
    ]
    stage_tests_df = pd.DataFrame(stage_tests)
    stage_tests_df.to_csv(out_dir / "stage_slope_tests.csv", index=False, encoding="utf-8-sig")

    # Topic interaction model (same design as core M4)
    topic_formula = (
        f"Log_RCR ~ HSS_Continuous * {MAIN_TOPIC_VAR} + "
        "log_team_size + log_prior_knowledge + HSS_Diversity + C(stage) + C(arxiv_primary_category)"
    )
    topic_result = fit_model(topic_formula, df, model_type="ols")
    save_summary_text(topic_result, "SLOPE_Base_TopicInteraction", out_dir)
    tidy_result(topic_result, "SLOPE_Base_TopicInteraction", "ols", "main", topic_formula).to_csv(
        out_dir / "SLOPE_Base_TopicInteraction_coef.csv", index=False, encoding="utf-8-sig"
    )

    topic_tests = [
        linear_combo_test(topic_result, "HSS_Continuous = 0", "No_HSS_concepts_total_slope", "topic_total_slope"),
        linear_combo_test(
            topic_result,
            "HSS_Continuous + HSS_Continuous:has_hss_concepts = 0",
            "Has_HSS_concepts_total_slope",
            "topic_total_slope",
        ),
        linear_combo_test(
            topic_result,
            "HSS_Continuous:has_hss_concepts = 0",
            "Has_HSS_concepts_minus_No_HSS_concepts",
            "topic_difference",
        ),
    ]
    topic_tests_df = pd.DataFrame(topic_tests)
    topic_tests_df.to_csv(out_dir / "topic_slope_tests.csv", index=False, encoding="utf-8-sig")

    print("Saved stage slope tests to:", out_dir / "stage_slope_tests.csv")
    print("Saved topic slope tests to:", out_dir / "topic_slope_tests.csv")



def make_descriptive_tables(df: pd.DataFrame, out_dir: Path):
    desc_cols = [
        "HSS_Continuous", "Any_Citation", "Log_Citations", "citations", "Log_RCR",
        "log_team_size", "log_prior_knowledge", "HSS_Diversity"
    ]
    desc_cols = [c for c in desc_cols if c in df.columns]
    df[desc_cols].describe().T.to_csv(out_dir / "supplement_descriptive_stats.csv", encoding="utf-8-sig")


# =========================================================
# 2. Load data
# =========================================================
df = load_and_prepare_data(DATA_PATH)
print("\nSupplement data snapshot:")
print(f"Rows: {len(df):,}")
print(f"Any_Citation = 1 rows: {df['Any_Citation'].sum():,}")
print(f"Cited-only rows: {(df['citations'] > 0).sum():,}")
make_descriptive_tables(df, OUT_DIR)

core_controls_stage = get_core_controls(topic_var=MAIN_TOPIC_VAR, fe_mode="stage")
core_controls_year = get_core_controls(topic_var=MAIN_TOPIC_VAR, fe_mode="year")


# =========================================================
# 3. Supplement model specs
# =========================================================
model_specs = [
    {
        "model_name": "S1_LogRCR_Threshold_YearFE",
        "formula": f"Log_RCR ~ C(HSS_Bin, Treatment(reference='Pure STEM')) + {core_controls_year}",
        "sample_name": "main",
        "model_type": "ols",
    },
    {
        "model_name": "S2_AnyCitation_Continuous_LPM",
        "formula": f"Any_Citation ~ HSS_Continuous + {core_controls_stage}",
        "sample_name": "main",
        "model_type": "ols",
    },
    {
        "model_name": "S2b_AnyCitation_Threshold_LPM",
        "formula": f"Any_Citation ~ C(HSS_Bin, Treatment(reference='Pure STEM')) + {core_controls_stage}",
        "sample_name": "main",
        "model_type": "ols",
    },
    {
        "model_name": "S3_AnyCitation_Continuous_GLMLogit",
        "formula": f"Any_Citation ~ HSS_Continuous + {core_controls_stage}",
        "sample_name": "main",
        "model_type": "glm_binom",
    },
    {
        "model_name": "S4_LogCitations_Continuous_StageFE",
        "formula": f"Log_Citations ~ HSS_Continuous + {core_controls_stage}",
        "sample_name": "main",
        "model_type": "ols",
    },
    {
        "model_name": "S4b_LogCitations_Threshold_StageFE",
        "formula": f"Log_Citations ~ C(HSS_Bin, Treatment(reference='Pure STEM')) + {core_controls_stage}",
        "sample_name": "main",
        "model_type": "ols",
    },
]

# Optional extensive-margin style diffusion outcomes if columns exist
if "Any_HSS_Citer" in df.columns and df["Any_HSS_Citer"].notna().any():
    model_specs.append({
        "model_name": "S5_AnyHSSCiter_Continuous_LPM",
        "formula": f"Any_HSS_Citer ~ HSS_Continuous + {core_controls_stage}",
        "sample_name": "main",
        "model_type": "ols",
    })

if "Any_NonCS_Citer" in df.columns and df["Any_NonCS_Citer"].notna().any():
    model_specs.append({
        "model_name": "S6_AnyNonCSCiter_Continuous_LPM",
        "formula": f"Any_NonCS_Citer ~ HSS_Continuous + {core_controls_stage}",
        "sample_name": "main",
        "model_type": "ols",
    })


# =========================================================
# 4. Run models
# =========================================================
all_coef_tables = []
model_registry = []

for spec in model_specs:
    try:
        result, coef_df = run_and_save_model(
            model_name=spec["model_name"],
            formula=spec["formula"],
            df=df,
            sample_name=spec["sample_name"],
            model_type=spec["model_type"],
            out_dir=OUT_DIR,
        )
        all_coef_tables.append(coef_df)
        model_registry.append({
            "model_name": spec["model_name"],
            "model_type": spec["model_type"],
            "sample_name": spec["sample_name"],
            "formula": spec["formula"],
            "nobs": float(result.nobs),
            "rsquared": getattr(result, "rsquared", np.nan),
            "rsquared_adj": getattr(result, "rsquared_adj", np.nan),
            "aic": getattr(result, "aic", np.nan),
            "bic": getattr(result, "bic", np.nan),
            "error": "",
        })
    except Exception as e:
        print("\n" + "!" * 80)
        print(f"Model failed: {spec['model_name']}")
        print(f"Reason: {e}")
        print("!" * 80)
        model_registry.append({
            "model_name": spec["model_name"],
            "model_type": spec["model_type"],
            "sample_name": spec["sample_name"],
            "formula": spec["formula"],
            "nobs": np.nan,
            "rsquared": np.nan,
            "rsquared_adj": np.nan,
            "aic": np.nan,
            "bic": np.nan,
            "error": str(e),
        })

run_slope_tests(df, OUT_DIR)

pd.DataFrame(model_registry).to_csv(OUT_DIR / "model_registry_supplement.csv", index=False, encoding="utf-8-sig")
if all_coef_tables:
    pd.concat(all_coef_tables, ignore_index=True).to_csv(
        OUT_DIR / "all_model_coefficients_long_supplement.csv",
        index=False,
        encoding="utf-8-sig",
    )

print("\n" + "=" * 80)
print("Supplement regressions complete.")
print(f"Results saved to: {OUT_DIR}")
print("Key files to inspect:")
print("  - S1_LogRCR_Threshold_YearFE_coef.csv")
print("  - S2_AnyCitation_Continuous_LPM_coef.csv")
print("  - S4_LogCitations_Continuous_StageFE_coef.csv")
print("  - stage_slope_tests.csv")
print("  - topic_slope_tests.csv")
print("=" * 80)
