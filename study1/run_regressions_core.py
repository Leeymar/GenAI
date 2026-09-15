# -*- coding: utf-8 -*-
"""
Core regression script for GenAI paper project

主输入文件：
    E:\PythonProject\GenAI\data_process\derived\genai_regression_enriched_v2.csv

主输出目录：
    E:\PythonProject\GenAI\data_process\reg_results_core

说明：
1. 主主题控制变量：has_hss_concepts（严格按 concepts 判断）
2. Socially_Embedded_Topic 仅作为扩展稳健性变量
3. HSS_Continuous = team_hss_dna_pct / 10
   => 系数解释为：团队 HSS 比例每增加 10 个百分点的边际变化
4. 标准误默认使用 HC1 robust SE
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy.contrasts import Treatment

warnings.filterwarnings("ignore")


# =========================================================
# 0. 路径设置
# =========================================================
DATA_PATH = Path(r"E:\PythonProject\GenAI\data_process\derived\genai_regression_enriched_v2.csv")
OUT_DIR = Path(r"E:\PythonProject\GenAI\data_process\reg_results_core")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 是否运行扩展稳健性
RUN_BROAD_TOPIC_ROBUSTNESS = True   # 用 Socially_Embedded_Topic 替代 has_hss_concepts
RUN_REFERENCE_APPENDIX = False      # 参考文献机制目前覆盖率偏低，默认不跑主文附录


# =========================================================
# 1. 基础设置
# =========================================================
STAGE_ORDER = ["Phase1", "Phase2", "Phase3"]
HSS_BIN_ORDER = ["Pure STEM", "Low HSS", "Medium HSS", "High HSS", "HSS-dominant"]

MAIN_TOPIC_VAR = "has_hss_concepts"                 # 严格按 concepts
BROAD_TOPIC_VAR = "social_topic_title_concepts"     # title + concepts 扩展版本

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
        "clean_work_id", "stage", "arxiv_primary_category",
        "HSS_Continuous", "HSS_Bin", "Log_RCR",
        "Hit_Rate_10_year", "Hit_Rate_10_year_cat",
        "has_hss_concepts", "HSS_Diversity",
        "log_team_size", "log_prior_knowledge",
        "Citation_Field_Diversity", "Citation_Entropy",
        "NonCS_Citation_Share", "HSS_Citing_Share",
        "citations", "citing_missing_flag"
    ]
    check_columns(df, required)

    # 数值列
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 类别列
    df["stage"] = pd.Categorical(df["stage"], categories=STAGE_ORDER, ordered=True)
    df["HSS_Bin"] = pd.Categorical(df["HSS_Bin"], categories=HSS_BIN_ORDER, ordered=True)

    # 严格 concepts 主题变量
    df[MAIN_TOPIC_VAR] = pd.to_numeric(df[MAIN_TOPIC_VAR], errors="coerce").fillna(0).astype(int)

    # 扩展主题变量：title + concepts
    if "social_topic_title_concepts" in df.columns:
        df[BROAD_TOPIC_VAR] = pd.to_numeric(df["social_topic_title_concepts"], errors="coerce").fillna(0).astype(int)
    elif "Socially_Embedded_Topic" in df.columns:
        df[BROAD_TOPIC_VAR] = pd.to_numeric(df["Socially_Embedded_Topic"], errors="coerce").fillna(0).astype(int)
    else:
        df[BROAD_TOPIC_VAR] = 0

    # 保险起见，限制合理区间
    if "HSS_Continuous" in df.columns:
        df = df[(df["HSS_Continuous"].isna()) | ((df["HSS_Continuous"] >= 0) & (df["HSS_Continuous"] <= 10))].copy()

    # 二元变量清理
    for col in ["Hit_Rate_10_year", "Hit_Rate_10_year_cat"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[~df[col].isin([0, 1]), col] = np.nan

    return df


def get_core_controls(topic_var: str, fe_mode: str = "stage") -> str:
    """
    主控制项：
    - 主题控制：topic_var
    - 团队规模：log_team_size
    - 前沿知识存量：log_prior_knowledge
    - 团队内部 HSS 多样性：HSS_Diversity
    - 固定效应：stage 或 year + arxiv_primary_category
    """
    if fe_mode == "stage":
        fe_part = "C(stage) + C(arxiv_primary_category)"
    elif fe_mode == "year":
        fe_part = "C(year) + C(arxiv_primary_category)"
    else:
        raise ValueError("fe_mode 只能是 'stage' 或 'year'")

    return f"{topic_var} + log_team_size + log_prior_knowledge + HSS_Diversity + {fe_part}"


def get_diffusion_controls(topic_var: str, fe_mode: str = "stage") -> str:
    """
    扩散模型控制项：
    - 主解释变量之外，额外控制 Log_RCR
    """
    if fe_mode == "stage":
        fe_part = "C(stage) + C(arxiv_primary_category)"
    elif fe_mode == "year":
        fe_part = "C(year) + C(arxiv_primary_category)"
    else:
        raise ValueError("fe_mode 只能是 'stage' 或 'year'")

    return f"{topic_var} + Log_RCR + log_team_size + log_prior_knowledge + HSS_Diversity + {fe_part}"


def get_sample(df: pd.DataFrame, sample_name: str) -> pd.DataFrame:
    """
    样本选择规则
    """
    if sample_name == "main":
        return df.copy()

    if sample_name == "diffusion_all":
        # 仅剔除 citations>0 但没有 profile 的 87 条
        return df[df["citing_missing_flag"] != "positive_citations_missing_profile"].copy()

    if sample_name == "diffusion_cited":
        # 仅在已有引用的论文上做扩散稳健性
        return df[
            (df["citations"] > 0) &
            (df["citing_missing_flag"] != "positive_citations_missing_profile")
        ].copy()

    if sample_name == "reference":
        # 覆盖率不高，因此门槛设严一点，仅作为附录试算
        return df[
            (df["Reference_Count"] > 0) &
            (df["RefMeta_Matched"] >= 3) &
            (df["RefMeta_Coverage"] >= 0.30)
        ].copy()

    raise ValueError(f"未知 sample_name: {sample_name}")


def fit_model(formula: str, data: pd.DataFrame, model_type: str = "ols"):
    """
    model_type:
    - ols: OLS with HC1 robust SE
    - glm_binom: GLM Binomial (可用于 top10% 的 logit-style 稳健性)
    """
    if model_type == "ols":
        result = smf.ols(formula=formula, data=data, missing="drop").fit(cov_type="HC1")
        return result

    if model_type == "glm_binom":
        result = smf.glm(
            formula=formula,
            data=data,
            family=sm.families.Binomial(),
            missing="drop"
        ).fit(cov_type="HC1")
        return result

    raise ValueError(f"未知 model_type: {model_type}")


def tidy_result(result, model_name: str, model_type: str, sample_name: str, formula: str) -> pd.DataFrame:
    ci = result.conf_int()
    ci.columns = ["ci_low", "ci_high"]

    # OLS 通常是 t 值；GLM 更接近 z 值，这里统一叫 stat
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
    })

    if hasattr(result, "rsquared"):
        out["rsquared"] = result.rsquared
        out["rsquared_adj"] = getattr(result, "rsquared_adj", np.nan)
    else:
        out["rsquared"] = np.nan
        out["rsquared_adj"] = np.nan

    out["aic"] = getattr(result, "aic", np.nan)
    out["bic"] = getattr(result, "bic", np.nan)

    return out


def save_summary_text(result, model_name: str, out_dir: Path):
    out_path = out_dir / f"{model_name}_summary.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result.summary().as_text())


def run_and_save_model(model_name: str, formula: str, df: pd.DataFrame, sample_name: str, model_type: str, out_dir: Path):
    sample_df = get_sample(df, sample_name)
    result = fit_model(formula=formula, data=sample_df, model_type=model_type)

    coef_df = tidy_result(
        result=result,
        model_name=model_name,
        model_type=model_type,
        sample_name=sample_name,
        formula=formula
    )

    coef_path = out_dir / f"{model_name}_coef.csv"
    coef_df.to_csv(coef_path, index=False, encoding="utf-8-sig")
    save_summary_text(result, model_name, out_dir)

    # 控制台简报
    key_terms = [t for t in result.params.index if ("HSS_Continuous" in t or "HSS_Bin" in t)]
    print("\n" + "=" * 80)
    print(f"Model: {model_name}")
    print(f"Type: {model_type}")
    print(f"Sample: {sample_name}")
    print(f"N = {int(result.nobs):,}")
    if hasattr(result, "rsquared"):
        print(f"R^2 = {result.rsquared:.4f}")
    print("Key terms:")
    for t in key_terms[:10]:
        print(f"  {t}: coef={result.params[t]:.4f}, se={result.bse[t]:.4f}, p={result.pvalues[t]:.4g}")

    return result, coef_df


def make_descriptive_tables(df: pd.DataFrame, out_dir: Path):
    desc_cols = [
        "team_hss_dna_pct", "HSS_Continuous", "total_authors", "log_team_size",
        "Prior_Knowledge_Stock", "log_prior_knowledge", "HSS_Diversity",
        "citations", "RCR", "Log_RCR",
        "Citation_Field_Diversity", "Citation_Entropy",
        "NonCS_Citation_Share", "HSS_Citing_Share"
    ]
    desc_cols = [c for c in desc_cols if c in df.columns]
    desc = df[desc_cols].describe().T
    desc.to_csv(out_dir / "descriptive_stats.csv", encoding="utf-8-sig")

    sample_counts = pd.DataFrame({
        "sample_name": ["all", "cited_only", "diffusion_all", "diffusion_cited", "reference_appendix_candidate"],
        "n": [
            len(df),
            int((df["citations"] > 0).sum()),
            len(get_sample(df, "diffusion_all")),
            len(get_sample(df, "diffusion_cited")),
            len(get_sample(df, "reference")),
        ]
    })
    sample_counts.to_csv(out_dir / "sample_counts.csv", index=False, encoding="utf-8-sig")

    crosstab_stage_bin = pd.crosstab(df["stage"], df["HSS_Bin"], dropna=False)
    crosstab_stage_bin.to_csv(out_dir / "stage_by_hss_bin.csv", encoding="utf-8-sig")

    citing_flag_counts = df["citing_missing_flag"].value_counts(dropna=False).rename_axis("flag").reset_index(name="n")
    citing_flag_counts.to_csv(out_dir / "citing_missing_flag_counts.csv", index=False, encoding="utf-8-sig")


def build_compact_key_table(all_coef_df: pd.DataFrame, out_dir: Path):
    """
    把主要关注系数抽出来，方便快速看表
    """
    keep_patterns = [
        "HSS_Continuous",
        "C(HSS_Bin, Treatment(reference='Pure STEM'))",
        "HSS_Continuous:C(stage)",
        "HSS_Continuous:has_hss_concepts",
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


# =========================================================
# 3. 读入数据
# =========================================================
df = load_and_prepare_data(DATA_PATH)

print("\nBasic data snapshot:")
print(f"Rows: {len(df):,}")
print(f"Cited papers: {(df['citations'] > 0).sum():,}")
print(f"Positive citations but missing profile: {(df['citing_missing_flag'] == 'positive_citations_missing_profile').sum():,}")
print(f"Strict concepts-based HSS topics: {df[MAIN_TOPIC_VAR].sum():,}")
print(f"Broad title+concepts social topics: {df[BROAD_TOPIC_VAR].sum():,}")

make_descriptive_tables(df, OUT_DIR)


# =========================================================
# 4. 设定模型
# =========================================================
core_controls_stage = get_core_controls(topic_var=MAIN_TOPIC_VAR, fe_mode="stage")
core_controls_year = get_core_controls(topic_var=MAIN_TOPIC_VAR, fe_mode="year")
diff_controls_stage = get_diffusion_controls(topic_var=MAIN_TOPIC_VAR, fe_mode="stage")

model_specs = []

# -------------------------
# A. 基准引用影响模型
# -------------------------
model_specs += [
    {
        "model_name": "M1_LogRCR_Continuous_StageFE",
        "formula": f"Log_RCR ~ HSS_Continuous + {core_controls_stage}",
        "sample_name": "main",
        "model_type": "ols",
    },
    {
        "model_name": "M1b_LogRCR_Continuous_YearFE",
        "formula": f"Log_RCR ~ HSS_Continuous + {core_controls_year}",
        "sample_name": "main",
        "model_type": "ols",
    },
    {
        "model_name": "M2_LogRCR_Threshold_StageFE",
        "formula": f"Log_RCR ~ C(HSS_Bin, Treatment(reference='Pure STEM')) + {core_controls_stage}",
        "sample_name": "main",
        "model_type": "ols",
    },
]

# -------------------------
# B. 异质性：阶段与主题
# -------------------------
model_specs += [
    {
        "model_name": "M3_LogRCR_StageInteraction",
        "formula": (
            "Log_RCR ~ HSS_Continuous * C(stage) + "
            f"{MAIN_TOPIC_VAR} + log_team_size + log_prior_knowledge + HSS_Diversity + C(arxiv_primary_category)"
        ),
        "sample_name": "main",
        "model_type": "ols",
    },
    {
        "model_name": "M4_LogRCR_TopicInteraction",
        "formula": (
            f"Log_RCR ~ HSS_Continuous * {MAIN_TOPIC_VAR} + "
            "log_team_size + log_prior_knowledge + HSS_Diversity + C(stage) + C(arxiv_primary_category)"
        ),
        "sample_name": "main",
        "model_type": "ols",
    },
]

# -------------------------
# C. 高影响概率：Top 10%
# 主版本：year × arxiv_primary_category 内前10%
# 稳健性：仅 year 内前10%
# -------------------------
model_specs += [
    {
        "model_name": "M5_Hit10YearCat_Continuous_LPM",
        "formula": f"Hit_Rate_10_year_cat ~ HSS_Continuous + {core_controls_stage}",
        "sample_name": "main",
        "model_type": "ols",
    },
    {
        "model_name": "M6_Hit10YearCat_Threshold_LPM",
        "formula": f"Hit_Rate_10_year_cat ~ C(HSS_Bin, Treatment(reference='Pure STEM')) + {core_controls_stage}",
        "sample_name": "main",
        "model_type": "ols",
    },
    {
        "model_name": "M6b_Hit10Year_Continuous_LPM",
        "formula": f"Hit_Rate_10_year ~ HSS_Continuous + {core_controls_stage}",
        "sample_name": "main",
        "model_type": "ols",
    },
    # logit-style 稳健性
    {
        "model_name": "M6c_Hit10YearCat_Continuous_GLMLogit",
        "formula": f"Hit_Rate_10_year_cat ~ HSS_Continuous + {core_controls_stage}",
        "sample_name": "main",
        "model_type": "glm_binom",
    },
]

# -------------------------
# D. 跨学科扩散模型
# 全样本：zero-citation 论文已按约定置 0
# cited-only：稳健性
# -------------------------
diffusion_outcomes = [
    "Citation_Field_Diversity",
    "Citation_Entropy",
    "NonCS_Citation_Share",
    "HSS_Citing_Share",
]

for outcome in diffusion_outcomes:
    model_specs.append({
        "model_name": f"D1_{outcome}_AllPapers",
        "formula": f"{outcome} ~ HSS_Continuous + {diff_controls_stage}",
        "sample_name": "diffusion_all",
        "model_type": "ols",
    })
    model_specs.append({
        "model_name": f"D2_{outcome}_CitedOnly",
        "formula": f"{outcome} ~ HSS_Continuous + {diff_controls_stage}",
        "sample_name": "diffusion_cited",
        "model_type": "ols",
    })

# -------------------------
# E. 扩展稳健性：改用 title + concepts 主题变量
# -------------------------
if RUN_BROAD_TOPIC_ROBUSTNESS:
    broad_controls_stage = get_core_controls(topic_var=BROAD_TOPIC_VAR, fe_mode="stage")
    model_specs += [
        {
            "model_name": "R1_LogRCR_Continuous_BroadTopicControl",
            "formula": f"Log_RCR ~ HSS_Continuous + {broad_controls_stage}",
            "sample_name": "main",
            "model_type": "ols",
        },
        {
            "model_name": "R2_Hit10YearCat_Continuous_BroadTopicControl",
            "formula": f"Hit_Rate_10_year_cat ~ HSS_Continuous + {broad_controls_stage}",
            "sample_name": "main",
            "model_type": "ols",
        },
    ]


# -------------------------
# F. 参考文献附录（默认关闭）
# 目前 coverage ~ 0.20，不建议直接上主文
# -------------------------
if RUN_REFERENCE_APPENDIX:
    model_specs += [
        {
            "model_name": "A1_RefAge_Continuous",
            "formula": f"Mean_Reference_Age ~ HSS_Continuous + {core_controls_stage}",
            "sample_name": "reference",
            "model_type": "ols",
        },
        {
            "model_name": "A2_RefImpact_Continuous",
            "formula": f"Mean_Log_Ref_Citations ~ HSS_Continuous + {core_controls_stage}",
            "sample_name": "reference",
            "model_type": "ols",
        },
        {
            "model_name": "A3_RecentRefShare_Continuous",
            "formula": f"Recent_Reference_Share_3y ~ HSS_Continuous + {core_controls_stage}",
            "sample_name": "reference",
            "model_type": "ols",
        },
    ]


# =========================================================
# 5. 执行模型
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
            out_dir=OUT_DIR
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

# 汇总输出
model_registry_df = pd.DataFrame(model_registry)
model_registry_df.to_csv(OUT_DIR / "model_registry.csv", index=False, encoding="utf-8-sig")

if all_coef_tables:
    all_coef_df = pd.concat(all_coef_tables, ignore_index=True)
    all_coef_df.to_csv(OUT_DIR / "all_model_coefficients_long.csv", index=False, encoding="utf-8-sig")
    build_compact_key_table(all_coef_df, OUT_DIR)

print("\n" + "=" * 80)
print("All done.")
print(f"Results saved to: {OUT_DIR}")
print("=" * 80)
