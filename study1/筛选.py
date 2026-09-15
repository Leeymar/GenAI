# -*- coding: utf-8 -*-
"""
3.5 调节效应模型：参考文献侧知识整合 × 团队文科浓度
作者：你自己运行即可
说明：
1. 默认核心自变量用 HSS_Continuous
2. 默认二元因变量 Hit_Rate_10_year_cat 用 OLS（线性概率模型）+ HC1稳健标准误
3. 若你前文二元变量使用 Logit / Binomial，请把 BINARY_MODEL 改成 "glm"
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")


# =========================================================
# 0. 基础配置
# =========================================================
DATA_PATH = r"E:\PythonProject\GenAI\data_process\derived\genai_regression_enriched_v3_fixed.csv"

# 输出目录
OUT_DIR = Path(DATA_PATH).parent / "reg_3_5_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 是否筛选样本。比如只跑 Phase1：
# FILTER_QUERY = 'stage == "Phase1"'
FILTER_QUERY = None

# 核心自变量：建议先用这个
X_VAR = "HSS_Continuous"

# 如果你后续要做稳健性检验，可改成：
# X_VAR = "team_hss_dna_pct"

# 二元因变量的模型类型：
# "ols" = 线性概率模型（默认）
# "glm" = Binomial GLM
BINARY_MODEL = "ols"

# 被解释变量：数据列名 -> 论文展示名
Y_VARS = {
    "Log_RCR": "Log_RCR_p",
    "Hit_Rate_10_year_cat": "Hit{10}_p",
    "Citation_Field_Diversity": "Citation_Field_Diversity_p",
    "HSS_Citing_Share": "HSS_Citing_Share_p",
    "Citation_Entropy": "Citation_Entropy_p",
}

# 调节变量：数据列名 -> 论文展示名
MODERATORS = {
    "reference_field_diversity": "参考文献学科数量",
    "reference_entropy": "参考文献学科熵",
    "hss_reference_share": "HSS参考文献占比",
    "cs_core_reference_share": "CS核心参考文献占比",
}

# 控制变量
CONTROLS = [
    "log_team_size",
    "log_prior_knowledge",
]

# 固定效应
FE_VARS = [
    "year",
    "arxiv_primary_category",
]

# 备注：这里只做“变量检查”，不参与模型
OPTIONAL_CHECK_VARS = [
    "team_hss_dna_pct",
    "HSS_Bin",
    "Hit_Rate_10_year",
    "reference_topic_coverage",
    "stage",
    "arxiv_type",
]


# =========================================================
# 1. 工具函数
# =========================================================
def significance_stars(p):
    """根据 p 值返回显著性星号"""
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.10:
        return "*"
    else:
        return ""


def fmt_coef_se(coef, se, p):
    """格式化成论文表格里常见的 '系数(标准误)' """
    if pd.isna(coef) or pd.isna(se):
        return ""
    return f"{coef:.4f}{significance_stars(p)}\n({se:.4f})"


def get_interaction_term(param_index, x_var, moderator):
    """
    从回归参数名中识别交互项。
    statsmodels 通常会生成：
    x_var:moderator 或 moderator:x_var
    """
    candidates = [
        f"{x_var}:{moderator}",
        f"{moderator}:{x_var}",
    ]
    for c in candidates:
        if c in param_index:
            return c

    # 兜底搜索
    for term in param_index:
        if ":" in term and x_var in term and moderator in term:
            return term

    raise KeyError(f"未找到交互项：{x_var} × {moderator}")


def build_formula(y, moderator, x_var, controls, fe_vars):
    """
    构建回归公式：
    y ~ x * moderator + controls + C(year) + C(arxiv_primary_category)
    """
    rhs_parts = [f"{x_var} * {moderator}"]
    rhs_parts.extend(controls)

    for fe in fe_vars:
        rhs_parts.append(f"C({fe})")

    rhs = " + ".join(rhs_parts)
    formula = f"{y} ~ {rhs}"
    return formula


def fit_model(df_sub, y, moderator, x_var, controls, fe_vars, binary_model="ols"):
    """
    拟合单个模型
    连续因变量：OLS + HC1
    二元因变量：
        - 默认 OLS + HC1（线性概率模型）
        - 可选 GLM Binomial + HC1
    """
    formula = build_formula(y, moderator, x_var, controls, fe_vars)

    if y == "Hit_Rate_10_year_cat" and binary_model.lower() == "glm":
        model = smf.glm(
            formula=formula,
            data=df_sub,
            family=sm.families.Binomial()
        ).fit(cov_type="HC1")
        model_type = "GLM-Binomial"
        fit_stat_name = "Pseudo_R2"
        # statsmodels GLM 不直接给 McFadden R2，这里留空
        fit_stat = np.nan
    else:
        model = smf.ols(
            formula=formula,
            data=df_sub
        ).fit(cov_type="HC1")
        model_type = "OLS"
        fit_stat_name = "R_squared"
        fit_stat = getattr(model, "rsquared", np.nan)

    return model, model_type, fit_stat_name, fit_stat, formula


def make_variable_check(df, numeric_cols, category_cols):
    """
    生成变量检查表，帮助你核对哪些变量真正被用于模型
    """
    rows = []

    for col in numeric_cols:
        ser = pd.to_numeric(df[col], errors="coerce")
        rows.append({
            "variable": col,
            "role": "numeric",
            "n_non_missing": ser.notna().sum(),
            "missing_rate": ser.isna().mean(),
            "n_unique": ser.nunique(dropna=True),
            "mean": ser.mean(),
            "std": ser.std(),
            "min": ser.min(),
            "p25": ser.quantile(0.25),
            "median": ser.median(),
            "p75": ser.quantile(0.75),
            "max": ser.max()
        })

    for col in category_cols:
        ser = df[col].astype("string")
        top_values = ser.value_counts(dropna=False).head(5).to_dict()
        rows.append({
            "variable": col,
            "role": "category",
            "n_non_missing": ser.notna().sum(),
            "missing_rate": ser.isna().mean(),
            "n_unique": ser.nunique(dropna=True),
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "p25": np.nan,
            "median": np.nan,
            "p75": np.nan,
            "max": np.nan,
            "top5_values": str(top_values)
        })

    return pd.DataFrame(rows)


# =========================================================
# 2. 读取与预处理数据
# =========================================================
print("=" * 90)
print("开始读取数据")
print("=" * 90)

df = pd.read_csv(DATA_PATH)
print(f"原始数据行数: {len(df):,}")
print(f"原始数据列数: {df.shape[1]:,}")

if FILTER_QUERY:
    df = df.query(FILTER_QUERY).copy()
    print(f"筛选后数据行数: {len(df):,}")
else:
    df = df.copy()

# 需要的列
required_numeric_cols = (
    [X_VAR]
    + list(Y_VARS.keys())
    + list(MODERATORS.keys())
    + CONTROLS
    + ["year"]
)

required_category_cols = [
    "arxiv_primary_category"
]

required_all = required_numeric_cols + required_category_cols

# 检查变量是否存在
missing_cols = [c for c in required_all if c not in df.columns]
if missing_cols:
    raise ValueError(f"以下列在数据中不存在，请检查：{missing_cols}")

# 转数值
for col in required_numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 固定效应分类列
for col in required_category_cols:
    df[col] = df[col].astype("string")

# 替换 inf
df = df.replace([np.inf, -np.inf], np.nan)

# 变量检查表
check_numeric_cols = [c for c in required_numeric_cols if c in df.columns]
check_category_cols = [c for c in required_category_cols if c in df.columns]
for c in OPTIONAL_CHECK_VARS:
    if c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            check_numeric_cols.append(c)
        else:
            check_category_cols.append(c)

# 去重保持顺序
check_numeric_cols = list(dict.fromkeys(check_numeric_cols))
check_category_cols = list(dict.fromkeys(check_category_cols))

variable_check_df = make_variable_check(df, check_numeric_cols, check_category_cols)
variable_check_path = OUT_DIR / "variable_check.csv"
variable_check_df.to_csv(variable_check_path, index=False, encoding="utf-8-sig")

print(f"变量检查表已输出: {variable_check_path}")


# =========================================================
# 3. 跑 20 个模型
# =========================================================
print("=" * 90)
print("开始运行调节效应模型")
print("=" * 90)

results = []
all_params = []
all_summaries = []

for y_col, y_paper_name in Y_VARS.items():
    for mod_col, mod_paper_name in MODERATORS.items():
        # 本模型实际使用的列
        use_cols = [y_col, X_VAR, mod_col] + CONTROLS + FE_VARS
        df_sub = df[use_cols].dropna().copy()

        # 跳过没有变异的情况
        if len(df_sub) == 0:
            results.append({
                "y_var": y_col,
                "y_label": y_paper_name,
                "moderator": mod_col,
                "moderator_label": mod_paper_name,
                "model_type": None,
                "n_obs": 0,
                "interaction_term": None,
                "interaction_coef": np.nan,
                "interaction_se": np.nan,
                "interaction_p": np.nan,
                "fit_stat_name": None,
                "fit_stat": np.nan,
                "formula": None,
                "note": "样本为空，未估计"
            })
            print(f"[跳过] {y_col} × {mod_col}：样本为空")
            continue

        if df_sub[y_col].nunique(dropna=True) <= 1:
            results.append({
                "y_var": y_col,
                "y_label": y_paper_name,
                "moderator": mod_col,
                "moderator_label": mod_paper_name,
                "model_type": None,
                "n_obs": len(df_sub),
                "interaction_term": None,
                "interaction_coef": np.nan,
                "interaction_se": np.nan,
                "interaction_p": np.nan,
                "fit_stat_name": None,
                "fit_stat": np.nan,
                "formula": None,
                "note": "因变量无变异，未估计"
            })
            print(f"[跳过] {y_col} × {mod_col}：因变量无变异")
            continue

        if df_sub[mod_col].nunique(dropna=True) <= 1:
            results.append({
                "y_var": y_col,
                "y_label": y_paper_name,
                "moderator": mod_col,
                "moderator_label": mod_paper_name,
                "model_type": None,
                "n_obs": len(df_sub),
                "interaction_term": None,
                "interaction_coef": np.nan,
                "interaction_se": np.nan,
                "interaction_p": np.nan,
                "fit_stat_name": None,
                "fit_stat": np.nan,
                "formula": None,
                "note": "调节变量无变异，未估计"
            })
            print(f"[跳过] {y_col} × {mod_col}：调节变量无变异")
            continue

        try:
            model, model_type, fit_stat_name, fit_stat, formula = fit_model(
                df_sub=df_sub,
                y=y_col,
                moderator=mod_col,
                x_var=X_VAR,
                controls=CONTROLS,
                fe_vars=FE_VARS,
                binary_model=BINARY_MODEL
            )

            interaction_term = get_interaction_term(model.params.index, X_VAR, mod_col)
            coef = model.params.get(interaction_term, np.nan)
            se = model.bse.get(interaction_term, np.nan)
            pval = model.pvalues.get(interaction_term, np.nan)

            results.append({
                "y_var": y_col,
                "y_label": y_paper_name,
                "moderator": mod_col,
                "moderator_label": mod_paper_name,
                "model_type": model_type,
                "n_obs": int(model.nobs),
                "interaction_term": interaction_term,
                "interaction_coef": coef,
                "interaction_se": se,
                "interaction_p": pval,
                "fit_stat_name": fit_stat_name,
                "fit_stat": fit_stat,
                "formula": formula,
                "note": ""
            })

            # 保存全参数
            params_df = pd.DataFrame({
                "term": model.params.index,
                "coef": model.params.values,
                "se": model.bse.values,
                "p_value": model.pvalues.values,
            })
            params_df["y_var"] = y_col
            params_df["y_label"] = y_paper_name
            params_df["moderator"] = mod_col
            params_df["moderator_label"] = mod_paper_name
            params_df["model_type"] = model_type
            params_df["n_obs"] = int(model.nobs)
            params_df["formula"] = formula
            all_params.append(params_df)

            # 保存摘要
            try:
                summary_text = model.summary().as_text()
            except Exception:
                summary_text = str(model.summary())

            block = []
            block.append("=" * 110)
            block.append(f"因变量: {y_col} ({y_paper_name})")
            block.append(f"调节变量: {mod_col} ({mod_paper_name})")
            block.append(f"模型类型: {model_type}")
            block.append(f"样本量: {int(model.nobs)}")
            block.append(f"{fit_stat_name}: {fit_stat}")
            block.append(f"公式: {formula}")
            block.append("-" * 110)
            block.append(summary_text)
            block.append("")
            all_summaries.append("\n".join(block))

            print(f"[完成] {y_col} × {mod_col} | N={int(model.nobs)} | 交互项={coef:.4f}, p={pval:.4f}")

        except Exception as e:
            results.append({
                "y_var": y_col,
                "y_label": y_paper_name,
                "moderator": mod_col,
                "moderator_label": mod_paper_name,
                "model_type": None,
                "n_obs": len(df_sub),
                "interaction_term": None,
                "interaction_coef": np.nan,
                "interaction_se": np.nan,
                "interaction_p": np.nan,
                "fit_stat_name": None,
                "fit_stat": np.nan,
                "formula": None,
                "note": f"报错: {repr(e)}"
            })
            print(f"[失败] {y_col} × {mod_col} -> {repr(e)}")


# =========================================================
# 4. 结果整理
# =========================================================
results_df = pd.DataFrame(results)

# 格式化交互项
results_df["interaction_fmt"] = results_df.apply(
    lambda r: fmt_coef_se(r["interaction_coef"], r["interaction_se"], r["interaction_p"]),
    axis=1
)

# 排序顺序
row_order = [MODERATORS[k] for k in MODERATORS.keys()]
col_order = [Y_VARS[k] for k in Y_VARS.keys()]

# 论文表6：格式化版
table6_fmt = results_df.pivot(
    index="moderator_label",
    columns="y_label",
    values="interaction_fmt"
).reindex(index=row_order, columns=col_order)

# 原始系数
table6_coef = results_df.pivot(
    index="moderator_label",
    columns="y_label",
    values="interaction_coef"
).reindex(index=row_order, columns=col_order)

# 标准误
table6_se = results_df.pivot(
    index="moderator_label",
    columns="y_label",
    values="interaction_se"
).reindex(index=row_order, columns=col_order)

# p值
table6_p = results_df.pivot(
    index="moderator_label",
    columns="y_label",
    values="interaction_p"
).reindex(index=row_order, columns=col_order)

# 样本量
table6_n = results_df.pivot(
    index="moderator_label",
    columns="y_label",
    values="n_obs"
).reindex(index=row_order, columns=col_order)

# 模型类型
table6_model_type = results_df.pivot(
    index="moderator_label",
    columns="y_label",
    values="model_type"
).reindex(index=row_order, columns=col_order)

# 合并参数表
if all_params:
    all_params_df = pd.concat(all_params, ignore_index=True)
else:
    all_params_df = pd.DataFrame()


# =========================================================
# 5. 导出结果
# =========================================================
long_csv_path = OUT_DIR / "moderation_results_long.csv"
all_params_path = OUT_DIR / "all_model_params.csv"
summary_txt_path = OUT_DIR / "model_summaries.txt"
xlsx_path = OUT_DIR / "table6_moderation_results.xlsx"

results_df.to_csv(long_csv_path, index=False, encoding="utf-8-sig")

if not all_params_df.empty:
    all_params_df.to_csv(all_params_path, index=False, encoding="utf-8-sig")

with open(summary_txt_path, "w", encoding="utf-8") as f:
    f.write("\n\n".join(all_summaries))

with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
    variable_check_df.to_excel(writer, sheet_name="variable_check", index=False)
    results_df.to_excel(writer, sheet_name="long_results", index=False)
    table6_fmt.to_excel(writer, sheet_name="table6_formatted")
    table6_coef.to_excel(writer, sheet_name="table6_coef")
    table6_se.to_excel(writer, sheet_name="table6_se")
    table6_p.to_excel(writer, sheet_name="table6_p")
    table6_n.to_excel(writer, sheet_name="table6_n")
    table6_model_type.to_excel(writer, sheet_name="table6_model_type")
    if not all_params_df.empty:
        all_params_df.to_excel(writer, sheet_name="all_params", index=False)

print("\n" + "=" * 90)
print("全部完成")
print("=" * 90)
print(f"输出目录: {OUT_DIR}")
print(f"变量检查表: {variable_check_path}")
print(f"长表结果:   {long_csv_path}")
print(f"Excel汇总:  {xlsx_path}")
print(f"模型参数:   {all_params_path if not all_params_df.empty else '无'}")
print(f"模型摘要:   {summary_txt_path}")

print("\n" + "=" * 90)
print("表6（格式化交互项结果）预览")
print("=" * 90)
print(table6_fmt)

print("\n" + "=" * 90)
print("你当前的模型设定")
print("=" * 90)
print(f"核心自变量 X_VAR = {X_VAR}")
print(f"二元变量模型 BINARY_MODEL = {BINARY_MODEL}")
print("被解释变量：")
for k, v in Y_VARS.items():
    print(f"  - {k}   ->   {v}")
print("调节变量：")
for k, v in MODERATORS.items():
    print(f"  - {k}   ->   {v}")
print("控制变量：", CONTROLS)
print("固定效应：", FE_VARS)
