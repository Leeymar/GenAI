import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# =========================
# 1. 路径设置
# =========================
input_csv = "E:\PythonProject\GenAI\data_process\derived\genai_regression_enriched_v3_fixed.csv"
output_excel = "three_tables.xlsx"

# =========================
# 2. 变量设置
# =========================
y_vars = [
    "Log_RCR",
    "Hit_Rate_10_year",
    "Citation_Field_Diversity",
    "HSS_Citing_Share",
    "Citation_Entropy"
]

main_x = "HSS_Continuous"
moderator = "hss_reference_share"
controls = ["log_team_size", "log_prior_knowledge"]
topic_fe = "arxiv_primary_category"
year_fe = "year"

# 中文显示名（可按需改）
var_name_map = {
    "HSS_Continuous": "团队HSS知识浓度",
    "hss_reference_share": "HSS参考文献占比",
    "HSS_Continuous:hss_reference_share": "团队HSS知识浓度 × HSS参考文献占比",
    "log_team_size": "团队规模（对数）",
    "log_prior_knowledge": "既有知识存量（对数）",
    "const": "常数项",
    "Intercept": "常数项",
}

dep_name_map = {
    "Log_RCR": "Log_RCR",
    "Hit_Rate_10_year": "Hit_Rate_10_year",
    "Citation_Field_Diversity": "Citation_Field_Diversity",
    "HSS_Citing_Share": "HSS_Citing_Share",
    "Citation_Entropy": "Citation_Entropy",
    "hss_reference_share": "HSS参考文献占比"
}

# =========================
# 3. 读取数据
# =========================
df = pd.read_csv(input_csv)

# 只保留需要字段，避免缺失值导致每个模型样本不一致
needed_cols = (
    y_vars
    + [main_x, moderator]
    + controls
    + [year_fe, topic_fe]
)

df = df.copy()
for col in needed_cols:
    if col not in df.columns:
        raise ValueError(f"缺少变量: {col}")

# year/category 处理为类别变量
df[year_fe] = df[year_fe].astype("category")
df[topic_fe] = df[topic_fe].astype("category")

# =========================
# 4. 工具函数
# =========================
def star(p):
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.10:
        return "*"
    return ""

def fmt_coef(x, p):
    if pd.isna(x):
        return ""
    return f"{x:.4f}{star(p)}"

def fmt_se(x):
    if pd.isna(x):
        return ""
    return f"({x:.4f})"

def get_formula(dep, model_type="baseline", add_topic_fe=False):
    """
    model_type:
      - baseline
      - interaction
      - hssref
    """
    ctrl_part = " + ".join(controls)
    year_part = f"C({year_fe})"
    topic_part = f" + C({topic_fe})" if add_topic_fe else ""

    if model_type == "baseline":
        return f"{dep} ~ {main_x} + {ctrl_part} + {year_part}{topic_part}"

    elif model_type == "interaction":
        return (
            f"{dep} ~ {main_x} + {moderator} + {main_x}:{moderator} + "
            f"{ctrl_part} + {year_part}{topic_part}"
        )

    elif model_type == "hssref":
        return f"{moderator} ~ {main_x} + {ctrl_part} + {year_part}{topic_part}"

    else:
        raise ValueError("model_type 必须是 baseline / interaction / hssref")

def run_model(dep, model_type="baseline", add_topic_fe=False):
    formula = get_formula(dep, model_type=model_type, add_topic_fe=add_topic_fe)

    # 针对不同模型筛选样本
    if model_type in ["baseline", "interaction"]:
        use_cols = [dep, main_x, moderator] + controls + [year_fe, topic_fe]
    else:
        use_cols = [moderator, main_x] + controls + [year_fe, topic_fe]

    data = df[use_cols].dropna().copy()

    model = smf.ols(formula=formula, data=data).fit(cov_type="HC1")
    return model, data.shape[0], formula

def make_regression_table(model_specs, row_order, sheet_name):
    """
    model_specs: list of dict
        {
          "col_name": "...",
          "model": fitted_model,
          "nobs": ...,
          "r2": ...,
          "year_fe": "Yes",
          "topic_fe": "No"
        }
    """
    rows = []

    for var in row_order:
        coef_row = {"变量": var_name_map.get(var, var)}
        se_row = {"变量": ""}

        for spec in model_specs:
            m = spec["model"]
            params = m.params
            pvals = m.pvalues
            bse = m.bse

            coef_row[spec["col_name"]] = fmt_coef(
                params.get(var, np.nan),
                pvals.get(var, np.nan)
            )
            se_row[spec["col_name"]] = fmt_se(bse.get(var, np.nan))

        rows.append(coef_row)
        rows.append(se_row)

    # 底部摘要
    summary_items = [
        ("控制变量", "Yes"),
        ("年份固定效应", None),
        ("主分类固定效应", None),
        ("样本量 N", None),
        ("R²", None),
    ]

    for label, fixed_val in summary_items:
        row = {"变量": label}
        for spec in model_specs:
            if label == "控制变量":
                row[spec["col_name"]] = fixed_val
            elif label == "年份固定效应":
                row[spec["col_name"]] = spec["year_fe"]
            elif label == "主分类固定效应":
                row[spec["col_name"]] = spec["topic_fe"]
            elif label == "样本量 N":
                row[spec["col_name"]] = int(spec["nobs"])
            elif label == "R²":
                row[spec["col_name"]] = f"{spec['r2']:.4f}"
        rows.append(row)

    out = pd.DataFrame(rows)
    return out

def build_baseline_table():
    model_specs = []

    for dep in y_vars:
        # 无主题固定效应
        m1, n1, f1 = run_model(dep, model_type="baseline", add_topic_fe=False)
        model_specs.append({
            "col_name": f"{dep_name_map.get(dep, dep)}_1",
            "model": m1,
            "nobs": n1,
            "r2": m1.rsquared,
            "year_fe": "Yes",
            "topic_fe": "No",
            "formula": f1
        })

        # 有主题固定效应
        m2, n2, f2 = run_model(dep, model_type="baseline", add_topic_fe=True)
        model_specs.append({
            "col_name": f"{dep_name_map.get(dep, dep)}_2",
            "model": m2,
            "nobs": n2,
            "r2": m2.rsquared,
            "year_fe": "Yes",
            "topic_fe": "Yes",
            "formula": f2
        })

    row_order = [
        main_x,
        "log_team_size",
        "log_prior_knowledge",
        "Intercept"
    ]

    table = make_regression_table(
        model_specs=model_specs,
        row_order=row_order,
        sheet_name="基准回归表"
    )
    return table, model_specs

def build_interaction_table():
    model_specs = []

    for dep in y_vars:
        # 无主题固定效应
        m1, n1, f1 = run_model(dep, model_type="interaction", add_topic_fe=False)
        model_specs.append({
            "col_name": f"{dep_name_map.get(dep, dep)}_1",
            "model": m1,
            "nobs": n1,
            "r2": m1.rsquared,
            "year_fe": "Yes",
            "topic_fe": "No",
            "formula": f1
        })

        # 有主题固定效应
        m2, n2, f2 = run_model(dep, model_type="interaction", add_topic_fe=True)
        model_specs.append({
            "col_name": f"{dep_name_map.get(dep, dep)}_2",
            "model": m2,
            "nobs": n2,
            "r2": m2.rsquared,
            "year_fe": "Yes",
            "topic_fe": "Yes",
            "formula": f2
        })

    row_order = [
        main_x,
        moderator,
        f"{main_x}:{moderator}",
        "log_team_size",
        "log_prior_knowledge",
        "Intercept"
    ]

    table = make_regression_table(
        model_specs=model_specs,
        row_order=row_order,
        sheet_name="交互回归表"
    )
    return table, model_specs

def build_hssref_table():
    model_specs = []

    # 无主题固定效应
    m1, n1, f1 = run_model(moderator, model_type="hssref", add_topic_fe=False)
    model_specs.append({
        "col_name": "HSS参考文献占比_1",
        "model": m1,
        "nobs": n1,
        "r2": m1.rsquared,
        "year_fe": "Yes",
        "topic_fe": "No",
        "formula": f1
    })

    # 有主题固定效应
    m2, n2, f2 = run_model(moderator, model_type="hssref", add_topic_fe=True)
    model_specs.append({
        "col_name": "HSS参考文献占比_2",
        "model": m2,
        "nobs": n2,
        "r2": m2.rsquared,
        "year_fe": "Yes",
        "topic_fe": "Yes",
        "formula": f2
    })

    row_order = [
        main_x,
        "log_team_size",
        "log_prior_knowledge",
        "Intercept"
    ]

    table = make_regression_table(
        model_specs=model_specs,
        row_order=row_order,
        sheet_name="HSS参考文献占比回归"
    )
    return table, model_specs

# =========================
# 5. 生成三张表
# =========================
baseline_table, baseline_specs = build_baseline_table()
interaction_table, interaction_specs = build_interaction_table()
hssref_table, hssref_specs = build_hssref_table()

# =========================
# 6. 导出 Excel
# =========================
with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
    baseline_table.to_excel(writer, sheet_name="基准回归表", index=False)
    interaction_table.to_excel(writer, sheet_name="交互回归表", index=False)
    hssref_table.to_excel(writer, sheet_name="HSS参考文献占比回归", index=False)

print(f"已输出: {output_excel}")
