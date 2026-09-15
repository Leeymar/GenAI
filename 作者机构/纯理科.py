import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# =========================
# 1. 读数据
# =========================
input_csv = "E:\PythonProject\GenAI\data_process\derived\genai_regression_enriched_v3_fixed.csv"
output_csv = "pure_stem_rows.csv"
output_excel = "pure_stem_controls_only.xlsx"

df = pd.read_csv(input_csv)

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

controls = ["log_team_size", "log_prior_knowledge"]
year_fe = "year"
topic_fe = "arxiv_primary_category"

# =========================
# 3. 找纯理工科团队
#    team_hss_dna_pct 是百分比变量：
#    0 = 纯理工科，100 ≈ 纯HSS
# =========================
pure_stem = df[np.isclose(df["team_hss_dna_pct"], 0, atol=1e-8)].copy()

print("========== 原始样本量 ==========")
print(len(df))

print("\n========== 筛选后样本量（纯理工科） ==========")
print(len(pure_stem))

print("\n========== 纯理工科样本前5行 ==========")
show_cols = [c for c in ["arxiv_id", "team_hss_dna_pct", "HSS_Continuous"] if c in pure_stem.columns]
print(pure_stem[show_cols].head())

print("\n========== HSS_Continuous 在纯理工科样本中的分布 ==========")
print(pure_stem["HSS_Continuous"].describe())
print("唯一值个数:", pure_stem["HSS_Continuous"].nunique())

# 导出纯理工科样本
pure_stem.to_csv(output_csv, index=False, encoding="utf-8-sig")
print(f"\n已导出纯理工科样本: {output_csv}")

# =========================
# 4. 星号和格式函数
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

# =========================
# 5. 在纯理工科子样本上跑“控制变量 + 固定效应”
#    注意：这里不能再估 HSS_Continuous
# =========================
def run_model(dep, add_topic_fe=False):
    use_cols = [dep] + controls + [year_fe, topic_fe]
    data = pure_stem[use_cols].dropna().copy()

    formula = f"{dep} ~ {' + '.join(controls)} + C({year_fe})"
    if add_topic_fe:
        formula += f" + C({topic_fe})"

    model = smf.ols(formula=formula, data=data).fit(cov_type="HC1")
    return model, len(data), formula

def build_table():
    model_specs = []

    for dep in y_vars:
        m1, n1, f1 = run_model(dep, add_topic_fe=False)
        model_specs.append({
            "col_name": f"{dep}_(1)",
            "model": m1,
            "nobs": n1,
            "r2": m1.rsquared,
            "year_fe": "Yes",
            "topic_fe": "No",
            "formula": f1
        })

        m2, n2, f2 = run_model(dep, add_topic_fe=True)
        model_specs.append({
            "col_name": f"{dep}_(2)",
            "model": m2,
            "nobs": n2,
            "r2": m2.rsquared,
            "year_fe": "Yes",
            "topic_fe": "Yes",
            "formula": f2
        })

    row_order = [
        "log_team_size",
        "log_prior_knowledge",
        "Intercept"
    ]

    rows = []

    for var in row_order:
        coef_row = {"变量": var}
        se_row = {"变量": ""}

        for spec in model_specs:
            m = spec["model"]
            coef_row[spec["col_name"]] = fmt_coef(
                m.params.get(var, np.nan),
                m.pvalues.get(var, np.nan)
            )
            se_row[spec["col_name"]] = fmt_se(
                m.bse.get(var, np.nan)
            )

        rows.append(coef_row)
        rows.append(se_row)

    # 底部摘要
    for label in ["年份固定效应", "主分类固定效应", "样本量 N", "R²"]:
        row = {"变量": label}
        for spec in model_specs:
            if label == "年份固定效应":
                row[spec["col_name"]] = spec["year_fe"]
            elif label == "主分类固定效应":
                row[spec["col_name"]] = spec["topic_fe"]
            elif label == "样本量 N":
                row[spec["col_name"]] = spec["nobs"]
            elif label == "R²":
                row[spec["col_name"]] = f"{spec['r2']:.4f}"
        rows.append(row)

    return pd.DataFrame(rows), model_specs

table, specs = build_table()

with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
    table.to_excel(writer, sheet_name="纯理工科_控制项回归", index=False)

print(f"\n已输出结果表: {output_excel}")
