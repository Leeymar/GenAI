import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# =========================
# 1) 路径与筛选条件
# =========================
INPUT_CSV = "E:\PythonProject\GenAI\data_process\derived\genai_regression_enriched_v3_fixed.csv"
FILTER_COL = "team_hss_dna_pct"
FILTER_VALUE = 0   # 如果最后确认“纯理工科团队”其实对应 0，把这里改成 0

OUTPUT_MATCHED_CSV = f"{FILTER_COL}_eq_{FILTER_VALUE}_rows.csv"
OUTPUT_BASELINE_XLSX = f"baseline_regression_{FILTER_COL}_eq_{FILTER_VALUE}.xlsx"

# =========================
# 2) 变量设置
# =========================
y_vars = [
    "Log_RCR",
    "Hit_Rate_10_year",
    "Citation_Field_Diversity",
    "HSS_Citing_Share",
    "Citation_Entropy",
]

main_x = "HSS_Continuous"
controls = ["log_team_size", "log_prior_knowledge"]
year_fe = "year"
topic_fe = "arxiv_primary_category"

var_name_map = {
    "HSS_Continuous": "团队HSS知识浓度",
    "log_team_size": "团队规模（对数）",
    "log_prior_knowledge": "既有知识存量（对数）",
    "Intercept": "常数项",
}

# =========================
# 3) 读数与基础检查
# =========================
df = pd.read_csv(INPUT_CSV)

required_cols = y_vars + [main_x, FILTER_COL] + controls + [year_fe, topic_fe]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"缺少字段: {missing}")

print("\n========== 原始样本量 ==========")
print(len(df))
print("\n========== team_hss_dna_pct 分布 ==========")
print(df[FILTER_COL].value_counts(dropna=False).sort_index())

# 提醒：先确认 1/0 的含义
if FILTER_VALUE == 1:
    print("\n[提醒] 你现在按 team_hss_dna_pct == 1 取样。")
    print("如果你们的数据里 1 不是‘纯理工科’，请把 FILTER_VALUE 改成正确值后再跑。")

matched = df.loc[df[FILTER_COL] == FILTER_VALUE].copy()
print("\n========== 筛选后样本量 ==========")
print(len(matched))

if matched.empty:
    raise ValueError(f"没有找到 {FILTER_COL} == {FILTER_VALUE} 的样本。")

# 导出所有匹配样本，便于你直接检查“找出来”的行
matched.to_csv(OUTPUT_MATCHED_CSV, index=False, encoding="utf-8-sig")
print(f"\n已导出匹配样本: {OUTPUT_MATCHED_CSV}")

# 如果有常见ID列，就顺手打印前几行看一下
candidate_id_cols = [
    "paper_id", "team_id", "article_id", "arxiv_id", "id", "paperid", "teamid"
]
id_cols = [c for c in candidate_id_cols if c in matched.columns]
if id_cols:
    print("\n========== 匹配样本前5行（含ID）==========")
    print(matched[id_cols + [FILTER_COL, main_x]].head())
else:
    print("\n========== 匹配样本前5行 ==========")
    preview_cols = [FILTER_COL, main_x] + [c for c in [year_fe, topic_fe] if c in matched.columns]
    print(matched[preview_cols].head())

print("\n========== HSS_Continuous 在筛选样本中的变异 ==========")
print(matched[main_x].describe())
print("唯一值个数:", matched[main_x].nunique(dropna=True))

if matched[main_x].nunique(dropna=True) <= 1:
    raise ValueError(
        f"筛选后的样本里 {main_x} 几乎没有变异，无法有效估计基准回归。"
        "这通常说明你筛选口径和主解释变量高度重合，需要改筛选条件或换识别策略。"
    )

# 类别变量处理
matched[year_fe] = matched[year_fe].astype("category")
matched[topic_fe] = matched[topic_fe].astype("category")

# =========================
# 4) 回归与格式化函数
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


def get_formula(dep, add_topic_fe=False):
    ctrl_part = " + ".join(controls)
    year_part = f"C({year_fe})"
    topic_part = f" + C({topic_fe})" if add_topic_fe else ""
    return f"{dep} ~ {main_x} + {ctrl_part} + {year_part}{topic_part}"


def run_model(dep, add_topic_fe=False):
    use_cols = [dep, main_x] + controls + [year_fe, topic_fe]
    data = matched[use_cols].dropna().copy()
    formula = get_formula(dep, add_topic_fe=add_topic_fe)
    model = smf.ols(formula=formula, data=data).fit(cov_type="HC1")
    return model, len(data), formula


def make_baseline_table(model_specs):
    row_order = [main_x, "log_team_size", "log_prior_knowledge", "Intercept"]
    rows = []

    for var in row_order:
        coef_row = {"变量": var_name_map.get(var, var)}
        se_row = {"变量": ""}
        for spec in model_specs:
            m = spec["model"]
            coef_row[spec["col_name"]] = fmt_coef(m.params.get(var, np.nan), m.pvalues.get(var, np.nan))
            se_row[spec["col_name"]] = fmt_se(m.bse.get(var, np.nan))
        rows.append(coef_row)
        rows.append(se_row)

    summary_labels = ["控制变量", "年份固定效应", "主分类固定效应", "样本量 N", "R²"]
    for label in summary_labels:
        row = {"变量": label}
        for spec in model_specs:
            if label == "控制变量":
                row[spec["col_name"]] = "Yes"
            elif label == "年份固定效应":
                row[spec["col_name"]] = spec["year_fe"]
            elif label == "主分类固定效应":
                row[spec["col_name"]] = spec["topic_fe"]
            elif label == "样本量 N":
                row[spec["col_name"]] = int(spec["nobs"])
            elif label == "R²":
                row[spec["col_name"]] = f"{spec['r2']:.4f}"
        rows.append(row)

    return pd.DataFrame(rows)

# =========================
# 5) 跑“筛选样本”的基准回归
# =========================
model_specs = []
for dep in y_vars:
    m1, n1, f1 = run_model(dep, add_topic_fe=False)
    model_specs.append({
        "col_name": f"{dep}_1",
        "model": m1,
        "nobs": n1,
        "r2": m1.rsquared,
        "year_fe": "Yes",
        "topic_fe": "No",
        "formula": f1,
    })

    m2, n2, f2 = run_model(dep, add_topic_fe=True)
    model_specs.append({
        "col_name": f"{dep}_2",
        "model": m2,
        "nobs": n2,
        "r2": m2.rsquared,
        "year_fe": "Yes",
        "topic_fe": "Yes",
        "formula": f2,
    })

baseline_table = make_baseline_table(model_specs)

# 模型摘要表
summary_rows = []
for spec in model_specs:
    summary_rows.append({
        "列名": spec["col_name"],
        "样本量": spec["nobs"],
        "R2": spec["r2"],
        "年份固定效应": spec["year_fe"],
        "主分类固定效应": spec["topic_fe"],
        "公式": spec["formula"],
    })
summary_df = pd.DataFrame(summary_rows)

note_df = pd.DataFrame({
    "项目": [
        "输入文件",
        "筛选条件",
        "筛选后样本量",
        "主解释变量",
        "控制变量",
        "模型1",
        "模型2",
        "稳健标准误",
        "提醒",
    ],
    "内容": [
        INPUT_CSV,
        f"{FILTER_COL} == {FILTER_VALUE}",
        len(matched),
        main_x,
        ", ".join(controls),
        f"Y ~ {main_x} + {' + '.join(controls)} + C({year_fe})",
        f"Y ~ {main_x} + {' + '.join(controls)} + C({year_fe}) + C({topic_fe})",
        "HC1",
        "若你要的其实是‘纯理工科团队’，请先核对 team_hss_dna_pct 的编码方向；很多命名下 1 可能表示 HSS 比例为100%。",
    ]
})

with pd.ExcelWriter(OUTPUT_BASELINE_XLSX, engine="openpyxl") as writer:
    baseline_table.to_excel(writer, sheet_name="基准回归表", index=False)
    summary_df.to_excel(writer, sheet_name="模型摘要", index=False)
    note_df.to_excel(writer, sheet_name="说明", index=False)

print(f"\n已输出基准回归表: {OUTPUT_BASELINE_XLSX}")
print("完成。")
