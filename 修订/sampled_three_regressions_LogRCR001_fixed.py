import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# =====================================================
# 纯理科团队稳健性检验
# 与原 sampled_three_regressions.xlsx 表格结构一致
#
# 三个sheet:
# 1. 基准回归表
# 2. 交互回归表
# 3. HSS参考文献占比回归
#
# 修改:
# Log_RCR = ln(RCR + 0.01)
# 删除 Hit_Rate_10_year
# =====================================================

INPUT_CSV = r"E:\PythonProject\GenAI\作者机构\sampled_31423_rows.csv"
OUTPUT_EXCEL = "sampled_three_regressions_LogRCR001.xlsx"

df = pd.read_csv(INPUT_CSV)

# 更新RCR转换
df["Log_RCR"] = np.log(df["RCR"] + 0.01)

# 删除不用变量
for col in ["Hit_Rate_10_year", "Hit_Rate_10_year_cat"]:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)


def add_star(p):
    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.10:
        return "*"
    return ""


def run_reg(y, extra="", topic_fe=False):
    formula = (
        f"{y} ~ HSS_Continuous "
        f"{extra} "
        "+ log_team_size "
        "+ log_prior_knowledge "
        "+ C(year)"
    )

    if topic_fe:
        formula += " + C(arxiv_primary_category)"

    return smf.ols(formula, data=df).fit(cov_type="HC1")


def table_from_models(models, variables):
    rows = []

    for v in variables:
        rows.append(
            {
                "变量": v,
                **{
                    k: (
                        f"{m.params[v]:.4f}{add_star(m.pvalues[v])}"
                        if v in m.params else ""
                    )
                    for k, m in models.items()
                }
            }
        )

        rows.append(
            {
                "变量": "",
                **{
                    k: (
                        f"({m.bse[v]:.4f})"
                        if v in m.params else ""
                    )
                    for k, m in models.items()
                }
            }
        )

    return pd.DataFrame(rows)


def add_info(table, models):
    table.loc[len(table)] = ["控制变量", "Yes", "Yes"]
    table.loc[len(table)] = ["年份固定效应", "Yes", "Yes"]
    table.loc[len(table)] = ["主分类固定效应", "No", "Yes"]
    table.loc[len(table)] = [
        "样本量 N",
        int(models["模型1"].nobs),
        int(models["模型2"].nobs)
    ]
    table.loc[len(table)] = [
        "R²",
        round(models["模型1"].rsquared, 4),
        round(models["模型2"].rsquared, 4)
    ]
    return table


# =========================
# 1. 基准回归表
# =========================

baseline_models = {
    "模型1": run_reg("Log_RCR", topic_fe=False),
    "模型2": run_reg("Log_RCR", topic_fe=True)
}

baseline_table = table_from_models(
    baseline_models,
    [
        "HSS_Continuous",
        "log_team_size",
        "log_prior_knowledge",
        "Intercept"
    ]
)

baseline_table = add_info(
    baseline_table,
    baseline_models
)


# =========================
# 2. 交互回归表
# =========================

interaction_models = {
    "模型1": run_reg(
        "Log_RCR",
        " + hss_reference_share + HSS_Continuous:hss_reference_share",
        False
    ),
    "模型2": run_reg(
        "Log_RCR",
        " + hss_reference_share + HSS_Continuous:hss_reference_share",
        True
    )
}

interaction_table = table_from_models(
    interaction_models,
    [
        "HSS_Continuous",
        "hss_reference_share",
        "HSS_Continuous:hss_reference_share",
        "log_team_size",
        "log_prior_knowledge",
        "Intercept"
    ]
)

interaction_table = add_info(
    interaction_table,
    interaction_models
)


# =========================
# 3. HSS参考文献占比回归
# =========================

hssref_models = {
    "模型1": run_reg(
        "hss_reference_share",
        "",
        False
    ),
    "模型2": run_reg(
        "hss_reference_share",
        "",
        True
    )
}

hssref_table = table_from_models(
    hssref_models,
    [
        "HSS_Continuous",
        "log_team_size",
        "log_prior_knowledge",
        "Intercept"
    ]
)

hssref_table = add_info(
    hssref_table,
    hssref_models
)


# =========================
# 输出
# =========================

with pd.ExcelWriter(
    OUTPUT_EXCEL,
    engine="openpyxl"
) as writer:

    baseline_table.to_excel(
        writer,
        sheet_name="基准回归表",
        index=False
    )

    interaction_table.to_excel(
        writer,
        sheet_name="交互回归表",
        index=False
    )

    hssref_table.to_excel(
        writer,
        sheet_name="HSS参考文献占比回归",
        index=False
    )


print("完成:", OUTPUT_EXCEL)
