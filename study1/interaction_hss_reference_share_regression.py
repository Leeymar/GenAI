#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
交互回归：检验 hss_reference_share 对 团队HSS知识浓度(HSS_Continuous)
与知识扩散结果之间关系的调节效应。

模型：
    Y ~ HSS_Continuous * hss_reference_share
        + log_team_size + log_prior_knowledge
        + C(year)
        [+ C(arxiv_primary_category)]

输出：
    1) interaction_pretty       适合直接看表
    2) interaction_core_long    仅核心变量长表
    3) interaction_full_params  全部参数（含固定效应虚拟变量）
    4) interaction_model_summary 各模型摘要
    5) readme                   口径说明

默认输入输出路径可以直接改下面两个常量，
也支持命令行：
    python interaction_hss_reference_share_regression.py input.csv output.xlsx
"""

from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


# ====== 默认路径：按你的本地工程口径写好，可直接改 ======
DEFAULT_INPUT_CSV = r"E:\PythonProject\GenAI\data_process\derived\genai_regression_enriched_v3_fixed.csv"
DEFAULT_OUTPUT_XLSX = r"E:\PythonProject\GenAI\修订\interaction_hss_reference.xlsx"


# ====== 变量配置 ======
OUTCOMES = OrderedDict([
    ("平均引用影响 (Log_RCR_p)", "Log_RCR"),
    ("施引学科广度 (Citation_Field_Diversity_p)", "Citation_Field_Diversity"),
    ("HSS施引占比 (HSS_Citing_Share_p)", "HSS_Citing_Share"),
    ("施引学科熵 (Citation_Entropy_p)", "Citation_Entropy"),
])

VAR_LABELS = OrderedDict([
    ("HSS_Continuous", "团队HSS浓度"),
    ("hss_reference_share", "HSS参考文献占比"),
    ("HSS_Continuous:hss_reference_share", "团队HSS浓度 × HSS参考文献占比"),
    ("log_team_size", "团队规模（对数）"),
    ("log_prior_knowledge", "先验知识存量（对数）"),
    ("Intercept", "常数项"),
])

DISPLAY_ROW_ORDER = [
    "团队HSS浓度",
    "HSS参考文献占比",
    "团队HSS浓度 × HSS参考文献占比",
    "团队规模（对数）",
    "先验知识存量（对数）",
    "常数项",
    "观测值",
    "R²",
    "调整后R²",
    "年份固定效应",
    "arXiv主分类固定效应",
]

REQUIRED_BASE_COLS = [
    "HSS_Continuous",
    "hss_reference_share",
    "log_team_size",
    "log_prior_knowledge",
    "year",
    "arxiv_primary_category",
]


def significance_stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def fmt_coef_se(coef: float, se: float, p: float) -> str:
    stars = significance_stars(p)
    return f"{coef:.4f}{stars}\n({se:.4f})"


def safe_read_csv(csv_path: Path) -> pd.DataFrame:
    # 兼容常见编码与大文件读取场景
    for enc in ["utf-8", "utf-8-sig", "gbk", "gb18030", "latin1"]:
        try:
            return pd.read_csv(csv_path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    # 最后直接抛原错误
    return pd.read_csv(csv_path, low_memory=False)


def validate_columns(df: pd.DataFrame, outcome_cols: List[str]) -> None:
    missing = [c for c in (REQUIRED_BASE_COLS + outcome_cols) if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少这些列：{missing}")


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # ==========================
    # Reviewer suggested adjustment:
    # Log_RCR = ln(RCR+0.01)
    # ==========================
    out["Log_RCR"] = np.log(out["RCR"] + 0.01)

    # 把明显应为数值的列转为数值；非法值转 NaN
    numeric_cols = [
        "HSS_Continuous",
        "hss_reference_share",
        "log_team_size",
        "log_prior_knowledge",
        "year",
        *OUTCOMES.values(),
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # 分类变量转成字符串，避免某些情况下类别读取混乱
    if "arxiv_primary_category" in out.columns:
        out["arxiv_primary_category"] = out["arxiv_primary_category"].astype("string")

    return out


def fit_one_model(df: pd.DataFrame, outcome_col: str, with_topic_fe: bool):
    rhs = [
        "HSS_Continuous * hss_reference_share",
        "log_team_size",
        "log_prior_knowledge",
        "C(year)",
    ]
    if with_topic_fe:
        rhs.append("C(arxiv_primary_category)")

    formula = f"{outcome_col} ~ " + " + ".join(rhs)

    needed_cols = [
        outcome_col,
        "HSS_Continuous",
        "hss_reference_share",
        "log_team_size",
        "log_prior_knowledge",
        "year",
    ]
    if with_topic_fe:
        needed_cols.append("arxiv_primary_category")

    model_df = df[needed_cols].dropna().copy()
    res = smf.ols(formula=formula, data=model_df).fit(cov_type="HC1")
    return res, model_df, formula


def collect_results(df: pd.DataFrame):
    core_rows = []
    all_param_rows = []
    summary_rows = []

    for outcome_display, outcome_col in OUTCOMES.items():
        for with_topic_fe, spec_name in [(False, "无主题FE"), (True, "有主题FE")]:
            res, model_df, formula = fit_one_model(df, outcome_col, with_topic_fe)

            summary_rows.append({
                "模型组": "调节回归",
                "被解释变量": outcome_display,
                "被解释变量列名": outcome_col,
                "规格": spec_name,
                "观测值": int(res.nobs),
                "R2": res.rsquared,
                "调整后R2": res.rsquared_adj,
                "年份固定效应": "是",
                "arXiv主分类固定效应": "是" if with_topic_fe else "否",
                "公式": formula,
            })

            for param_name in res.params.index:
                coef = res.params[param_name]
                se = res.bse[param_name]
                pval = res.pvalues[param_name]

                label = VAR_LABELS.get(param_name, param_name)
                row = {
                    "被解释变量": outcome_display,
                    "被解释变量列名": outcome_col,
                    "规格": spec_name,
                    "变量": label,
                    "参数名": param_name,
                    "系数": coef,
                    "标准误": se,
                    "p值": pval,
                    "显著性": significance_stars(pval),
                    "格式化": fmt_coef_se(coef, se, pval),
                    "观测值": int(res.nobs),
                    "R2": res.rsquared,
                    "调整后R2": res.rsquared_adj,
                    "年份固定效应": "是",
                    "arXiv主分类固定效应": "是" if with_topic_fe else "否",
                    "公式": formula,
                }
                all_param_rows.append(row)
                if param_name in VAR_LABELS:
                    core_rows.append(row)

    core_df = pd.DataFrame(core_rows)
    full_df = pd.DataFrame(all_param_rows)
    summary_df = pd.DataFrame(summary_rows)
    return core_df, full_df, summary_df


def build_pretty_table(core_df: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
    # 列顺序：每个因变量 2 列（无主题FE / 有主题FE）
    col_defs = []
    for outcome_display, _outcome_col in OUTCOMES.items():
        col_defs.extend([
            (outcome_display, "无主题FE"),
            (outcome_display, "有主题FE"),
        ])

    # 先建空表
    pretty = pd.DataFrame({"变量": DISPLAY_ROW_ORDER})
    for outcome_display, spec in col_defs:
        pretty[f"{outcome_display}_{spec}"] = ""

    # 填系数行
    for _, row in core_df.iterrows():
        row_name = row["变量"]
        col_name = f"{row['被解释变量']}_{row['规格']}"
        pretty.loc[pretty["变量"] == row_name, col_name] = row["格式化"]

    # 填模型统计
    for _, row in summary_df.iterrows():
        col_name = f"{row['被解释变量']}_{row['规格']}"
        pretty.loc[pretty["变量"] == "观测值", col_name] = int(row["观测值"])
        pretty.loc[pretty["变量"] == "R²", col_name] = f"{row['R2']:.4f}"
        pretty.loc[pretty["变量"] == "调整后R²", col_name] = f"{row['调整后R2']:.4f}"
        pretty.loc[pretty["变量"] == "年份固定效应", col_name] = row["年份固定效应"]
        pretty.loc[pretty["变量"] == "arXiv主分类固定效应", col_name] = row["arXiv主分类固定效应"]

    return pretty


def build_readme(input_csv: str, output_xlsx: str) -> pd.DataFrame:
    rows = [
        ["输入文件", input_csv],
        ["输出文件", output_xlsx],
        ["模型", "调节回归：Y ~ HSS_Continuous * hss_reference_share + log_team_size + log_prior_knowledge + C(year) [+ C(arxiv_primary_category)]"],
        ["核心自变量", "HSS_Continuous"],
        ["调节变量", "hss_reference_share"],
        ["交互项", "HSS_Continuous × hss_reference_share"],
        ["控制变量1", "log_team_size"],
        ["控制变量2", "log_prior_knowledge"],
        ["年份固定效应变量", "year"],
        ["arXiv主分类固定效应变量", "arxiv_primary_category"],
        ["稳健标准误", "HC1"],
        ["说明", "Hit_Rate_10_year 继续按 OLS 线性概率模型处理，以保持与你现有表一致"],
        ["说明", "由于 hss_reference_share 有缺失，交互回归样本量会小于基准回归"],
        ["因变量显示名", "CSV 实际列名"],
    ]
    for display, col in OUTCOMES.items():
        rows.append([display, col])
    return pd.DataFrame(rows, columns=["项目", "内容"])


def autofit_excel(writer: pd.ExcelWriter, sheets: List[str]) -> None:
    # openpyxl 引擎下调列宽，纯便利用，不影响回归结果
    for sheet_name in sheets:
        ws = writer.book[sheet_name]
        df = writer.sheets[sheet_name]
        del df  # 仅为保持接口一致
        for col_cells in ws.columns:
            max_len = 0
            col_letter = col_cells[0].column_letter
            for cell in col_cells:
                val = "" if cell.value is None else str(cell.value)
                max_len = max(max_len, len(val))
            ws.column_dimensions[col_letter].width = min(max(max_len + 2, 10), 40)


def main():
    input_csv = Path(sys.argv[1]) if len(sys.argv) >= 2 else Path(DEFAULT_INPUT_CSV)
    output_xlsx = Path(sys.argv[2]) if len(sys.argv) >= 3 else Path(DEFAULT_OUTPUT_XLSX)

    if not input_csv.exists():
        raise FileNotFoundError(f"找不到输入 CSV：{input_csv}")

    df = safe_read_csv(input_csv)
    validate_columns(df, list(OUTCOMES.values()))
    df = prepare_data(df)

    core_df, full_df, summary_df = collect_results(df)
    pretty_df = build_pretty_table(core_df, summary_df)
    readme_df = build_readme(str(input_csv), str(output_xlsx))

    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        pretty_df.to_excel(writer, index=False, sheet_name="interaction_pretty")
        core_df.to_excel(writer, index=False, sheet_name="interaction_core_long")
        full_df.to_excel(writer, index=False, sheet_name="interaction_full_params")
        summary_df.to_excel(writer, index=False, sheet_name="interaction_model_summary")
        readme_df.to_excel(writer, index=False, sheet_name="readme")
        autofit_excel(
            writer,
            [
                "interaction_pretty",
                "interaction_core_long",
                "interaction_full_params",
                "interaction_model_summary",
                "readme",
            ],
        )

    print(f"已输出：{output_xlsx}")


if __name__ == "__main__":
    main()
