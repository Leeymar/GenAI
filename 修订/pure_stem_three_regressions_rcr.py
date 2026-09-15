import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pathlib import Path

# ============================================================
# 1. 路径设置
# ============================================================

# 直接使用已经抽好的 31423 条样本
INPUT_CSV = r"E:\PythonProject\GenAI\作者机构\sampled_31423_rows.csv"

# 回归结果输出
OUTPUT_EXCEL = "sampled_three_regressions.xlsx"


# ============================================================
# 2. 变量设置
# ============================================================

# 五个主要因变量
y_vars = [
    "Log_RCR",
    "Hit_Rate_10_year",
    "Citation_Field_Diversity",
    "HSS_Citing_Share",
    "Citation_Entropy"
]

# 核心解释变量
main_x = "HSS_Continuous"

# 调节变量 / 机制变量
moderator = "hss_reference_share"

# 控制变量
controls = [
    "log_team_size",
    "log_prior_knowledge"
]

# 固定效应
year_fe = "year"
topic_fe = "arxiv_primary_category"


# ============================================================
# 3. 表格显示名称
# ============================================================

var_name_map = {
    "HSS_Continuous": "团队HSS知识浓度",
    "hss_reference_share": "HSS参考文献占比",
    "HSS_Continuous:hss_reference_share":
        "团队HSS知识浓度 × HSS参考文献占比",
    "log_team_size": "团队规模（对数）",
    "log_prior_knowledge": "既有知识存量（对数）",
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


# ============================================================
# 4. 基础函数
# ============================================================

def star(p):
    """
    根据 p 值生成显著性星号
    """
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
    """
    格式化回归系数
    """
    if pd.isna(x):
        return ""

    return f"{x:.4f}{star(p)}"


def fmt_se(x):
    """
    格式化标准误
    """
    if pd.isna(x):
        return ""

    return f"({x:.4f})"


def is_constant(series, atol=1e-12):
    """
    检查变量在当前回归样本中是否没有变异
    """
    s = series.dropna()

    if len(s) == 0:
        return True

    if s.nunique(dropna=True) <= 1:
        return True

    if pd.api.types.is_numeric_dtype(s):
        return np.nanstd(
            s.to_numpy(dtype=float)
        ) <= atol

    return False


def interaction_var_name(x, z):
    """
    返回交互项名称
    """
    return f"{x}:{z}"


# ============================================================
# 5. 获取模型所需变量
# ============================================================

def get_requested_rhs(model_type):
    """
    返回理论上需要进入模型的解释变量
    """

    if model_type == "baseline":

        return [
            main_x
        ] + controls

    elif model_type == "interaction":

        return [
            main_x,
            moderator,
            interaction_var_name(main_x, moderator)
        ] + controls

    elif model_type == "hssref":

        return [
            main_x
        ] + controls

    else:

        raise ValueError(
            "model_type 必须是 baseline / interaction / hssref"
        )


def get_used_columns(dep, model_type):
    """
    返回回归前需要从 DataFrame 中提取的变量
    """

    if model_type in ["baseline", "interaction"]:

        cols = [
            dep,
            main_x,
            moderator,
            *controls,
            year_fe,
            topic_fe
        ]

    elif model_type == "hssref":

        cols = [
            moderator,
            main_x,
            *controls,
            year_fe,
            topic_fe
        ]

    else:

        raise ValueError(
            "model_type 必须是 baseline / interaction / hssref"
        )

    # 去重并保持原顺序
    return list(dict.fromkeys(cols))


# ============================================================
# 6. 判断变量是否可以估计
# ============================================================

def detect_estimability(data, dep, model_type):
    """
    自动检查：
    1. 因变量是否有变异
    2. 核心解释变量是否有变异
    3. 调节变量是否有变异
    4. 交互项是否有变异
    5. 控制变量是否有变异

    若没有变异，则从回归公式中自动移除。
    """

    requested_rhs = get_requested_rhs(model_type)

    statuses = {}

    # -------------------------------
    # 核心变量
    # -------------------------------

    main_constant = (
        is_constant(data[main_x])
        if main_x in data.columns
        else True
    )

    mod_constant = (
        is_constant(data[moderator])
        if moderator in data.columns
        else True
    )

    if main_x in requested_rhs:

        statuses[main_x] = (
            "omit_no_variation"
            if main_constant
            else "estimated"
        )

    if moderator in requested_rhs:

        statuses[moderator] = (
            "omit_no_variation"
            if mod_constant
            else "estimated"
        )

    # -------------------------------
    # 交互项
    # -------------------------------

    inter = interaction_var_name(
        main_x,
        moderator
    )

    if inter in requested_rhs:

        if main_constant or mod_constant:

            statuses[inter] = (
                "omit_no_variation"
            )

        else:

            prod = (
                data[main_x]
                * data[moderator]
            )

            statuses[inter] = (
                "omit_no_variation"
                if is_constant(prod)
                else "estimated"
            )

    # -------------------------------
    # 控制变量
    # -------------------------------

    for c in controls:

        statuses[c] = (
            "omit_no_variation"
            if is_constant(data[c])
            else "estimated"
        )

    # -------------------------------
    # 因变量
    # -------------------------------

    dep_constant = is_constant(
        data[dep]
    )

    dep_status = (
        "dep_no_variation"
        if dep_constant
        else "estimated"
    )

    # -------------------------------
    # 最终进入公式的变量
    # -------------------------------

    used_terms = []

    if model_type == "baseline":

        for v in [
            main_x,
            *controls
        ]:

            if statuses.get(v) == "estimated":
                used_terms.append(v)

    elif model_type == "interaction":

        for v in [
            main_x,
            moderator
        ]:

            if statuses.get(v) == "estimated":
                used_terms.append(v)

        if statuses.get(inter) == "estimated":

            used_terms.append(
                f"{main_x}:{moderator}"
            )

        for c in controls:

            if statuses.get(c) == "estimated":
                used_terms.append(c)

    elif model_type == "hssref":

        for v in [
            main_x,
            *controls
        ]:

            if statuses.get(v) == "estimated":
                used_terms.append(v)

    return {
        "dep_status": dep_status,
        "statuses": statuses,
        "used_terms": used_terms
    }


# ============================================================
# 7. 构造回归公式
# ============================================================

def build_formula(
    dep,
    model_type,
    used_terms,
    add_topic_fe=False
):

    rhs_parts = list(
        used_terms
    )

    # 年份固定效应始终加入
    rhs_parts.append(
        f"C({year_fe})"
    )

    # 第二组模型加入主分类固定效应
    if add_topic_fe:

        rhs_parts.append(
            f"C({topic_fe})"
        )

    rhs = (
        " + ".join(rhs_parts)
        if rhs_parts
        else "1"
    )

    if model_type in [
        "baseline",
        "interaction"
    ]:

        return (
            f"{dep} ~ {rhs}"
        )

    elif model_type == "hssref":

        return (
            f"{moderator} ~ {rhs}"
        )

    else:

        raise ValueError(
            "model_type 必须是 baseline / interaction / hssref"
        )


# ============================================================
# 8. 执行单个回归
# ============================================================

def run_model(
    dep,
    model_type="baseline",
    add_topic_fe=False,
    df=None
):

    # -------------------------------
    # 获取模型所需数据
    # -------------------------------

    use_cols = get_used_columns(
        dep,
        model_type
    )

    # 与 statsmodels 默认 complete-case 逻辑一致
    data = (
        df[use_cols]
        .dropna()
        .copy()
    )

    # -------------------------------
    # 检查是否还有样本
    # -------------------------------

    if len(data) == 0:

        return {
            "model": None,
            "nobs": 0,
            "r2": np.nan,
            "formula": "",
            "statuses": {},
            "note": "缺失值删除后无有效样本。"
        }

    # -------------------------------
    # 转为类别变量
    # -------------------------------

    data[year_fe] = (
        data[year_fe]
        .astype("category")
    )

    data[topic_fe] = (
        data[topic_fe]
        .astype("category")
    )

    # -------------------------------
    # 判断模型变量是否可估计
    # -------------------------------

    actual_dep = (
        dep
        if model_type in [
            "baseline",
            "interaction"
        ]
        else moderator
    )

    diag = detect_estimability(
        data=data,
        dep=actual_dep,
        model_type=model_type
    )

    # -------------------------------
    # 构造公式
    # -------------------------------

    formula = build_formula(
        dep=dep,
        model_type=model_type,
        used_terms=diag["used_terms"],
        add_topic_fe=add_topic_fe
    )

    # -------------------------------
    # 因变量无变异则不估计
    # -------------------------------

    if (
        diag["dep_status"]
        == "dep_no_variation"
    ):

        return {
            "model": None,
            "nobs": int(
                data.shape[0]
            ),
            "r2": np.nan,
            "formula": formula,
            "statuses": diag["statuses"],
            "note":
                "因变量在该样本中无变异，模型未估计。"
        }

    # -------------------------------
    # OLS + HC1 稳健标准误
    # -------------------------------

    model = smf.ols(
        formula=formula,
        data=data
    ).fit(
        cov_type="HC1"
    )

    return {
        "model": model,
        "nobs": int(
            model.nobs
        ),
        "r2": float(
            model.rsquared
        ),
        "formula": formula,
        "statuses": diag["statuses"],
        "note": ""
    }


# ============================================================
# 9. 从模型中提取结果
# ============================================================

def value_for_var(spec, var):

    model = spec["model"]

    statuses = spec[
        "statuses"
    ]

    status = statuses.get(
        var,
        "estimated"
    )

    if status == "omit_no_variation":

        return (
            "不可估计（无变异）",
            ""
        )

    if model is None:

        return (
            "未估计",
            ""
        )

    # statsmodels 可能交换交互项变量顺序
    if var == interaction_var_name(
        main_x,
        moderator
    ):

        cand_names = [
            f"{main_x}:{moderator}",
            f"{moderator}:{main_x}"
        ]

    else:

        cand_names = [
            var
        ]

    found_name = None

    for name in cand_names:

        if name in model.params.index:

            found_name = name
            break

    if found_name is None:

        return "", ""

    coef = fmt_coef(
        model.params.get(
            found_name,
            np.nan
        ),
        model.pvalues.get(
            found_name,
            np.nan
        )
    )

    se = fmt_se(
        model.bse.get(
            found_name,
            np.nan
        )
    )

    return coef, se


# ============================================================
# 10. 制作回归表
# ============================================================

def make_regression_table(
    model_specs,
    row_order
):

    rows = []

    # -------------------------------
    # 回归系数和标准误
    # -------------------------------

    for var in row_order:

        coef_row = {
            "变量":
                var_name_map.get(
                    var,
                    var
                )
        }

        se_row = {
            "变量": ""
        }

        for spec in model_specs:

            coef, se = value_for_var(
                spec,
                var
            )

            coef_row[
                spec["col_name"]
            ] = coef

            se_row[
                spec["col_name"]
            ] = se

        rows.append(
            coef_row
        )

        rows.append(
            se_row
        )

    # -------------------------------
    # 模型统计信息
    # -------------------------------

    summary_labels = [
        "控制变量",
        "年份固定效应",
        "主分类固定效应",
        "样本量 N",
        "R²"
    ]

    for label in summary_labels:

        row = {
            "变量": label
        }

        for spec in model_specs:

            if label == "控制变量":

                row[
                    spec["col_name"]
                ] = "Yes"

            elif label == "年份固定效应":

                row[
                    spec["col_name"]
                ] = "Yes"

            elif label == "主分类固定效应":

                row[
                    spec["col_name"]
                ] = spec["topic_fe"]

            elif label == "样本量 N":

                row[
                    spec["col_name"]
                ] = spec["nobs"]

            elif label == "R²":

                row[
                    spec["col_name"]
                ] = (
                    ""
                    if pd.isna(
                        spec["r2"]
                    )
                    else
                    f"{spec['r2']:.4f}"
                )

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# 11. 基准回归
# ============================================================

def build_baseline_table(df_sub):

    model_specs = []

    for dep in y_vars:

        # ----------------------------------
        # 模型1：
        # 年份固定效应
        # 不加主分类固定效应
        # ----------------------------------

        m1 = run_model(
            dep,
            model_type="baseline",
            add_topic_fe=False,
            df=df_sub
        )

        model_specs.append({
            "col_name":
                f"{dep_name_map.get(dep, dep)}_1",
            "topic_fe": "No",
            **m1
        })

        # ----------------------------------
        # 模型2：
        # 年份 + 主分类固定效应
        # ----------------------------------

        m2 = run_model(
            dep,
            model_type="baseline",
            add_topic_fe=True,
            df=df_sub
        )

        model_specs.append({
            "col_name":
                f"{dep_name_map.get(dep, dep)}_2",
            "topic_fe": "Yes",
            **m2
        })

    row_order = [
        main_x,
        "log_team_size",
        "log_prior_knowledge",
        "Intercept"
    ]

    table = make_regression_table(
        model_specs,
        row_order
    )

    return table, model_specs


# ============================================================
# 12. 交互项回归
# ============================================================

def build_interaction_table(df_sub):

    model_specs = []

    for dep in y_vars:

        # ----------------------------------
        # 模型1：
        # 年份固定效应
        # ----------------------------------

        m1 = run_model(
            dep,
            model_type="interaction",
            add_topic_fe=False,
            df=df_sub
        )

        model_specs.append({
            "col_name":
                f"{dep_name_map.get(dep, dep)}_1",
            "topic_fe": "No",
            **m1
        })

        # ----------------------------------
        # 模型2：
        # 年份 + 主分类固定效应
        # ----------------------------------

        m2 = run_model(
            dep,
            model_type="interaction",
            add_topic_fe=True,
            df=df_sub
        )

        model_specs.append({
            "col_name":
                f"{dep_name_map.get(dep, dep)}_2",
            "topic_fe": "Yes",
            **m2
        })

    row_order = [
        main_x,
        moderator,
        interaction_var_name(
            main_x,
            moderator
        ),
        "log_team_size",
        "log_prior_knowledge",
        "Intercept"
    ]

    table = make_regression_table(
        model_specs,
        row_order
    )

    return table, model_specs


# ============================================================
# 13. HSS参考文献占比回归
# ============================================================

def build_hssref_table(df_sub):

    model_specs = []

    # ----------------------------------
    # 模型1：
    # 仅年份固定效应
    # ----------------------------------

    m1 = run_model(
        moderator,
        model_type="hssref",
        add_topic_fe=False,
        df=df_sub
    )

    model_specs.append({
        "col_name":
            "HSS参考文献占比_1",
        "topic_fe": "No",
        **m1
    })

    # ----------------------------------
    # 模型2：
    # 年份 + 主分类固定效应
    # ----------------------------------

    m2 = run_model(
        moderator,
        model_type="hssref",
        add_topic_fe=True,
        df=df_sub
    )

    model_specs.append({
        "col_name":
            "HSS参考文献占比_2",
        "topic_fe": "Yes",
        **m2
    })

    row_order = [
        main_x,
        "log_team_size",
        "log_prior_knowledge",
        "Intercept"
    ]

    table = make_regression_table(
        model_specs,
        row_order
    )

    return table, model_specs


# ============================================================
# 14. 简单样本诊断
# ============================================================

def build_diagnostics(df_sub):

    inter = (
        df_sub[main_x]
        * df_sub[moderator]
    )

    diag_rows = [
        [
            "输入样本总量",
            len(df_sub)
        ],
        [
            f"{main_x} 非缺失样本",
            df_sub[main_x].notna().sum()
        ],
        [
            f"{main_x} 唯一值个数",
            df_sub[main_x].nunique(
                dropna=True
            )
        ],
        [
            f"{main_x} 是否无变异",
            "Yes"
            if is_constant(
                df_sub[main_x]
            )
            else "No"
        ],
        [
            f"{moderator} 非缺失样本",
            df_sub[moderator].notna().sum()
        ],
        [
            f"{moderator} 唯一值个数",
            df_sub[moderator].nunique(
                dropna=True
            )
        ],
        [
            f"{moderator} 是否无变异",
            "Yes"
            if is_constant(
                df_sub[moderator]
            )
            else "No"
        ],
        [
            f"{main_x}:{moderator} 是否无变异",
            "Yes"
            if is_constant(inter)
            else "No"
        ],
        [
            year_fe + " 类别数",
            df_sub[year_fe].nunique(
                dropna=True
            )
        ],
        [
            topic_fe + " 类别数",
            df_sub[topic_fe].nunique(
                dropna=True
            )
        ],
        [
            "说明",
            "本脚本不再进行抽样，直接使用 INPUT_CSV 中的全部样本进行回归；各模型因缺失值不同，实际回归样本量 N 可能低于输入样本总量。"
        ]
    ]

    return pd.DataFrame(
        diag_rows,
        columns=[
            "项目",
            "结果"
        ]
    )


# ============================================================
# 15. 读取数据
# ============================================================

input_path = Path(
    INPUT_CSV
)

if not input_path.exists():

    raise FileNotFoundError(
        f"找不到输入文件: {INPUT_CSV}"
    )


df = pd.read_csv(
    input_path
)

df["Log_RCR"] = np.log(df["RCR"] + 0.01)

# ============================================================
# 16. 检查所需变量
# ============================================================

required_cols = (
    y_vars
    + [
        main_x,
        moderator
    ]
    + controls
    + [
        year_fe,
        topic_fe
    ]
)

missing = [
    c
    for c in required_cols
    if c not in df.columns
]

if missing:

    raise ValueError(
        f"缺少变量: {missing}"
    )


# ============================================================
# 17. 基础信息
# ============================================================

print(
    "\n========== 输入样本 =========="
)

print(
    f"文件: {INPUT_CSV}"
)

print(
    f"样本量: {len(df):,}"
)


print(
    "\n========== 核心变量 =========="
)

for col in [
    main_x,
    moderator
]:

    print(
        f"{col}: "
        f"N={df[col].notna().sum():,}, "
        f"nunique={df[col].nunique(dropna=True):,}, "
        f"constant={is_constant(df[col])}"
    )


# ============================================================
# 18. 三组回归
# ============================================================

print(
    "\n========== 正在运行基准回归 =========="
)

baseline_table, baseline_specs = (
    build_baseline_table(df)
)


print(
    "\n========== 正在运行交互项回归 =========="
)

interaction_table, interaction_specs = (
    build_interaction_table(df)
)


print(
    "\n========== 正在运行HSS参考文献占比回归 =========="
)

hssref_table, hssref_specs = (
    build_hssref_table(df)
)


# ============================================================
# 19. 样本诊断
# ============================================================

diag_table = build_diagnostics(
    df
)


# ============================================================
# 20. 输出 Excel
# ============================================================

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

    diag_table.to_excel(
        writer,
        sheet_name="样本诊断",
        index=False
    )


print(
    "\n========== 完成 =========="
)

print(
    f"输入样本量: {len(df):,}"
)

print(
    f"回归结果已输出: {OUTPUT_EXCEL}"
)
