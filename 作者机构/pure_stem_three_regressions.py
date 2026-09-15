import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pathlib import Path

# =========================
# 1. 路径设置
# =========================
INPUT_CSV = r"E:\PythonProject\GenAI\data_process\derived\genai_regression_enriched_v3_fixed.csv"
OUTPUT_EXCEL = "sampled_three_regressions.xlsx"
OUTPUT_FILTERED_CSV = "sampled_31423_rows.csv"

# =========================
# 2. 抽样设置
#    team_hss_dna_pct 是百分比：
#    0   = 纯理工科团队
#    100 = 纯HSS团队
# =========================
filter_var = "team_hss_dna_pct"

TARGET_TOTAL = 31423
FILTER_MAX = 50.0           # 只保留 < 50
ZERO_ATOL = 1e-8
RANDOM_SEED = 20260626

# 为了保证最终样本正好 31423，默认允许在某档样本不足时有放回抽样
# 要严格唯一团队，把它改成 False；样本不足时程序会直接报错
ALLOW_REPLACEMENT_IF_NEEDED = True

# 不规则、递减的目标配额
TARGET_COUNTS = {
    "0": 28464,
    "(0,10]": 1031,
    "(10,20]": 793,
    "(20,30]": 547,
    "(30,40]": 349,
    "(40,50)": 239,   # 这里是不含 50 的，严格 < 50
}

BIN_ORDER = ["0", "(0,10]", "(10,20]", "(20,30]", "(30,40]", "(40,50)"]

if sum(TARGET_COUNTS.values()) != TARGET_TOTAL:
    raise ValueError("TARGET_COUNTS 之和必须等于 TARGET_TOTAL。")

# =========================
# 3. 变量设置
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
year_fe = "year"
topic_fe = "arxiv_primary_category"

var_name_map = {
    "HSS_Continuous": "团队HSS知识浓度",
    "hss_reference_share": "HSS参考文献占比",
    "HSS_Continuous:hss_reference_share": "团队HSS知识浓度 × HSS参考文献占比",
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

# =========================
# 4. 基础函数
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

def is_constant(series, atol=1e-12):
    s = series.dropna()
    if len(s) == 0:
        return True
    if s.nunique(dropna=True) <= 1:
        return True
    if pd.api.types.is_numeric_dtype(s):
        return np.nanstd(s.to_numpy(dtype=float)) <= atol
    return False

def interaction_var_name(x, z):
    return f"{x}:{z}"

def get_requested_rhs(model_type):
    if model_type == "baseline":
        return [main_x] + controls
    elif model_type == "interaction":
        return [main_x, moderator, interaction_var_name(main_x, moderator)] + controls
    elif model_type == "hssref":
        return [main_x] + controls
    else:
        raise ValueError("model_type 必须是 baseline / interaction / hssref")

def get_used_columns(dep, model_type):
    cols = [main_x, moderator] + controls + [year_fe, topic_fe, filter_var]
    if model_type in ["baseline", "interaction"]:
        cols = [dep] + cols
    elif model_type == "hssref":
        cols = [moderator] + cols
    return list(dict.fromkeys(cols))

def detect_estimability(data, dep, model_type):
    requested_rhs = get_requested_rhs(model_type)
    statuses = {}

    main_constant = is_constant(data[main_x]) if main_x in data.columns else True
    mod_constant = is_constant(data[moderator]) if moderator in data.columns else True

    if main_x in requested_rhs:
        statuses[main_x] = "omit_no_variation" if main_constant else "estimated"

    if moderator in requested_rhs:
        statuses[moderator] = "omit_no_variation" if mod_constant else "estimated"

    inter = interaction_var_name(main_x, moderator)
    if inter in requested_rhs:
        if main_constant or mod_constant:
            statuses[inter] = "omit_no_variation"
        else:
            prod = data[main_x] * data[moderator]
            statuses[inter] = "omit_no_variation" if is_constant(prod) else "estimated"

    for c in controls:
        statuses[c] = "omit_no_variation" if is_constant(data[c]) else "estimated"

    dep_constant = is_constant(data[dep])
    dep_status = "dep_no_variation" if dep_constant else "estimated"

    used_terms = []

    if model_type == "baseline":
        for v in [main_x] + controls:
            if statuses.get(v) == "estimated":
                used_terms.append(v)

    elif model_type == "interaction":
        for v in [main_x, moderator]:
            if statuses.get(v) == "estimated":
                used_terms.append(v)
        if statuses.get(inter) == "estimated":
            used_terms.append(f"{main_x}:{moderator}")
        for c in controls:
            if statuses.get(c) == "estimated":
                used_terms.append(c)

    elif model_type == "hssref":
        for v in [main_x] + controls:
            if statuses.get(v) == "estimated":
                used_terms.append(v)

    return {
        "dep_status": dep_status,
        "statuses": statuses,
        "used_terms": used_terms
    }

def build_formula(dep, model_type, used_terms, add_topic_fe=False):
    rhs_parts = list(used_terms)
    rhs_parts.append(f"C({year_fe})")
    if add_topic_fe:
        rhs_parts.append(f"C({topic_fe})")

    rhs = " + ".join(rhs_parts) if rhs_parts else "1"

    if model_type in ["baseline", "interaction"]:
        return f"{dep} ~ {rhs}"
    elif model_type == "hssref":
        return f"{moderator} ~ {rhs}"
    else:
        raise ValueError("model_type 必须是 baseline / interaction / hssref")

def run_model(dep, model_type="baseline", add_topic_fe=False, df=None):
    use_cols = get_used_columns(dep, model_type)
    data = df[use_cols].dropna().copy()

    data[year_fe] = data[year_fe].astype("category")
    data[topic_fe] = data[topic_fe].astype("category")

    diag = detect_estimability(
        data=data,
        dep=(dep if model_type in ["baseline", "interaction"] else moderator),
        model_type=model_type
    )

    formula = build_formula(
        dep=dep,
        model_type=model_type,
        used_terms=diag["used_terms"],
        add_topic_fe=add_topic_fe
    )

    if diag["dep_status"] == "dep_no_variation":
        return {
            "model": None,
            "nobs": int(data.shape[0]),
            "r2": np.nan,
            "formula": formula,
            "statuses": diag["statuses"],
            "note": "因变量在该子样本中无变异，模型未估计。"
        }

    model = smf.ols(formula=formula, data=data).fit(cov_type="HC1")

    return {
        "model": model,
        "nobs": int(data.shape[0]),
        "r2": float(model.rsquared),
        "formula": formula,
        "statuses": diag["statuses"],
        "note": ""
    }

def value_for_var(spec, var):
    model = spec["model"]
    statuses = spec["statuses"]
    status = statuses.get(var, "estimated")

    if status == "omit_no_variation":
        return "不可估计（无变异）", ""
    if model is None:
        return "未估计", ""

    if var == interaction_var_name(main_x, moderator):
        cand_names = [f"{main_x}:{moderator}", f"{moderator}:{main_x}"]
    else:
        cand_names = [var]

    found_name = None
    for name in cand_names:
        if name in model.params.index:
            found_name = name
            break

    if found_name is None:
        return "", ""

    coef = fmt_coef(model.params.get(found_name, np.nan), model.pvalues.get(found_name, np.nan))
    se = fmt_se(model.bse.get(found_name, np.nan))
    return coef, se

def make_regression_table(model_specs, row_order):
    rows = []

    for var in row_order:
        coef_row = {"变量": var_name_map.get(var, var)}
        se_row = {"变量": ""}

        for spec in model_specs:
            coef, se = value_for_var(spec, var)
            coef_row[spec["col_name"]] = coef
            se_row[spec["col_name"]] = se

        rows.append(coef_row)
        rows.append(se_row)

    summary_labels = [
        "控制变量",
        "年份固定效应",
        "主分类固定效应",
        "样本量 N",
        "R²"
    ]

    for label in summary_labels:
        row = {"变量": label}
        for spec in model_specs:
            if label == "控制变量":
                row[spec["col_name"]] = "Yes"
            elif label == "年份固定效应":
                row[spec["col_name"]] = "Yes"
            elif label == "主分类固定效应":
                row[spec["col_name"]] = spec["topic_fe"]
            elif label == "样本量 N":
                row[spec["col_name"]] = spec["nobs"]
            elif label == "R²":
                row[spec["col_name"]] = "" if pd.isna(spec["r2"]) else f"{spec['r2']:.4f}"
        rows.append(row)

    return pd.DataFrame(rows)

def build_baseline_table(df_sub):
    model_specs = []
    for dep in y_vars:
        m1 = run_model(dep, model_type="baseline", add_topic_fe=False, df=df_sub)
        model_specs.append({
            "col_name": f"{dep_name_map.get(dep, dep)}_1",
            "topic_fe": "No",
            **m1
        })

        m2 = run_model(dep, model_type="baseline", add_topic_fe=True, df=df_sub)
        model_specs.append({
            "col_name": f"{dep_name_map.get(dep, dep)}_2",
            "topic_fe": "Yes",
            **m2
        })

    row_order = [main_x, "log_team_size", "log_prior_knowledge", "Intercept"]
    table = make_regression_table(model_specs, row_order)
    return table, model_specs

def build_interaction_table(df_sub):
    model_specs = []
    for dep in y_vars:
        m1 = run_model(dep, model_type="interaction", add_topic_fe=False, df=df_sub)
        model_specs.append({
            "col_name": f"{dep_name_map.get(dep, dep)}_1",
            "topic_fe": "No",
            **m1
        })

        m2 = run_model(dep, model_type="interaction", add_topic_fe=True, df=df_sub)
        model_specs.append({
            "col_name": f"{dep_name_map.get(dep, dep)}_2",
            "topic_fe": "Yes",
            **m2
        })

    row_order = [
        main_x,
        moderator,
        interaction_var_name(main_x, moderator),
        "log_team_size",
        "log_prior_knowledge",
        "Intercept"
    ]
    table = make_regression_table(model_specs, row_order)
    return table, model_specs

def build_hssref_table(df_sub):
    model_specs = []

    m1 = run_model(moderator, model_type="hssref", add_topic_fe=False, df=df_sub)
    model_specs.append({
        "col_name": "HSS参考文献占比_1",
        "topic_fe": "No",
        **m1
    })

    m2 = run_model(moderator, model_type="hssref", add_topic_fe=True, df=df_sub)
    model_specs.append({
        "col_name": "HSS参考文献占比_2",
        "topic_fe": "Yes",
        **m2
    })

    row_order = [main_x, "log_team_size", "log_prior_knowledge", "Intercept"]
    table = make_regression_table(model_specs, row_order)
    return table, model_specs

# =========================
# 5. 抽样函数
# =========================
def assign_sampling_bin(series):
    s = pd.to_numeric(series, errors="coerce")

    out = pd.Series(pd.NA, index=series.index, dtype="object")

    out[np.isclose(s, 0.0, atol=ZERO_ATOL)] = "0"
    out[(s > 0.0) & (s <= 10.0)] = "(0,10]"
    out[(s > 10.0) & (s <= 20.0)] = "(10,20]"
    out[(s > 20.0) & (s <= 30.0)] = "(20,30]"
    out[(s > 30.0) & (s <= 40.0)] = "(30,40]"
    out[(s > 40.0) & (s < FILTER_MAX)] = "(40,50)"   # 严格小于50

    return out


def compute_target_counts():
    targets = TARGET_COUNTS.copy()

    # 顺手做个校验：必须递减
    nonzero_bins = [b for b in BIN_ORDER if b != "0"]
    nonzero_vals = [targets[b] for b in nonzero_bins]
    for i in range(len(nonzero_vals) - 1):
        if nonzero_vals[i] <= nonzero_vals[i + 1]:
            raise ValueError("非零分箱目标数必须严格递减。")

    if sum(targets.values()) != TARGET_TOTAL:
        raise ValueError("目标配额之和不等于 TARGET_TOTAL。")

    return targets

def draw_one_bin(subdf, n_target, random_state, label):
    if n_target == 0:
        return subdf.iloc[0:0].copy(), False

    if len(subdf) == 0:
        raise ValueError(f"分组 {label} 没有可抽样观测，无法完成目标抽样。")

    replace = len(subdf) < n_target

    if replace and not ALLOW_REPLACEMENT_IF_NEEDED:
        raise ValueError(
            f"分组 {label} 可用样本仅 {len(subdf)} 条，目标 {n_target} 条；"
            f"当前设定不允许有放回抽样。"
        )

    if replace:
        print(f"警告：分组 {label} 可用样本仅 {len(subdf)} 条，目标 {n_target} 条，将使用有放回抽样补足。")

    drawn = subdf.sample(
        n=n_target,
        replace=replace,
        random_state=random_state
    ).copy()

    return drawn, replace

def build_sample(df, filter_var):
    work = df.copy()

    # 记录原始行号，便于检查是否出现重复抽样
    work = work.reset_index(drop=True)
    work["orig_row_id"] = work.index

    work[filter_var] = pd.to_numeric(work[filter_var], errors="coerce")
    work["sampling_bin"] = assign_sampling_bin(work[filter_var])

    # 只保留 0 <= team_hss_dna_pct < 50 且分箱成功的
    eligible = work[work["sampling_bin"].notna()].copy()

    target_counts = compute_target_counts()

    # 检查目标分箱是否都存在
    for b, n_target in target_counts.items():
        if n_target > 0 and (eligible["sampling_bin"] == b).sum() == 0:
            raise ValueError(f"分组 {b} 在原始数据中没有观测，无法完成目标抽样。")

    rng = np.random.default_rng(RANDOM_SEED)
    sampled_parts = []
    draw_log = []

    for b in BIN_ORDER:
        sub = eligible[eligible["sampling_bin"] == b].copy()
        rs = int(rng.integers(1, 10**9))
        n_target = target_counts.get(b, 0)

        drawn, replace = draw_one_bin(sub, n_target, rs, b)
        sampled_parts.append(drawn)

        draw_log.append({
            "bin": b,
            "available_n": len(sub),
            "target_n": n_target,
            "actual_n": len(drawn),
            "with_replacement": replace
        })

    sampled = pd.concat(sampled_parts, axis=0, ignore_index=True)

    # 打乱顺序
    sampled = sampled.sample(
        frac=1,
        random_state=int(rng.integers(1, 10**9))
    ).reset_index(drop=True)

    return eligible, sampled, pd.DataFrame(draw_log), target_counts

def build_diagnostics(df_all, df_eligible, df_sub, draw_log):
    inter = interaction_var_name(main_x, moderator)
    if moderator in df_sub.columns:
        inter_series = df_sub[main_x] * df_sub[moderator]
    else:
        inter_series = pd.Series(dtype=float)

    duplicated_draws = len(df_sub) - df_sub["orig_row_id"].nunique()

    diag_rows = [
        ["原始样本量", len(df_all)],
        [f"可抽样样本量（0 ≤ {filter_var} < 50）", len(df_eligible)],
        ["最终模拟样本量", len(df_sub)],
        ["最终模拟样本去重后原始团队数", df_sub["orig_row_id"].nunique()],
        ["重复抽样条数（若>0表示某些分箱样本不足，启用了有放回抽样）", duplicated_draws],
        [f"{main_x} 唯一值个数", int(df_sub[main_x].nunique(dropna=True))],
        [f"{main_x} 是否无变异", "Yes" if is_constant(df_sub[main_x]) else "No"],
        [f"{moderator} 唯一值个数", int(df_sub[moderator].nunique(dropna=True))],
        [f"{moderator} 是否无变异", "Yes" if is_constant(df_sub[moderator]) else "No"],
        [f"{inter} 是否无变异", "Yes" if is_constant(inter_series) else "No"],
    ]

    for _, row in draw_log.iterrows():
        diag_rows.append([f"{row['bin']} 可用样本量", int(row["available_n"])])
        diag_rows.append([f"{row['bin']} 目标抽样量", int(row["target_n"])])
        diag_rows.append([f"{row['bin']} 实际抽样量", int(row["actual_n"])])
        diag_rows.append([f"{row['bin']} 是否有放回抽样", "Yes" if row["with_replacement"] else "No"])

    diag_rows.append([
        "说明",
        "本脚本按 team_hss_dna_pct 分箱抽样：0组=28464，其余(0,10]、(10,20]、(20,30]、(30,40]、(40,50) 采用不规则递减配额 1031、793、547、349、239，总量固定为31423。"
    ])

    return pd.DataFrame(diag_rows, columns=["项目", "结果"])

# =========================
# 6. 读取数据并抽样
# =========================
input_path = Path(INPUT_CSV)
if not input_path.exists():
    raise FileNotFoundError(f"找不到输入文件: {INPUT_CSV}")

df = pd.read_csv(input_path)

required_cols = (
    y_vars
    + [main_x, moderator]
    + controls
    + [year_fe, topic_fe, filter_var]
)

missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"缺少变量: {missing}")

df_eligible, df_sub, draw_log, target_counts = build_sample(df, filter_var=filter_var)

# 导出抽样样本
df_sub.to_csv(OUTPUT_FILTERED_CSV, index=False, encoding="utf-8-sig")

print("========== 原始样本量 ==========")
print(len(df))

print(f"\n========== 可抽样样本量（0 ≤ {filter_var} < 50） ==========")
print(len(df_eligible))

print("\n========== 目标抽样配额 ==========")
for b in BIN_ORDER:
    print(f"{b}: {target_counts.get(b, 0)}")

print("\n========== 实际抽样情况 ==========")
print(draw_log.to_string(index=False))

print("\n========== 最终模拟样本量 ==========")
print(len(df_sub))

print("\n========== 最终样本分箱分布 ==========")
print(df_sub["sampling_bin"].value_counts().reindex(BIN_ORDER, fill_value=0))

print("\n========== 关键变量变异 ==========")
for col in [main_x, moderator]:
    print(f"{col}: nunique={df_sub[col].nunique(dropna=True)}, constant={is_constant(df_sub[col])}")
print(f"{main_x}:{moderator}: constant={is_constant(df_sub[main_x] * df_sub[moderator])}")

# =========================
# 7. 三张回归表
# =========================
baseline_table, baseline_specs = build_baseline_table(df_sub)
interaction_table, interaction_specs = build_interaction_table(df_sub)
hssref_table, hssref_specs = build_hssref_table(df_sub)

diag_table = build_diagnostics(df, df_eligible, df_sub, draw_log)

print("\n========== 诊断信息 ==========")
print(diag_table.to_string(index=False))

# =========================
# 8. 导出 Excel
# =========================
with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
    baseline_table.to_excel(writer, sheet_name="基准回归表", index=False)
    interaction_table.to_excel(writer, sheet_name="交互回归表", index=False)
    hssref_table.to_excel(writer, sheet_name="HSS参考文献占比回归", index=False)

print(f"\n已输出抽样样本: {OUTPUT_FILTERED_CSV}")
print(f"已输出结果表: {OUTPUT_EXCEL}")
