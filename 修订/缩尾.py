import pandas as pd
from pathlib import Path


# ============================================================
# 1. 路径
# ============================================================

INPUT_CSV = (
    r"E:\PythonProject\GenAI\data_process\derived"
    r"\genai_regression_enriched_v3_fixed_winsor1pct.csv"
)


OUTPUT_CSV = (
    r"E:\PythonProject\GenAI\data_process\derived"
    r"\genai_regression_enriched_v3_fixed_winsor1pct.csv"
)

OUTPUT_LOG = (
    r"E:\PythonProject\GenAI\data_process\derived"
    r"\winsor1pct_log.csv"
)


# ============================================================
# 2. 缩尾比例
# ============================================================

LOWER_Q = 0.01
UPPER_Q = 0.99


# ============================================================
# 3. 需要缩尾的连续变量
# ============================================================

WINSOR_COLS = [

    # -------------------------
    # 因变量
    # -------------------------

    "Log_RCR",

    "Citation_Field_Diversity",

    "HSS_Citing_Share",

    "Citation_Entropy",

    # -------------------------
    # 核心解释变量
    # -------------------------

    "HSS_Continuous",

    # -------------------------
    # HSS参考文献变量
    # -------------------------

    "hss_reference_share",

    # -------------------------
    # 控制变量
    # -------------------------

    "log_team_size",

    "log_prior_knowledge"
]


# ============================================================
# 4. 为什么没有 Hit_Rate_10_year？
# ============================================================

# 如果 Hit_Rate_10_year 是 0/1 二元变量，
# 不应该对它进行缩尾。
#
# 如果你的 Hit_Rate_10_year 实际上不是0/1，
# 而是连续比例变量，则可以把：
#
# "Hit_Rate_10_year"
#
# 加入 WINSOR_COLS。


# ============================================================
# 5. 读取数据
# ============================================================

input_path = Path(
    INPUT_CSV
)

if not input_path.exists():

    raise FileNotFoundError(
        f"找不到输入文件：{INPUT_CSV}"
    )


print(
    "========== 正在读取原始数据 =========="
)

df = pd.read_csv(
    input_path,
    low_memory=False
)

print(
    f"原始样本量：{len(df):,}"
)


# ============================================================
# 6. 检查变量
# ============================================================

missing_cols = [
    col
    for col in WINSOR_COLS
    if col not in df.columns
]

if missing_cols:

    raise ValueError(
        f"以下缩尾变量不存在：{missing_cols}"
    )


# ============================================================
# 7. 创建副本
# ============================================================

df_w = df.copy()


# ============================================================
# 8. 执行1%缩尾
# ============================================================

log_rows = []

print(
    "\n========== 开始1%双侧缩尾 =========="
)

for col in WINSOR_COLS:

    # 转为数值
    s = pd.to_numeric(
        df_w[col],
        errors="coerce"
    )

    valid = s.dropna()

    if len(valid) == 0:

        print(
            f"{col}: 无有效数值，跳过。"
        )

        log_rows.append({
            "变量": col,
            "有效样本量": 0,
            "P1": None,
            "P99": None,
            "低于P1数量": 0,
            "高于P99数量": 0,
            "总缩尾数量": 0
        })

        continue

    # ----------------------------------
    # 计算1%和99%分位点
    # ----------------------------------

    lower_value = valid.quantile(
        LOWER_Q
    )

    upper_value = valid.quantile(
        UPPER_Q
    )

    # ----------------------------------
    # 统计真正会被修改的观测
    # ----------------------------------

    lower_n = int(
        (s < lower_value).sum()
    )

    upper_n = int(
        (s > upper_value).sum()
    )

    # ----------------------------------
    # Winsorize
    #
    # 小于P1 → P1
    # 大于P99 → P99
    # 中间保持不变
    # NaN保持NaN
    # ----------------------------------

    df_w[col] = s.clip(
        lower=lower_value,
        upper=upper_value
    )

    log_rows.append({
        "变量": col,
        "有效样本量": int(valid.shape[0]),
        "P1": lower_value,
        "P99": upper_value,
        "低于P1数量": lower_n,
        "高于P99数量": upper_n,
        "总缩尾数量":
            lower_n + upper_n
    })

    print(
        f"{col}: "
        f"P1={lower_value:.6f}, "
        f"P99={upper_value:.6f}, "
        f"下端处理={lower_n:,}, "
        f"上端处理={upper_n:,}"
    )


# ============================================================
# 9. 保存缩尾数据
# ============================================================

df_w.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 10. 保存缩尾日志
# ============================================================

winsor_log = pd.DataFrame(
    log_rows
)

winsor_log.to_csv(
    OUTPUT_LOG,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 11. 输出结果
# ============================================================

print(
    "\n========== 缩尾完成 =========="
)

print(
    f"原始样本量：{len(df):,}"
)

print(
    f"缩尾后样本量：{len(df_w):,}"
)

print(
    "注意：Winsorize不会删除任何观测，"
    "因此样本量应保持完全不变。"
)

print(
    f"\n缩尾数据已输出：\n{OUTPUT_CSV}"
)

print(
    f"\n缩尾日志已输出：\n{OUTPUT_LOG}"
)

print(
    "\n========== 缩尾统计 =========="
)

print(
    winsor_log.to_string(
        index=False
    )
)
