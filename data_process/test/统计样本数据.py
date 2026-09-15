import pandas as pd
import numpy as np


def calculate_grouped_statistics():
    print("📊 启动【表 2：分组子集对照】精准提数引擎...\n")

    # 1. 加载最新的基准数据
    # 如果你跑了之前的 API 补全脚本改了名字，请把这里改成对应的文件名
    file_path = 'Ultimate_Regression_Base.csv'
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"❌ 找不到文件 {file_path}，请检查路径！")
        return

    # 2. 核心操作：硬性切分实验组和对照组
    # 纯理工(Control) = 0, 跨界(Experimental) > 0
    df['Group'] = np.where(df['team_hss_dna_pct'] > 0, 'Experimental', 'Control')

    # 3. 定义你需要填入表 2 的变量
    variables = {
        'Log_RCR': 'Log_RCR',
        'Team_Size': 'total_authors',
        'Prior_Knowledge': 'Prior_Knowledge_Stock'
    }

    phases = ['Phase1', 'Phase2', 'Phase3']
    groups = ['Control', 'Experimental']

    # 辅助函数：安全计算统计量
    def get_stats(sub_df):
        if len(sub_df) == 0:
            return "0", "N/A", "N/A", "N/A", "N/A"
        obs = len(sub_df)
        mean_v = f"{sub_df.mean():.3f}"
        std_v = f"{sub_df.std():.3f}"
        min_v = f"{sub_df.min():.2f}"
        max_v = f"{sub_df.max():.2f}"
        return str(obs), mean_v, std_v, min_v, max_v

    # 4. 按 LaTeX 表格的结构顺序进行打印
    print("=" * 80)
    print("📋 可以直接抄进 LaTeX 的数据对照表")
    print("=" * 80)

    for var_name, col_name in variables.items():
        if col_name not in df.columns:
            print(f"⚠️ 找不到列 {col_name}，跳过...")
            continue

        print(f"\n--- 📍 正在统计变量: {var_name} ---")

        for group in groups:
            group_label = "Control: Pure STEM" if group == 'Control' else "Experimental: HSS Cross"
            print(f"  👉 【{group_label}】")

            for phase in phases:
                # 筛选当前组、当前阶段、并且该变量不为空的数据
                sub_df = df[(df['Group'] == group) & (df['stage'] == phase)][col_name].dropna()

                obs, mean_v, std_v, min_v, max_v = get_stats(sub_df)

                # 打印出可以直接填入 LaTeX xxx 位置的格式
                print(
                    f"     {phase:<8} | Obs: {obs:<6} | Mean: {mean_v:<6} | SD: {std_v:<6} | Min: {min_v:<5} | Max: {max_v:<6}")

    print("\n" + "=" * 80)
    print("✨ 统计完毕！请直接将上面的数字对应填入 LaTeX 代码的 xxx 中。")


# 运行统计！
calculate_grouped_statistics()