import pandas as pd
import numpy as np
import json
import os


def build_dataset_and_generate_table1():
    print("🚧 启动【表 1：描述性统计】数据组装与提数引擎...\n")

    # ==========================================
    # 步骤 1：加载基准表 (arXiv Master)
    # ==========================================
    print("📥 1. 加载 arXiv 主表...")
    dna_file = 'Ultimate_ArXiv_DNA_Master.csv'
    df = pd.read_csv(dna_file)
    # 清理 work_id，确保后续匹配无误
    df['clean_work_id'] = df['work_id'].astype(str).str.replace('https://openalex.org/', '').str.strip()

    # ==========================================
    # 步骤 2：加载引用数据计算 Log_RCR
    # ==========================================
    print("📥 2. 匹配被引数据并计算 Log_RCR...")
    citation_file = 'E:\PythonProject\GenAI\data_process\process\citation_mapping.json'  # 用你之前的路径
    if os.path.exists(citation_file):
        with open(citation_file, 'r', encoding='utf-8') as f:
            cite_map = json.load(f)

        df['citations'] = df['work_id'].astype(str).map(lambda x: cite_map.get(x, {}).get('citations', 0))
        df['year'] = df['work_id'].astype(str).map(lambda x: cite_map.get(x, {}).get('year', 2023))

        # 计算 RCR 和 log_RCR (复用你之前写好的完美逻辑)
        yearly_avg = df.groupby('year')['citations'].mean().reset_index()
        yearly_avg.rename(columns={'citations': 'yearly_mean'}, inplace=True)
        df = pd.merge(df, yearly_avg, on='year', how='left')
        df['RCR'] = df['citations'] / df['yearly_mean'].replace(0, 1)
        df['Log_RCR'] = np.log1p(df['RCR'])
    else:
        print("⚠️ 未找到引用文件，跳过 Log_RCR 计算...")
        df['Log_RCR'] = np.nan

    # ==========================================
    # 步骤 3：加载 JSONL 获取参考文献和作者列表
    # ==========================================
    print("📥 3. 遍历 JSONL 提取 Reference_Count 和 Author_IDs...")
    ref_count_dict = {}
    work_authors_dict = {}
    jsonl_files = [
        'E:\PythonProject\GenAI\data_scrap_v2\stage_1\phase1_hybrid_cleaned.jsonl',
        'E:\PythonProject\GenAI\data_scrap_v2\stage_2\phase2_hybrid_cleaned.jsonl',
        'E:\PythonProject\GenAI\data_scrap_v2\stage_3\phase3_hybrid_cleaned.jsonl']

    for j_file in jsonl_files:
        if os.path.exists(j_file):
            print(f"   -> 正在强力解析: {j_file}")
            with open(j_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # 跳过损坏的 JSON 行

                    # 1. 安全提取并清洗 work_id (暴力砍掉网址，只留 W 开头的纯 ID)
                    w_id_raw = record.get('work_id')
                    if not w_id_raw: continue  # 如果连文章 ID 都没有，直接丢弃
                    w_id = str(w_id_raw).split('/')[-1].strip()

                    # 2. 安全提取参考文献数量
                    ref_list = record.get('referenced_works')
                    # 如果 ref_list 是 None，判定为 0 篇引用
                    ref_count_dict[w_id] = len(ref_list) if isinstance(ref_list, list) else 0

                    # 3. 终极防御式提取 author_ids (彻底解决 NoneType 报错)
                    authors_raw = record.get('author_ids')
                    clean_authors = []
                    if isinstance(authors_raw, list):
                        for a in authors_raw:
                            if a:  # 过滤掉潜伏的 null / None
                                # 暴力砍掉网址，只留 A 开头的纯 ID
                                clean_a = str(a).split('/')[-1].strip()
                                clean_authors.append(clean_a)

                    work_authors_dict[w_id] = clean_authors
        else:
            print(f"   ⚠️ 找不到文件: {j_file}，请检查路径。")

    df['Reference_Count'] = df['clean_work_id'].map(ref_count_dict).fillna(0)

    # ==========================================
    # 步骤 4：计算团队历史资历 (Prior_Knowledge_Stock)
    # ==========================================
    print("📥 4. 查阅作者画像表，计算 Prior_Knowledge_Stock...")
    author_file = 'E:\PythonProject\GenAI\data_process\process\MASTER_AUTHOR_PROFILES_12YEARS.csv'
    if os.path.exists(author_file):
        # 真实表头已确认！直接正常读取
        df_authors = pd.read_csv(author_file)

        # 暴力剥离 URL，只取 A 开头的纯 ID（对应你的 author_id 列）
        df_authors['clean_author_id'] = df_authors['author_id'].astype(str).apply(
            lambda x: str(x).split('/')[-1].strip())

        # 强制把 total_papers 转成数字（防止有的地方读成了字符串）
        df_authors['total_papers'] = pd.to_numeric(df_authors['total_papers'], errors='coerce').fillna(0)

        # 建立 author_id 到 total_papers 的快速映射字典
        author_paper_map = dict(zip(df_authors['clean_author_id'], df_authors['total_papers']))

        def calc_prior_knowledge(w_id):
            authors = work_authors_dict.get(w_id, [])
            # 把这篇论文所有作者的 total_papers 加起来
            total = sum(author_paper_map.get(a_id, 0) for a_id in authors)
            return total

        df['Prior_Knowledge_Stock'] = df['clean_work_id'].apply(calc_prior_knowledge)
        print(f"   🎯 成功提取了 {len(author_paper_map)} 位作者的发文资历！")
    else:
        print("⚠️ 未找到作者画像表，跳过资历计算...")
        df['Prior_Knowledge_Stock'] = np.nan

    # ==========================================
    # 步骤 5：终极输出 —— 打印顶刊描述性统计表
    # ==========================================
    print("\n" + "=" * 80)
    print("🏆 顶刊格式描述性统计表 (Table 1 - Phase Analysis)")
    print("=" * 80)

    # 格式化辅助函数：Mean (Std)
    def fmt(series):
        s = series.dropna()
        if len(s) == 0: return "N/A"
        return f"{s.mean():.3f} ({s.std():.3f})"

    # 定义要统计的变量和对应的列名
    stats_vars = {
        'Log_RCR (被引影响力)': 'Log_RCR',
        'HSS_Concentration (文科浓度 %)': 'team_hss_dna_pct',
        'Team_Size (团队人数)': 'total_authors',
        'Reference_Count (参考文献数)': 'Reference_Count',
        'Prior_Knowledge (历史总发文量)': 'Prior_Knowledge_Stock'
    }

    # 按阶段分组
    phases = sorted(df['stage'].dropna().unique())

    # 构造表头
    header = f"| {'Variables':<30} | {'(2) arXiv Overall':<20} |"
    for p in phases: header += f" {p:<16} |"
    print(header)
    print("-" * len(header))

    # 输出样本量
    obs_row = f"| {'Obs (样本量)':<30} | {len(df):<20} |"
    for p in phases: obs_row += f" {len(df[df['stage'] == p]):<16} |"
    print(obs_row)

    # 输出各变量统计
    for label, col in stats_vars.items():
        if col not in df.columns: continue
        row_str = f"| {label:<30} | {fmt(df[col]):<20} |"
        for p in phases:
            row_str += f" {fmt(df[df['stage'] == p][col]):<16} |"
        print(row_str)

    print("=" * 80)

    # 保存这份包含了所有完整变量的终极回归基表！
    df.to_csv('Ultimate_Regression_Base.csv', index=False)
    print("\n💾 拼接完整的新数据集已保存为: Ultimate_Regression_Base.csv (可以直接拿去跑回归了！)")


# 引爆统计！
build_dataset_and_generate_table1()