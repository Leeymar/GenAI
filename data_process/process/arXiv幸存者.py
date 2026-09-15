import pandas as pd
import json


def build_ultimate_arxiv_did_panel():
    print("🚀 启动【arXiv 完美 Log-RCR 桥接与融合】引擎...\n")

    # ==========================================
    # 1. 读取 OLS 基准表，提取完美的 Log-RCR
    # ==========================================
    ultimate_csv = r'E:\PythonProject\GenAI\data_process\test\Ultimate_Regression_Base.csv'
    print(f"📖 1. 正在加载权威 OLS 底表: {ultimate_csv.split(chr(92))[-1]} ...")

    try:
        df_ultimate = pd.read_csv(ultimate_csv)
    except FileNotFoundError:
        print("❌ 找不到 Ultimate_Regression_Base.csv，请确认路径！")
        return

    # 构建极速查询字典: {clean_work_id: {'Log_RCR': 2.126, 'year': 2014}}
    # 这样我们在遍历 JSONL 时可以用 O(1) 的速度秒查
    rcr_dict = {}
    for _, row in df_ultimate.iterrows():
        wid = str(row['clean_work_id']).strip()
        rcr_dict[wid] = {
            'Log_RCR': row['Log_RCR'],
            'year': row['year']
        }
    print(f"✅ 成功提取了 {len(rcr_dict):,} 篇 arXiv 论文的权威影响力数据！")

    # ==========================================
    # 2. 读取 DID 双胞胎名单 (我们要追踪的目标)
    # ==========================================
    print("👥 2. 正在加载 DID 双胞胎白名单...")
    try:
        df_panel = pd.read_csv('PSM_DID_Final_Panel.csv')
    except FileNotFoundError:
        print("❌ 找不到 PSM_DID_Final_Panel.csv，请先确认文件存在！")
        return

    target_authors = set(df_panel['author_id'].unique())
    print(f"   -> 锁定追踪目标: {len(target_authors):,} 位双胞胎作者。")

    # ==========================================
    # 3. 遍历 JSONL，顺藤摸瓜寻找作者
    # ==========================================
    print("🔗 3. 正在穿越 JSONL 连结 作者 与 论文 (桥接开始)...")
    jsonl_files = [
        r'E:\PythonProject\GenAI\data_scrap_v2\stage_1\phase1_hybrid_cleaned.jsonl',
        r'E:\PythonProject\GenAI\data_scrap_v2\stage_2\phase2_hybrid_cleaned.jsonl',
        r'E:\PythonProject\GenAI\data_scrap_v2\stage_3\phase3_hybrid_cleaned.jsonl'
    ]

    author_year_rcr_records = []

    for file_path in jsonl_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    paper = json.loads(line)

                    raw_wid = paper.get('work_id')
                    authors = paper.get('author_ids', [])

                    if not raw_wid or not authors: continue

                    # 洗干净 work_id
                    clean_wid = raw_wid.split('/')[-1]

                    # 💥 核心判定：这篇论文在不在我们的 arXiv 权威字典里？
                    if clean_wid in rcr_dict:
                        paper_log_rcr = rcr_dict[clean_wid]['Log_RCR']
                        paper_year = rcr_dict[clean_wid]['year']

                        # 把这份荣誉 (Log_RCR) 分发给这篇论文的每一位双胞胎作者
                        for aid in authors:
                            if aid in target_authors:
                                author_year_rcr_records.append({
                                    'author_id': aid,
                                    'year': paper_year,
                                    'Log_RCR': paper_log_rcr
                                })
        except FileNotFoundError:
            print(f"❌ 找不到文件: {file_path}")

    # ==========================================
    # 4. 聚合与最终拼接
    # ==========================================
    print("👨‍🔬 4. 正在计算作者在 arXiv 上的年度平均影响力...")
    df_records = pd.DataFrame(author_year_rcr_records)

    if len(df_records) == 0:
        print("⚠️ 警告：没有匹配到任何数据！请检查 ID 格式。")
        return

    # 按作者和年份聚合，算出 Mean Log_RCR
    author_year_rcr = df_records.groupby(['author_id', 'year'])['Log_RCR'].mean().reset_index()
    author_year_rcr.rename(columns={'Log_RCR': 'mean_log_rcr'}, inplace=True)

    print("✨ 5. 正在与 DID 双胞胎面板进行终极拼接 (Inner Join 保证血统纯正)...")
    # 我们用 inner join，只保留那些真正在 arXiv 上发过文章的作者-年份记录！
    df_final = df_panel.merge(author_year_rcr, on=['author_id', 'year'], how='inner')

    output_file = 'arXiv_PSM_DID_Panel_with_Impact.csv'
    df_final.to_csv(output_file, index=False)

    treated_count = df_final[df_final['is_treated'] == 1]['author_id'].nunique()
    control_count = df_final[df_final['is_treated'] == 0]['author_id'].nunique()

    print("\n=========================================================")
    print(f"🎉 史诗级胜利！arXiv 专属 DID 黄金面板已生成！")
    print(f"💾 数据已保存至: {output_file}")
    print("-" * 50)
    print("【arXiv 硬核生存大挑战结果】")
    print(f"🎓 跨界理科生 (Treated): 在 arXiv 上幸存 {treated_count:,} 人")
    print(f"🎓 纯血双胞胎 (Control): 在 arXiv 上幸存 {control_count:,} 人")
    print(f"总计保留 Author-Year 观测行数: {len(df_final):,} 行")
    print("=========================================================")


build_ultimate_arxiv_did_panel()