import pandas as pd
import os


def clean_data_and_extract_authors(jsonl_file_path):
    print(f"📊 正在加载 Phase 1 原始双轨制数据：{jsonl_file_path}")

    if not os.path.exists(jsonl_file_path):
        raise FileNotFoundError(f"❌ 找不到文件: {jsonl_file_path}")

    # 1. 读取 JSONL
    df = pd.read_json(jsonl_file_path, lines=True)
    initial_count = len(df)
    print(f"📥 原始文献总数：{initial_count} 篇")

    # ==========================================
    # 2. 核心清洗：剔除 author_ids 为空的“幽灵文献”
    # df['author_ids'].map(len) 会计算每行列表中元素的个数
    # 我们只保留那些作者数量 > 0 的行
    # ==========================================
    df_clean = df[df['author_ids'].map(len) > 0].copy()

    cleaned_count = len(df_clean)
    ghost_count = initial_count - cleaned_count
    print(f"🧹 清洗完成！发现并剔除了 {ghost_count} 篇无作者的“幽灵文献”。")
    print(f"✨ 最终绝对有效文献数：{cleaned_count} 篇")

    # (可选) 将清洗后的干净数据重新覆盖保存，方便以后复用
    df_clean.to_json("phase1_hybrid_cleaned.jsonl", orient='records', lines=True, force_ascii=False)

    # ---------------------------------------------------------
    # 3. 顺势爆发：直接把清洗后的数据炸开 (explode)，算作者 T0！
    # ---------------------------------------------------------
    print("\n🚀 正在从干净的文献库中提取跨界大佬名单...")
    df_exploded = df_clean.explode('author_ids')
    df_exploded = df_exploded.dropna(subset=['author_ids'])  # 二次保险，防 NaN

    # 按照作者 ID 分组，计算他们在 14-16 年的发文量和最小年份
    author_stats = df_exploded.groupby('author_ids').agg(
        T0_year=('publication_year', 'min'),
        paper_count=('work_id', 'count')
    ).reset_index()

    # 按照发文量降序排列
    author_stats = author_stats.sort_values(by=['paper_count', 'T0_year'], ascending=[False, True])

    output_csv = 'phase1_authors_T0_final.csv'
    author_stats.to_csv(output_csv, index=False)

    print("-" * 50)
    print(f"🎉 大功告成！共提取出 {len(author_stats)} 位有名有姓的真实学者。")
    print(f"💾 作者名单已保存至：{output_csv}")
    print("🏆 前 5 名高产祖师爷预览：")
    print(author_stats.head(5).to_string(index=False))
    print("-" * 50)


# 执行！请确保文件名跟你本地下载的文件名一致
clean_data_and_extract_authors("phase1_hybrid_final.jsonl")