import json
import os


def analyze_phase2_basic_stats(jsonl_file):
    print(f"📊 启动第二阶段数据盘点：分析 {jsonl_file} ...\n")

    if not os.path.exists(jsonl_file):
        print(f"❌ 找不到文件：{jsonl_file}，请确认文件是否在当前目录下！")
        return

    valid_papers_count = 0
    empty_authors_count = 0
    unique_authors = set()

    # 逐行流式读取，完美应对超大文件
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                authors = data.get('author_ids', [])

                # 🛑 核心清洗逻辑：如果作者列表为空，直接丢弃！
                if not authors:
                    empty_authors_count += 1
                    continue

                # ✅ 统计有效论文
                valid_papers_count += 1

                # 👥 收集独立作者（使用 Set 自动去重）
                for author_id in authors:
                    unique_authors.add(author_id)

            except Exception as e:
                continue

    print("================================================================")
    print(" 📈 第二阶段 (2017-2021) 核心数据集战报")
    print("================================================================")
    print(f"🗑️ 数据清洗：已丢弃 (无作者) 论文: {empty_authors_count:>8,.0f} 篇")
    print(f"📄 最终有效主论文总数 (Papers):    {valid_papers_count:>8,.0f} 篇")
    print(f"👥 独立作者总数 (Unique Authors):  {len(unique_authors):>8,.0f} 人")
    print("================================================================\n")
    print("💡 接下来，这十几万名作者就是我们下一阶段“原生学科溯源”的宝藏库！")


# 请确保传入的是你刚刚跑出来的终极合并文件
analyze_phase2_basic_stats('phase2_hybrid_cleaned.jsonl')