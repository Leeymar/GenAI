import os


def count_jsonl_records():
    # 定义你要统计的三个阶段的文件名
    files = {
        "Phase 1 (2014-2016)": r'E:\PythonProject\GenAI\data_scrap_v2\stage_1\phase1_hybrid_cleaned.jsonl',
        "Phase 2 (2017-2021)": r'E:\PythonProject\GenAI\data_scrap_v2\stage_2\phase2_hybrid_cleaned.jsonl',
        "Phase 3 (2022-2026)": r'E:\PythonProject\GenAI\data_scrap_v2\stage_3\phase3_hybrid_cleaned.jsonl'
    }

    total_records = 0

    print("📊 启动极速统计引擎 (低内存消耗模式)...\n")
    print("-" * 50)

    for phase_name, filename in files.items():
        if os.path.exists(filename):
            # 使用生成器表达式逐行读取，不占用额外内存，统计速度极快
            with open(filename, 'r', encoding='utf-8') as f:
                count = sum(1 for _ in f)

            total_records += count
            # 格式化输出，带上千位分隔符，方便你直接抄进论文
            print(f"✅ {phase_name} [{filename}]: {count:,} 篇")
        else:
            print(f"❌ 找不到文件: {filename} (请检查文件名或路径)")

    print("-" * 50)
    print(f"🏆 终极合并总数 (Total N): {total_records:,} 篇")
    print("-" * 50)

    print("\n💡 提示：运行完毕后，你可以直接将控制台里这些数字填入刚才论文里的 [N1], [N2], [N3] 和 [Total N] 中。")


if __name__ == "__main__":
    count_jsonl_records()