import json
import os


def clean_empty_disciplines():
    print("🧹 启动【空数据清洗手术刀】...\n")

    # 你的原始数据文件（如果你的名字不一样，请在这里修改）
    INPUT_FILE = 'citing_disciplines_master.jsonl'
    # 清洗后生成的新文件
    OUTPUT_FILE = 'citing_disciplines_master.jsonl'

    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到文件: {INPUT_FILE}")
        return

    removed_count = 0
    kept_count = 0

    # 读取原文件，将非空数据写入新文件
    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, \
            open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:

        for line in f_in:
            if not line.strip(): continue

            try:
                record = json.loads(line)
                # 核心判断：如果 citing_disciplines 是空字典 {}
                if not record.get("citing_disciplines"):
                    removed_count += 1
                else:
                    # 如果有数据，原封不动地写入新文件
                    f_out.write(line)
                    kept_count += 1
            except Exception as e:
                print(f"⚠️ 解析错误，跳过该行: {e}")

    print("✨ 清洗完成！")
    print(f"✅ 保留了有效数据: {kept_count} 篇")
    print(f"🗑️ 剔除了空缺数据: {removed_count} 篇")
    print(f"\n👉 下一步：请把原文件备份或删除，然后把 【{OUTPUT_FILE}】 重命名为 【{INPUT_FILE}】，即可重新运行主程序！")


if __name__ == "__main__":
    clean_empty_disciplines()