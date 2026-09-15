import json
import os


def split_workload():
    print("🚀 启动【分布式任务切分器】...\n")

    CITATION_FILE = 'citation_mapping.json'
    OUTPUT_FILE = 'citing_disciplines_master.jsonl'

    # 1. 读取基础全量池
    if not os.path.exists(CITATION_FILE):
        print("❌ 找不到 citation_mapping.json！")
        return

    with open(CITATION_FILE, 'r', encoding='utf-8') as f:
        citation_mapping = json.load(f)

    # 2. 读取已完成的 IDs（完美继承你刚才跑出的数据）
    processed_ids = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    processed_ids.add(json.loads(line)['work_id'])
                except:
                    pass
    print(f"✅ 检测到本地已完成: {len(processed_ids)} 篇。")

    # 3. 筛选出还未抓取的数据
    todo_dict = {}
    for wid, info in citation_mapping.items():
        if info.get('citations', 0) > 0 and wid not in processed_ids:
            todo_dict[wid] = info

    todo_items = list(todo_dict.items())
    total_todo = len(todo_items)
    print(f"⏳ 剔除已完成部分，剩余任务总数: {total_todo} 篇。\n")

    if total_todo == 0:
        print("🎉 已经全部抓完了，不需要切分！")
        return

    # 4. 拦腰斩断，一分为二
    mid_point = total_todo // 2
    part1_dict = dict(todo_items[:mid_point])
    part2_dict = dict(todo_items[mid_point:])

    # 5. 生成两份独立的长名单
    with open('todo_mapping_part1.json', 'w', encoding='utf-8') as f:
        json.dump(part1_dict, f, ensure_ascii=False)

    with open('todo_mapping_part2.json', 'w', encoding='utf-8') as f:
        json.dump(part2_dict, f, ensure_ascii=False)

    print("✂️ 切分大功告成！")
    print(f"🖥️ 设备 A 任务清单: {len(part1_dict)} 篇 (保存为 todo_mapping_part1.json)")
    print(f"💻 设备 B 任务清单: {len(part2_dict)} 篇 (保存为 todo_mapping_part2.json)")
    print("\n👉 下一步：把 part2 拷给第二台电脑，两台电脑同时启动多线程代码！")


if __name__ == "__main__":
    split_workload()