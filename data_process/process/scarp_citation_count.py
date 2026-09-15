import json
import os
import requests
import time

# ==========================================
# 填入你的邮箱加入 OpenAlex "礼貌池" (提速防封)
EMAIL = "1525652634@qq.com"
# ==========================================

INPUT_FILES = [
    'E:\PythonProject\GenAI\data_scrap_v2\stage_1\phase1_hybrid_cleaned.jsonl',
    'E:\PythonProject\GenAI\data_scrap_v2\stage_2\phase2_hybrid_cleaned.jsonl',
    'E:\PythonProject\GenAI\data_scrap_v2\stage_3\phase3_hybrid_cleaned.jsonl'
]
OUTPUT_FILE = 'citation_mapping.json'


def load_existing_progress():
    """读取已经抓取过的进度，避免重复抓取"""
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except:
                return {}
    return {}


def save_progress(data):
    """实时保存数据到硬盘"""
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_citations_in_batches(ids, existing_data, batch_size=50):
    headers = {'User-Agent': f'mailto:{EMAIL}'}
    base_url = "https://api.openalex.org/works"

    # 🎯 核心逻辑：过滤掉已经存在于 existing_data 中的 ID
    remaining_ids = [wid for wid in ids if wid not in existing_data]
    total_remaining = len(remaining_ids)

    if total_remaining == 0:
        print("🎉 所有 ID 均已抓取完毕，无需继续！")
        return existing_data

    total_batches = (total_remaining // batch_size) + 1
    print(f"🚀 本地已存 {len(existing_data)} 篇。还剩 {total_remaining} 篇需抓取...")
    print(f"📡 预计发送 {total_batches} 个批次请求...")

    # 因为我们传进来的是 remaining_ids，所以从 0 开始遍历它
    for i in range(0, total_remaining, batch_size):
        batch_ids = remaining_ids[i:i + batch_size]
        id_filter = "|".join(batch_ids)

        params = {
            'filter': f'openalex:{id_filter}',
            'select': 'id,publication_year,cited_by_count',
            'per-page': batch_size
        }

        current_batch_num = i // batch_size + 1

        try:
            response = requests.get(base_url, params=params, headers=headers, timeout=15)
            if response.status_code == 200:
                data = response.json().get('results', [])
                for item in data:
                    work_id = item.get('id', '').split('/')[-1]
                    # 更新到大字典里
                    existing_data[work_id] = {
                        'year': item.get('publication_year'),
                        'citations': item.get('cited_by_count', 0)
                    }
                print(f"✅ 批次 {current_batch_num}/{total_batches} 成功！")

                # 🛡️ 【保命机制】：每抓完 100 个批次，强制存盘一次！
                if current_batch_num % 100 == 0:
                    save_progress(existing_data)
                    print(f"💾 触发自动存档：进度已安全保存至硬盘 (当前共 {len(existing_data)} 篇)")

            else:
                print(f"⚠️ 批次 {current_batch_num} 抓取失败，状态码: {response.status_code}")
                time.sleep(2)
        except Exception as e:
            print(f"❌ 批次 {current_batch_num} 发生网络错误: {e}")
            time.sleep(2)

        # 遵守礼貌原则，每批请求后停顿 0.1 秒
        time.sleep(0.1)

    # 全部跑完后，做最后一次完整存盘
    save_progress(existing_data)
    return existing_data


def main():
    print("📂 正在从本地文件中提取论文 ID...")
    all_work_ids = set()

    for path in INPUT_FILES:
        if not os.path.exists(path):
            print(f"忽略文件 {path} (未找到)")
            continue

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    work = json.loads(line)
                    # 提取 work_id (之前修复的地方)
                    raw_id = work.get('work_id')
                    if raw_id:
                        clean_id = str(raw_id).split('/')[-1]
                        all_work_ids.add(clean_id)
                except:
                    pass

    ids_list = list(all_work_ids)
    print(f"🎯 本地文件共解析出 {len(ids_list)} 篇独立论文。")

    if len(ids_list) == 0:
        print("❌ 提取失败！请检查文件是否存在以及字段名是否正确。")
        return

    # 1. 先读取硬盘上可能存在的历史进度
    existing_data = load_existing_progress()

    # 2. 把目标 ID 和历史进度传给抓取函数
    final_data = get_citations_in_batches(ids_list, existing_data)

    print(f"\n🎉 完美收工！共集齐 {len(final_data)} 篇论文的被引量数据。")
    print(f"💾 数据均已安全保存在: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()