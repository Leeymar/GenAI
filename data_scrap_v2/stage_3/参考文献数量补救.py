import json
import requests
import time
import os


def rescue_reference_counts_only(input_file, output_file, api_key):
    # --- 1. 预扫描：统计为空的数据量 ---
    print(f"🔍 正在扫描原始文件: {input_file}...")
    empty_work_ids = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            # 仅锁定 referenced_works 为空列表的数据
            if isinstance(item.get('referenced_works'), list) and len(item['referenced_works']) == 0:
                w_id = item['work_id'].split('/')[-1]
                empty_work_ids.append(w_id)

    total_empty = len(empty_work_ids)
    print(f"\n📢 扫描完成！")
    print(f"📊 发现 'referenced_works' 为空的数据共计: {total_empty} 条")
    print("-" * 50)

    if total_empty == 0:
        print("✅ 没有发现为空的数据，无需操作。")
        return

    # --- 2. 批量查询并实时保存 ---
    print(f"🚀 开始通过 API 补全数量 (每 50 条自动保存一次)...")

    batch_size = 50
    rescued_total = 0

    # 使用追加模式写入，确保“及时保存”
    with open(output_file, 'a', encoding='utf-8') as f_out:
        for i in range(0, total_empty, batch_size):
            batch_ids = empty_work_ids[i: i + batch_size]
            ids_filter = "|".join(batch_ids)

            # 仅请求 id 和引文总数，保证最高速度
            url = f"https://api.openalex.org/works?filter=openalex:{ids_filter}&select=id,referenced_works_count&api_key={api_key}"

            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    results = response.json().get('results', [])

                    # 批量写入这一组结果
                    for res in results:
                        output_item = {
                            "work_id": res['id'],  # 保持原始格式 https://openalex.org/W...
                            "referenced_works_count": res.get('referenced_works_count', 0)
                        }
                        f_out.write(json.dumps(output_item, ensure_ascii=False) + '\n')
                        rescued_total += 1

                    # 刷新磁盘缓冲区，确保数据及时写入硬盘
                    f_out.flush()
                    os.fsync(f_out.fileno())
                else:
                    print(f"⚠️ 批次 {i // batch_size + 1} 请求失败 (Status: {response.status_code})")

                # 实时进度反馈
                if (i + batch_size) % 500 == 0 or (i + batch_size) >= total_empty:
                    print(f"⏳ 进度: {min(i + batch_size, total_empty)} / {total_empty} | 已找回: {rescued_total} 条")

                # 有 API Key 可以极快，这里设置微小延时防止网络波动
                time.sleep(0.05)

            except Exception as e:
                print(f"❌ 批次 {i // batch_size + 1} 发生致命错误: {e}")
                continue

    print("-" * 50)
    print(f"✨ 任务结束！")
    print(f"💾 补全数据已存入: {output_file}")
    print(f"✅ 最终成功补全数量: {rescued_total} 条")


# --- 执行 ---
API_KEY = "3VymPsRtP5VnRD1hZDR1OQ"
INPUT_JSONL = 'phase3_hybrid_cleaned.jsonl'
OUTPUT_JSONL = 'phase3_rescued_ref_counts.jsonl'

rescue_reference_counts_only(INPUT_JSONL, OUTPUT_JSONL, API_KEY)