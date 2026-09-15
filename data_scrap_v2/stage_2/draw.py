import json
import os
import time
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock


def profile_phase2_newcomers_pwfc():
    print("🧬 启动【第二阶段 32万新兵大军】全学科 DNA 测序仪...\n")

    input_csv = 'phase2_newcomers_to_fetch.csv'
    output_jsonl = 'phase2_newcomers_scopus_pwfc.jsonl'

    # ⚠️ 保持高速通道畅通
    MY_EMAIL = "1525652634@qq.com"
    base_url = "https://api.openalex.org/authors"

    if not os.path.exists(input_csv):
        print(f"❌ 找不到输入文件 {input_csv}！请确认分流脚本已跑完。")
        return

    # ---------------------------------------------------------
    # 1. 初始化 32 万大名单与断点续传
    # ---------------------------------------------------------
    processed_ids = set()
    if os.path.exists(output_jsonl):
        with open(output_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        processed_ids.add(json.loads(line)['author_ids'])
                    except:
                        pass
        print(f"   ➤ 发现本地已有 {len(processed_ids):,} 份断点档案，将自动跳过。")

    df_authors = pd.read_csv(input_csv)
    tasks_dict = {}
    for _, row in df_authors.iterrows():
        a_id_full = str(row['author_id']).strip()
        if a_id_full not in processed_ids:
            # 继承 CSV 里纯净的 T0 和 发文量
            tasks_dict[a_id_full] = {"T0_year": int(row['T0_year']), "paper_count": int(row['paper_count'])}

    task_ids = list(tasks_dict.keys())
    total_tasks = len(task_ids)
    print(f"   ➤ 实际需要向 API 提取 DNA 的新兵数量: {total_tasks:,} 人\n")

    if total_tasks == 0:
        print("🎉 所有 32 万新兵均已测序完毕！")
        return

    # ---------------------------------------------------------
    # 2. 核心算法：PWFC 学科基因提取
    # ---------------------------------------------------------
    def parse_all_scopus_fields_weighted(author_data):
        topics = author_data.get('topics', [])
        if not topics: return "Unknown", {}

        field_counts = {}
        total_valid_papers = 0
        for t in topics:
            field_name = t.get('field', {}).get('display_name', 'Unknown')
            count = t.get('count', 0)
            if field_name != 'Unknown' and count > 0:
                field_counts[field_name] = field_counts.get(field_name, 0) + count
                total_valid_papers += count

        field_weights = {}
        primary_origin = "Unknown"
        if total_valid_papers > 0:
            for field, count in field_counts.items():
                field_weights[field] = round(count / total_valid_papers, 4)
            sorted_fields = sorted(field_counts.items(), key=lambda x: x[1], reverse=True)
            primary_origin = sorted_fields[0][0] if sorted_fields else "Unknown"
        return primary_origin, field_weights

    # ---------------------------------------------------------
    # 3. 工业级高并发引擎
    # ---------------------------------------------------------
    batch_size = 50
    batches = [task_ids[i:i + batch_size] for i in range(0, total_tasks, batch_size)]
    total_batches = len(batches)
    print(f"🗂️ 任务已切分为 {total_batches:,} 个批次并发执行，预计耗时 20-30 分钟。")

    session = requests.Session()
    save_lock = Lock()
    completed_batches = 0

    def fetch_batch(batch_list):
        short_ids = [aid.split('/')[-1] for aid in batch_list]
        id_filter = "|".join(short_ids)
        # 🎯 极致优化：只拉 id 和 topics，不碰 works_count
        params = {"filter": f"openalex:{id_filter}", "select": "id,topics", "per-page": batch_size, "mailto": MY_EMAIL}

        for _ in range(3):
            try:
                res = session.get(base_url, params=params, timeout=20)
                if res.status_code == 429:
                    time.sleep(8)  # 遇到限流，强制冷却 8 秒
                    continue
                res.raise_for_status()

                returned_authors = {a['id']: a for a in res.json().get('results', [])}
                batch_records = []
                for full_id in batch_list:
                    api_data = returned_authors.get(full_id, {})
                    primary, dna_weights = parse_all_scopus_fields_weighted(api_data)
                    record = {
                        "author_ids": full_id,
                        "T0_year": tasks_dict[full_id]["T0_year"],
                        "paper_count": tasks_dict[full_id]["paper_count"],
                        "primary_origin_field": primary,
                        "scopus_dna_weights": dna_weights
                    }
                    batch_records.append(record)
                return batch_records
            except Exception as e:
                time.sleep(3)

        # 兜底防断档
        return [
            {"author_ids": aid, "T0_year": tasks_dict[aid]["T0_year"], "paper_count": tasks_dict[aid]["paper_count"],
             "primary_origin_field": "Error", "scopus_dna_weights": {}} for aid in batch_list]

    start_time = time.time()

    # 5 线程火力全开
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_batch, b): b for b in batches}
        for future in as_completed(futures):
            records = future.result()
            with save_lock:
                with open(output_jsonl, 'a', encoding='utf-8') as f_out:
                    for r in records: f_out.write(json.dumps(r, ensure_ascii=False) + '\n')

                completed_batches += 1
                # 每跑 50 批（约 2500 人）打印一次进度，减少控制台刷屏导致的卡顿
                if completed_batches % 50 == 0 or completed_batches == total_batches:
                    progress = (completed_batches / total_batches) * 100
                    elapsed = time.time() - start_time
                    speed = completed_batches / elapsed if elapsed > 0 else 0
                    print(f"⚡ 进度: [{completed_batches}/{total_batches}] ({progress:.1f}%) | 速度: {speed:.1f} 批/秒")

    print("\n" + "🔥" * 20)
    print("🏆 第二阶段 32 万新兵的 PWFC 基因测序史诗级收官！")
    print(f"💾 终极档案库已安全保存至: {output_jsonl}")


# 🚀 点火发射！
profile_phase2_newcomers_pwfc()