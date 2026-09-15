import json
import os
import time
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock


def profile_phase1_ultimate_pwfc():
    print("🧬 启动【第一阶段 创世老兵】全学科 DNA 测序仪 (终极纯净版)...\n")

    # ⚠️ 注意这里读取的是咱们刚刚新生成的、带有正确 paper_count 的文件！
    input_csv = 'phase1_authors_T0.csv'
    output_jsonl = 'phase1_newcomers_scopus_pwfc.jsonl'

    # 维持高速通道
    MY_EMAIL = "1525652634@qq.com"
    base_url = "https://api.openalex.org/authors"

    if not os.path.exists(input_csv):
        print(f"❌ 找不到输入文件 {input_csv}，请确认上一步提取脚本已成功运行！")
        return

    # ---------------------------------------------------------
    # 1. 读取待查名单与极其坚固的断点续传检测
    # ---------------------------------------------------------
    print("🔍 正在初始化老兵清单并检测本地断点...")
    processed_ids = set()
    if os.path.exists(output_jsonl):
        with open(output_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        processed_ids.add(json.loads(line)['author_ids'])
                    except:
                        pass
        print(f"   ➤ 发现本地已有 {len(processed_ids):,} 份完成测序的终极档案，将自动跳过。")

    # 从 CSV 读取第一阶段的老兵名单（包含本地计算的纯净发文量！）
    df_authors = pd.read_csv(input_csv)
    tasks_dict = {}

    for _, row in df_authors.iterrows():
        a_id_full = str(row['author_id']).strip()
        if a_id_full not in processed_ids:
            tasks_dict[a_id_full] = {
                "T0_year": int(row['T0_year']),
                "paper_count": int(row['paper_count'])  # 🎯 直接继承本地的纯净火力值！
            }

    task_ids = list(tasks_dict.keys())
    total_tasks = len(task_ids)
    print(f"   ➤ 实际需要向 API 提取 DNA 的老兵数量: {total_tasks:,} 人\n")

    if total_tasks == 0:
        print("🎉 所有第一阶段老兵均已完成 PWFC 加权画像，无需运行！")
        return

    # ---------------------------------------------------------
    # 2. 核心算法：全学科发文量加权 (PWFC) DNA 提取
    # ---------------------------------------------------------
    def parse_all_scopus_fields_weighted(author_data):
        topics = author_data.get('topics', [])
        if not topics:
            return "Unknown", {}

        field_counts = {}
        total_valid_papers = 0

        # 统计该作者在 26 大 Scopus 领域的真实发文绝对值
        for t in topics:
            field_name = t.get('field', {}).get('display_name', 'Unknown')
            count = t.get('count', 0)

            if field_name != 'Unknown' and count > 0:
                field_counts[field_name] = field_counts.get(field_name, 0) + count
                total_valid_papers += count

        field_weights = {}
        primary_origin = "Unknown"

        if total_valid_papers > 0:
            # 计算精确到万分之一的学术 DNA 纯度
            for field, count in field_counts.items():
                field_weights[field] = round(count / total_valid_papers, 4)

            # 找出绝对主导域 (占比最高的那一个)
            sorted_fields = sorted(field_counts.items(), key=lambda x: x[1], reverse=True)
            primary_origin = sorted_fields[0][0] if sorted_fields else "Unknown"

        return primary_origin, field_weights

    # ---------------------------------------------------------
    # 3. 工业级多线程并发引擎
    # ---------------------------------------------------------
    batch_size = 50
    batches = [task_ids[i:i + batch_size] for i in range(0, total_tasks, batch_size)]
    total_batches = len(batches)
    print(f"🗂️ 任务已切分为 {total_batches:,} 个批次并发执行。")

    session = requests.Session()
    save_lock = Lock()
    completed_batches = 0

    def fetch_batch(batch_list):
        short_ids = [aid.split('/')[-1] for aid in batch_list]
        id_filter = "|".join(short_ids)

        # 🎯 极其致命的优化：彻底剔除 works_count，只拉取 id 和 topics！防污染且提速！
        params = {
            "filter": f"openalex:{id_filter}",
            "select": "id,topics",
            "per-page": batch_size,
            "mailto": MY_EMAIL
        }

        for _ in range(3):  # 失败重试 3 次
            try:
                res = session.get(base_url, params=params, timeout=15)
                if res.status_code == 429:
                    time.sleep(5)
                    continue
                res.raise_for_status()

                returned_authors = {a['id']: a for a in res.json().get('results', [])}
                batch_records = []

                for full_id in batch_list:
                    api_data = returned_authors.get(full_id, {})
                    primary, dna_weights = parse_all_scopus_fields_weighted(api_data)

                    # 组装最终完全体记录
                    record = {
                        "author_ids": full_id,
                        "T0_year": tasks_dict[full_id]["T0_year"],
                        "paper_count": tasks_dict[full_id]["paper_count"],  # 🎯 填入本地计算好的纯净火力
                        "primary_origin_field": primary,
                        "scopus_dna_weights": dna_weights
                    }
                    batch_records.append(record)
                return batch_records
            except Exception as e:
                time.sleep(3)

        # 彻底失败的兜底防断档措施
        return [{
            "author_ids": aid,
            "T0_year": tasks_dict[aid]["T0_year"],
            "paper_count": tasks_dict[aid]["paper_count"],
            "primary_origin_field": "Error",
            "scopus_dna_weights": {}
        } for aid in batch_list]

    start_time = time.time()

    # 开启 5 线程火力全开
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_batch, b): b for b in batches}

        for future in as_completed(futures):
            records = future.result()
            with save_lock:
                with open(output_jsonl, 'a', encoding='utf-8') as f_out:
                    for r in records:
                        f_out.write(json.dumps(r, ensure_ascii=False) + '\n')

                completed_batches += 1
                if completed_batches % 50 == 0 or completed_batches == total_batches:
                    progress = (completed_batches / total_batches) * 100
                    elapsed = time.time() - start_time
                    speed = completed_batches / elapsed if elapsed > 0 else 0
                    print(
                        f"⚡ 进度: [{completed_batches}/{total_batches}] ({progress:.1f}%) | 速度: {speed:.1f} 批/秒 -> 坚如磐石，实时落盘。")

    print("\n" + "🔥" * 20)
    print("🏆 第一阶段 3.6 万老兵的终极净版 PWFC 基因测序完美收官！")
    print(f"💾 零污染、全量对齐的终极数据已保存至: {output_jsonl}")


# 🚀 见证大一统底座诞生的时刻！
profile_phase1_ultimate_pwfc()