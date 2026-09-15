import json
import os
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock


def hunt_and_fix_errors():
    print("🛠️ 启动【Error 猎杀与修复引擎】...\n")

    # ⚠️ 目标文件：你第三阶段剔除老兵后的纯净文件
    target_file = 'phase3_newcomers_scopus_pwfc.jsonl'
    fixed_output_file = 'phase3_newcomers_scopus_pwfc_FIXED.jsonl'

    MY_EMAIL = "1525652634@qq.com"
    base_url = "https://api.openalex.org/authors"

    if not os.path.exists(target_file):
        print(f"❌ 找不到文件 {target_file}，请检查文件名！")
        return

    # ---------------------------------------------------------
    # 1. 扫描文件，分离健康数据与 Error 数据
    # ---------------------------------------------------------
    print(f"🔍 正在扫描 {target_file} 寻找网络 Error 残骸...")

    healthy_records = []
    error_tasks = {}

    with open(target_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                # 检查是否是 Error 或者是 Unknown (有时 API 返回空也会标 Unknown)
                if data.get('primary_origin_field') == 'Error':
                    a_id = data['author_ids']
                    error_tasks[a_id] = {
                        "T0_year": data["T0_year"],
                        "paper_count": data["paper_count"]
                    }
                else:
                    healthy_records.append(data)
            except:
                pass

    total_errors = len(error_tasks)
    print(f"   ➤ 扫描完毕！健康数据: {len(healthy_records):,} 条")
    print(f"   ➤ 抓获 Error 残骸: {total_errors:,} 条\n")

    if total_errors == 0:
        print("🎉 你的文件极其完美，没有任何 Error，不需要修复！")
        return

    # ---------------------------------------------------------
    # 2. 核心算法：PWFC 学科基因提取 (复用)
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
    # 3. 针对 Error 发起复活打击
    # ---------------------------------------------------------
    task_ids = list(error_tasks.keys())
    batch_size = 50
    batches = [task_ids[i:i + batch_size] for i in range(0, total_errors, batch_size)]
    total_batches = len(batches)
    print(f"🚑 启动 API 抢救机制，共需发起 {total_batches} 批次请求...")

    session = requests.Session()
    save_lock = Lock()
    completed_batches = 0
    fixed_records = []

    def fetch_batch(batch_list):
        short_ids = [aid.split('/')[-1] for aid in batch_list]
        id_filter = "|".join(short_ids)
        params = {"filter": f"openalex:{id_filter}", "select": "id,topics", "per-page": batch_size, "mailto": MY_EMAIL}

        # 给这些顽固的 Error 增加重试次数到 5 次，并且增加超时时间到 30 秒！
        for _ in range(5):
            try:
                res = session.get(base_url, params=params, timeout=30)
                if res.status_code == 429:
                    time.sleep(10)  # 遇阻强制深呼吸
                    continue
                res.raise_for_status()

                returned_authors = {a['id']: a for a in res.json().get('results', [])}
                batch_results = []
                for full_id in batch_list:
                    api_data = returned_authors.get(full_id, {})
                    primary, dna_weights = parse_all_scopus_fields_weighted(api_data)
                    record = {
                        "author_ids": full_id,
                        "T0_year": error_tasks[full_id]["T0_year"],
                        "paper_count": error_tasks[full_id]["paper_count"],
                        "primary_origin_field": primary,
                        "scopus_dna_weights": dna_weights
                    }
                    batch_results.append(record)
                return batch_results
            except Exception as e:
                time.sleep(5)

        # 彻底死透了（比如 OpenAlex 库里这个人真的被删了）
        return [
            {"author_ids": aid, "T0_year": error_tasks[aid]["T0_year"], "paper_count": error_tasks[aid]["paper_count"],
             "primary_origin_field": "Ghost", "scopus_dna_weights": {}} for aid in batch_list]

    # 多线程火力全开
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_batch, b): b for b in batches}
        for future in as_completed(futures):
            records = future.result()
            with save_lock:
                fixed_records.extend(records)
                completed_batches += 1
                progress = (completed_batches / total_batches) * 100
                print(f"⚡ 抢救进度: [{completed_batches}/{total_batches}] ({progress:.1f}%)")

    # ---------------------------------------------------------
    # 4. 缝合身体，生成终极纯净版
    # ---------------------------------------------------------
    print("\n🧬 正在缝合健康数据与抢救回来的数据...")
    all_final_records = healthy_records + fixed_records

    with open(fixed_output_file, 'w', encoding='utf-8') as f_out:
        for r in all_final_records:
            f_out.write(json.dumps(r, ensure_ascii=False) + '\n')

    print("================================================================")
    print(" 🏆 Error 清除手术完美成功！")
    print("================================================================")
    print(f"✅ 最终完美档案数量: {len(all_final_records):,} 条")
    print(f"💾 已保存为极其纯净的: {fixed_output_file}")
    print("💡 提示：在跑大一统 `MASTER` 融合脚本时，记得把文件名更新成这个 `_FIXED` 的版本！")


# 启动修复！
hunt_and_fix_errors()