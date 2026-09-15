import pandas as pd
import requests
import time
import os
import json
import threading
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# ⚙️ 核心配置区
# ==========================================
YOUR_EMAIL = "2212408015@stmail.ujs.edu.cn"
OPENALEX_API_KEY = "3VymPsRtP5VnRD1hZDR1OQ"

MAX_WORKERS = 15  # 批量模式下，15 个并发已经快到飞起了
BATCH_SIZE = 50  # 每次请求携带 50 个 ID (15 * 50 = 一次并发处理 750 篇！)

write_lock = threading.Lock()


def create_robust_session():
    """复用你的神级底层网络架构"""
    session = requests.Session()
    adapter = HTTPAdapter(
        pool_connections=MAX_WORKERS,
        pool_maxsize=MAX_WORKERS,
        max_retries=Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    )
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


def fetch_batch_venues(batch_ids, current_phase, session):
    """批量查询并剥离 arXiv"""
    id_string = "|".join(batch_ids)
    api_url = f"https://api.openalex.org/works?filter=openalex:{id_string}&select=id,primary_location,locations&per-page={BATCH_SIZE}&mailto={YOUR_EMAIL}&api_key={OPENALEX_API_KEY}"

    batch_records = []
    rescued_count = 0

    try:
        response = session.get(api_url, timeout=20)
        response.raise_for_status()
        results = response.json().get("results", [])

        for item in results:
            clean_id = str(item.get("id", "")).split('/')[-1]
            final_venue = "Unknown Venue"
            is_arxiv = False

            primary_loc = item.get("primary_location")
            if primary_loc and isinstance(primary_loc, dict):
                source = primary_loc.get("source")
                if source and isinstance(source, dict):
                    final_venue = source.get("display_name", "Unknown Venue")

            # 🕵️‍♂️ arXiv 伪装剥离引擎
            if final_venue and "arxiv" in str(final_venue).lower():
                is_arxiv = True
                locations = item.get("locations", [])
                for loc in locations:
                    if loc and isinstance(loc, dict):
                        source = loc.get("source")
                        if source and isinstance(source, dict):
                            alt_venue = source.get("display_name", "")
                            if alt_venue and "arxiv" not in str(alt_venue).lower():
                                final_venue = alt_venue
                                rescued_count += 1
                                break

            if not final_venue:
                final_venue = "Unknown Venue"

            batch_records.append({
                "work_id": clean_id,
                "stage": current_phase,
                "venue": final_venue,
                "was_arxiv_originally": is_arxiv
            })
    except Exception as e:
        # 重试机制已经在 session 层面处理了，这里如果还报错，说明彻底无响应
        pass

    return batch_records, rescued_count


def run_ultimate_extraction():
    print(f"🚀 启动【群攻批量 + 连接池多线程】终极融合引擎...\n")

    SOURCE_PHASE_FILES = {
        "Phase1": r'E:\PythonProject\GenAI\data_scrap_v2\stage_1\phase1_hybrid_cleaned.jsonl',
        "Phase2": r'E:\PythonProject\GenAI\data_scrap_v2\stage_2\phase2_hybrid_cleaned.jsonl',
        "Phase3": r'E:\PythonProject\GenAI\data_scrap_v2\stage_3\phase3_hybrid_cleaned.jsonl'
    }

    global_session = create_robust_session()

    for current_phase, file_path in SOURCE_PHASE_FILES.items():
        print("\n" + "█" * 60)
        print(f"🎯 正在开启 {current_phase} 的极限抓取战役...")

        OUTPUT_CSV = f'{current_phase.lower()}_venues_mapping.csv'
        if not os.path.exists(file_path): continue

        # 1. 提取全量 ID
        all_ids = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    raw_id = data.get("work_id", "")
                    clean_id = str(raw_id).split('/')[-1]
                    if clean_id: all_ids.append(clean_id)
                except:
                    pass
        all_ids = list(set(all_ids))

        # 2. 读取断点
        processed_ids = set()
        if os.path.exists(OUTPUT_CSV):
            try:
                done_df = pd.read_csv(OUTPUT_CSV)
                processed_ids = set(done_df['work_id'].astype(str).tolist())
            except:
                pass

        target_ids = [vid for vid in all_ids if vid not in processed_ids]

        if not target_ids:
            print(f"🎉 {current_phase} 已经全量抓取完毕！")
            continue

        print(f"📥 剩余需抓取: {len(target_ids)} 篇。已启用 {MAX_WORKERS} 并发 × {BATCH_SIZE} 批量模式。")

        # 写入表头
        if not os.path.exists(OUTPUT_CSV):
            pd.DataFrame(columns=['work_id', 'stage', 'venue', 'was_arxiv_originally']).to_csv(OUTPUT_CSV, index=False,
                                                                                               encoding='utf-8-sig')

        # 切分批次
        batches = [target_ids[i:i + BATCH_SIZE] for i in range(0, len(target_ids), BATCH_SIZE)]
        total_rescued = 0

        # 3. 核心并发执行
        with open(OUTPUT_CSV, 'a', encoding='utf-8-sig', newline='') as f_out:
            # 初始化一个轻量级的 csv 写入器
            import csv
            writer = csv.writer(f_out)

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_batch = {executor.submit(fetch_batch_venues, batch, current_phase, global_session): batch for
                                   batch in batches}

                for future in tqdm(as_completed(future_to_batch), total=len(batches), desc=f"{current_phase} 抓取进度"):
                    records, rescued = future.result()
                    total_rescued += rescued

                    if records:
                        # 🛡️ 触发你的绝招：带锁的实时安全落盘！
                        with write_lock:
                            for rec in records:
                                writer.writerow(
                                    [rec['work_id'], rec['stage'], rec['venue'], rec['was_arxiv_originally']])
                            f_out.flush()

        print(f"✅ {current_phase} 多线程批量抓取结束！成功从 arXiv 救回 {total_rescued} 篇。")


if __name__ == "__main__":
    run_ultimate_extraction()