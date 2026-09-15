import pandas as pd
import requests
import os
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

MAX_WORKERS = 15  # 并发线程数（你的 API 权限可以支撑这个速度）
BATCH_SIZE = 50  # 批量查询数

write_lock = threading.Lock()


def create_robust_session():
    """建立极其稳定的底层连接池"""
    session = requests.Session()
    adapter = HTTPAdapter(
        pool_connections=MAX_WORKERS,
        pool_maxsize=MAX_WORKERS,
        max_retries=Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    )
    session.mount('https://', adapter)
    return session


def fetch_arxiv_keys_worker(batch_w_ids, session):
    """子线程：拿 50 个 W_ID 去换 arXiv 身份证"""
    w_id_string = "|".join(batch_w_ids)
    api_url = f"https://api.openalex.org/works?filter=openalex:{w_id_string}&select=id,ids,primary_location&per-page={BATCH_SIZE}&api_key={OPENALEX_API_KEY}&mailto={YOUR_EMAIL}"

    extracted_keys = []
    try:
        response = session.get(api_url, timeout=20)
        response.raise_for_status()
        results = response.json().get('results', [])

        for item in results:
            w_id = str(item.get('id', '')).split('/')[-1]
            arxiv_id = None

            # 强力破解 OpenAlex 里的各种隐身术
            doi_url = item.get("ids", {}).get("doi", "")
            if "arxiv." in doi_url:
                arxiv_id = doi_url.split("arxiv.")[-1]
            if not arxiv_id:
                landing_url = item.get("primary_location", {}).get("landing_page_url", "")
                if landing_url and "/abs/" in landing_url:
                    arxiv_id = landing_url.split("/abs/")[-1]
            if not arxiv_id:
                arxiv_url = item.get("ids", {}).get("arxiv", "")
                if arxiv_url and "/abs/" in arxiv_url:
                    arxiv_id = arxiv_url.split("/abs/")[-1]

            if arxiv_id:
                clean_arxiv_id = arxiv_id.split('v')[0]  # 剥离 v1/v2 版本号
                extracted_keys.append({"work_id": w_id, "arxiv_id": clean_arxiv_id})
    except Exception as e:
        # 重试已在底层处理，实在不行就跳过
        pass

    return extracted_keys


def run_step1_extract_keys():
    print("🚀 启动【第一步：OpenAlex 高并发云端钥匙提取引擎】...\n")

    PHASES = ['phase1', 'phase2', 'phase3']
    global_session = create_robust_session()

    for phase in PHASES:
        print("\n" + "█" * 60)
        print(f"🎯 正在执行 {phase} 的钥匙提取战役...")

        mapping_csv = f"{phase}_venues_mapping.csv"
        output_csv = f"{phase}_arxiv_keys.csv"

        if not os.path.exists(mapping_csv):
            print(f"⚠️ 找不到阵地表 {mapping_csv}，已跳过。")
            continue

        # 1. 获取目标名单
        df_map = pd.read_csv(mapping_csv, dtype=str)
        # 兼容 "ArXiv.org" 和 "arXiv (Cornell University)"
        mask = (df_map['was_arxiv_originally'] == 'True') | (
            df_map['venue'].fillna('').str.lower().str.contains('arxiv'))
        target_w_ids = list(set(df_map[mask]['work_id'].tolist()))

        # 2. 断点续传保护
        processed_w_ids = set()
        if os.path.exists(output_csv):
            try:
                done_df = pd.read_csv(output_csv, dtype=str)
                processed_w_ids = set(done_df['work_id'].tolist())
            except:
                pass

        target_w_ids = [w for w in target_w_ids if w not in processed_w_ids]

        if not target_w_ids:
            print(f"🎉 {phase} 全量钥匙已提取完毕！")
            continue

        print(f"📥 剩余需云端破译: {len(target_w_ids)} 篇。已启用 {MAX_WORKERS} 并发 × {BATCH_SIZE} 批量模式。")

        # 写入表头
        if not os.path.exists(output_csv):
            pd.DataFrame(columns=['work_id', 'arxiv_id']).to_csv(output_csv, index=False, encoding='utf-8-sig')

        batches = [target_w_ids[i:i + BATCH_SIZE] for i in range(0, len(target_w_ids), BATCH_SIZE)]
        total_keys_extracted = 0

        # 3. 核心并发执行
        with open(output_csv, 'a', encoding='utf-8-sig', newline='') as f_out:
            import csv
            writer = csv.writer(f_out)

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_batch = {executor.submit(fetch_arxiv_keys_worker, batch, global_session): batch for batch in
                                   batches}

                for future in tqdm(as_completed(future_to_batch), total=len(batches), desc=f"{phase} 破译进度"):
                    records = future.result()

                    if records:
                        total_keys_extracted += len(records)
                        # 🛡️ 加锁实时落盘，绝对安全
                        with write_lock:
                            for rec in records:
                                writer.writerow([rec['work_id'], rec['arxiv_id']])
                            f_out.flush()

        print(f"✅ {phase} 钥匙提取战役结束！成功拿到 {total_keys_extracted} 把 arXiv 钥匙。")


if __name__ == "__main__":
    run_step1_extract_keys()