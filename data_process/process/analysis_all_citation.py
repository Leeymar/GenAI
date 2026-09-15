import json
import os
import time
import requests
import threading
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ==========================================
# ⚙️ 核心配置区
# ==========================================
YOUR_EMAIL = "2212408015@stmail.ujs.edu.cn"
OPENALEX_API_KEY = "JnzLogerSBHjQLbbhdSPoh"

CITATION_FILE = 'citation_mapping.json'
OUTPUT_FILE = 'citing_disciplines_master.jsonl'

# 🚀 并发数配置
MAX_WORKERS = 20

# 🛡️ 文件写入安全锁
write_lock = threading.Lock()


def create_robust_session():
    session = requests.Session()
    adapter = HTTPAdapter(
        pool_connections=MAX_WORKERS,
        pool_maxsize=MAX_WORKERS,
        max_retries=Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    )
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


def fetch_single_work(work_id, session):
    clean_id = work_id.split('/')[-1]
    base_url = f"https://api.openalex.org/works?filter=cites:{clean_id}&select=id,concepts&per-page=200"

    if YOUR_EMAIL:
        base_url += f"&mailto={YOUR_EMAIL}"
    if OPENALEX_API_KEY:
        base_url += f"&api_key={OPENALEX_API_KEY}"

    discipline_counts = {}
    cursor = "*"

    while cursor:
        url = f"{base_url}&cursor={cursor}"
        try:
            response = session.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()

            for citing_work in data.get('results', []):
                for concept in citing_work.get('concepts', []):
                    if concept.get('level') == 0:
                        discipline_name = concept.get('display_name')
                        if discipline_name:
                            discipline_counts[discipline_name] = discipline_counts.get(discipline_name, 0) + 1

            cursor = data.get('meta', {}).get('next_cursor')

        except Exception as e:
            time.sleep(2)
            break

    return work_id, discipline_counts


def run_multithread_extraction():
    print(f"🚀 启动【尊享 VIP API Key 多线程极速版】(并发数: {MAX_WORKERS})...\n")

    # 1. 加载目标清单
    if not os.path.exists(CITATION_FILE):
        print(f"❌ 找不到清单文件 {CITATION_FILE}，请检查路径！")
        return

    with open(CITATION_FILE, 'r', encoding='utf-8') as f:
        citation_mapping = json.load(f)

    # ==========================================
    # 🌟 核心战略修改：从低引用到高引用排序
    # ==========================================
    print("📊 正在执行战术排序：优先处理低引用的长尾论文，加速收割...")

    # 提取 (论文ID, 引用量) 的组合，并排除引用量为0的
    valid_items = [(wid, info.get('citations', 0)) for wid, info in citation_mapping.items() if
                   info.get('citations', 0) > 0]

    # 按照引用量（索引为1的元素）从小到大进行升序排列！
    valid_items.sort(key=lambda x: x[1])

    # 提取排序后的纯 ID 列表
    all_target_ids = [item[0] for item in valid_items]
    # ==========================================

    # 2. 读取断点续传进度
    processed_ids = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    processed_ids.add(json.loads(line)['work_id'])
                except:
                    pass
        print(f"✅ 断点续传：已跳过本地完成的 {len(processed_ids)} 篇。")

    # 3. 计算剩余任务（由于之前排过序，这里剔除已完成任务后，依然保持从小到大的顺序）
    todo_ids = [wid for wid in all_target_ids if wid not in processed_ids]
    print(f"⏳ 剩余需要极速处理的论文: {len(todo_ids)}\n")

    if not todo_ids:
        print("🎉 这部分任务已经全部抓完啦！")
        return

    global_session = create_robust_session()

    # 4. 开启多线程狂飙模式
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f_out:

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # 提交任务
            future_to_id = {executor.submit(fetch_single_work, wid, global_session): wid for wid in todo_ids}

            # 监控进度
            for future in tqdm(as_completed(future_to_id), total=len(todo_ids), desc="🕸️ 正在从低到高疯狂收割"):
                try:
                    work_id, disciplines = future.result()
                    result_record = {
                        "work_id": work_id,
                        "citing_disciplines": disciplines
                    }

                    with write_lock:
                        f_out.write(json.dumps(result_record, ensure_ascii=False) + '\n')
                        f_out.flush()

                except Exception as e:
                    pass

    print("\n💾 伟大的工程完成！数据安全入库！")


if __name__ == "__main__":
    run_multithread_extraction()