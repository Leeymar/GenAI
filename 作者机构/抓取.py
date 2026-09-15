import pandas as pd
import json
import requests
import time
from tqdm import tqdm
import os

# ==========================================
# ⚙️ 配置区域 (请在此处填入您的信息)
# ==========================================
API_KEY = "j8aosOdEHrGKJaNMn6sePz"  # 如果有 Premium API Key 填这里，没有留空
EMAIL = "2212408015@stmail.ujs.edu.cn"  # 强烈建议填入，进入 Polite Pool (礼貌池)

CSV_PATH = r"E:\PythonProject\GenAI\data_process\test\Ultimate_Regression_Base_with_Diversity.csv"
JSONL_FILES = [
    r"E:\PythonProject\GenAI\data_scrap_v2\stage_1\phase1_hybrid_cleaned.jsonl",
    r"E:\PythonProject\GenAI\data_scrap_v2\stage_2\phase2_hybrid_cleaned.jsonl",
    r"E:\PythonProject\GenAI\data_scrap_v2\stage_3\phase3_hybrid_cleaned.jsonl"
]
OUTPUT_FILE = r"/作者机构/Author_Institutions_Cleaned.jsonl"

# ==========================================
# 步骤 1 & 2：读取论文名单并提取所有目标作者
# ==========================================
print("▶ 正在读取论文清单与作者 JSONL 文件...")
df_base = pd.read_csv(CSV_PATH)
valid_work_ids = set(df_base['clean_work_id'].dropna().astype(str).tolist())

unique_authors = set()
for file in JSONL_FILES:
    if not os.path.exists(file):
        continue
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                wid = data.get('work_id', '').split('/')[-1]
                if wid in valid_work_ids:
                    for author_url in data.get('author_ids', []):
                        aid = author_url.split('/')[-1]
                        unique_authors.add(aid)
            except Exception:
                continue

print(f"  √ 提取完毕！文献中共有 {len(unique_authors)} 位唯一的作者。")

# ==========================================
# 步骤 3：【核心】JSONL 断点续传逻辑
# ==========================================
processed_authors = set()
if os.path.exists(OUTPUT_FILE):
    print(f"\n▶ 发现已存在的 JSONL 输出文件，正在读取已保存的进度...")
    try:
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    if 'author_id' in record:
                        processed_authors.add(record['author_id'])
        print(f"  √ 已加载 {len(processed_authors)} 位已处理过的作者，将自动跳过他们。")
    except Exception as e:
        print(f"  [警告] 读取进度失败: {e}")

pending_authors = list(unique_authors - processed_authors)
print(f"  ★ 本次实际需要向 API 查询的作者数量: {len(pending_authors)}")

if len(pending_authors) == 0:
    print("\n🎉 所有作者的机构信息均已抓取完毕，无需继续执行！")
    exit()

# ==========================================
# 步骤 4：批量查询并【实时追加保存为 JSONL】
# ==========================================
headers = {"User-Agent": f"mailto:{EMAIL}"}
chunk_size = 100

print("\n▶ 开始向 OpenAlex API 获取数据 (随时可中断，下次自动续传)...")
for i in tqdm(range(0, len(pending_authors), chunk_size)):
    chunk = pending_authors[i:i + chunk_size]
    filter_str = "|".join(chunk)

    url = f"https://api.openalex.org/authors?filter=openalex:{filter_str}&per-page={chunk_size}"
    if API_KEY: url += f"&api_key={API_KEY}"

    max_retries = 3
    current_batch_data = []

    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=20)
            if response.status_code == 200:
                results = response.json().get('results', [])
                for auth in results:
                    auth_id = auth.get('id', '').split('/')[-1]
                    name = auth.get('display_name', '')

                    # 🐞 核心修复：OpenAlex 中是复数 last_known_institutions 且为列表
                    inst_list = auth.get('last_known_institutions', [])
                    # 取列表中的第一个机构作为代表机构，如果为空则设为空字典
                    inst = inst_list[0] if inst_list else {}

                    current_batch_data.append({
                        'author_id': auth_id,
                        'author_name': name,
                        'institution_name': inst.get('display_name'),
                        'country_code': inst.get('country_code'),
                        'institution_type': inst.get('type')
                    })

                # 实时追加写入
                if current_batch_data:
                    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f_out:
                        for record in current_batch_data:
                            f_out.write(json.dumps(record, ensure_ascii=False) + '\n')

                break

            elif response.status_code == 429:
                time.sleep(2)
            else:
                print(f"\n  [错误] 状态码: {response.status_code}, 稍后将跳过此批次。")
                break
        except Exception as e:
            time.sleep(2)

    if not API_KEY:
        time.sleep(0.15)

print(f"\n▶ 抓取任务结束！数据已实时安全写入 JSONL 文件:\n  {OUTPUT_FILE}")