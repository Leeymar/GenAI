import pandas as pd
import requests
import time
import math
import os
from tqdm import tqdm


def fetch_author_institutions_robust():
    print("🚀 启动【OpenAlex 机构实时抓取引擎】(防崩溃 + API Key 版)...\n")

    # ==========================================
    # 1. 账号与文件配置 (请填入你的信息)
    # ==========================================
    API_KEY = "3VymPsRtP5VnRD1hZDR1OQ"  # 你的专属秘钥
    EMAIL = "2212408015@stu.ujs.edu.cn"  # 你的邮箱

    input_file = 'arXiv_Balanced_DID_Panel.csv'
    output_file = 'Author_Institutions_Map.csv'

    # ==========================================
    # 2. 读取名单并实现【断点续传】逻辑
    # ==========================================
    try:
        df_panel = pd.read_csv(input_file)
        all_authors = df_panel['author_id'].dropna().unique().tolist()
        clean_ids = [aid.split('/')[-1] for aid in all_authors]
    except FileNotFoundError:
        print(f"❌ 找不到 {input_file}，请确认路径！")
        return

    # 检查是否已经有部分抓取结果（断点续传）
    already_scraped_ids = set()
    if os.path.exists(output_file):
        try:
            df_existing = pd.read_csv(output_file)
            already_scraped_ids = set(df_existing['author_id'].str.split('/').str[-1])
            print(f"📂 发现本地已有记录！已跳过 {len(already_scraped_ids):,} 个已抓取的作者。")
        except Exception as e:
            pass
    else:
        # 如果文件不存在，先写个表头进去
        pd.DataFrame(columns=['author_id', 'institution', 'country_code']).to_csv(output_file, index=False)

    # 剔除已经抓过的，只保留还需要抓的
    ids_to_scrape = [aid for aid in clean_ids if aid not in already_scraped_ids]

    if not ids_to_scrape:
        print("✅ 所有作者都已经抓取完毕，无需重复抓取！")
        return

    print(f"🎯 本次任务需要抓取 {len(ids_to_scrape):,} 位作者。")

    # ==========================================
    # 3. 配置 API 与高并发请求
    # ==========================================
    headers = {"User-Agent": f"mailto:{EMAIL}"}
    base_url = "https://api.openalex.org/authors"

    batch_size = 50
    total_batches = math.ceil(len(ids_to_scrape) / batch_size)

    print(f"⚙️ 开始分批请求 API，共计 {total_batches} 批次，数据将实时落盘...")

    for i in tqdm(range(total_batches), desc="抓取进度"):
        batch_ids = ids_to_scrape[i * batch_size: (i + 1) * batch_size]
        id_filter = "|".join(batch_ids)

        params = {
            "filter": f"openalex:{id_filter}",
            "per-page": batch_size,
            "select": "id,last_known_institutions",
            "api_key": API_KEY  # 💥 挂载你的专属 API Key
        }

        batch_results = []
        try:
            # 增加 timeout 防止死锁
            response = requests.get(base_url, params=params, headers=headers, timeout=15)
            if response.status_code == 200:
                results = response.json().get('results', [])

                for author in results:
                    author_id = author.get('id')
                    institutions = author.get('last_known_institutions', [])

                    if institutions and len(institutions) > 0:
                        primary_inst = institutions[0]
                        inst_name = primary_inst.get('display_name', '')
                        country_code = primary_inst.get('country_code', '')
                    else:
                        inst_name = "Unknown"
                        country_code = "Unknown"

                    batch_results.append({
                        'author_id': author_id,
                        'institution': inst_name,
                        'country_code': country_code
                    })
            elif response.status_code == 429:
                print(f"\n⚠️ 触发限流 (429)，休眠 5 秒...")
                time.sleep(5)
            else:
                print(f"\n⚠️ 第 {i} 批请求异常，状态码: {response.status_code}")
                time.sleep(1)

        except Exception as e:
            print(f"\n❌ 第 {i} 批网络请求报错: {e}")
            time.sleep(2)

        # ==========================================
        # 4. 【核心防御】实时写入，边抓边存
        # ==========================================
        if batch_results:
            df_batch = pd.DataFrame(batch_results)
            # mode='a' 表示追加写入 (Append)，不写表头
            df_batch.to_csv(output_file, mode='a', header=False, index=False)

        # 挂载了 API Key 速度可以稍快，但还是建议保留极其微小的间隔防封
        time.sleep(0.05)

    print("\n=========================================================")
    print(f"🎉 增量抓取任务圆满结束！所有数据已安全落盘至: {output_file}")
    print("=========================================================")


fetch_author_institutions_robust()