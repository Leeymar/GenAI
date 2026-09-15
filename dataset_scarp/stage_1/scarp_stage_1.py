import requests
import json
import time
import pandas as pd
import os


def fetch_and_merge_hybrid_data():
    base_url = "https://api.openalex.org"

    # ---------------------------------------------------------
    # 1. 自动获取 Topic ID
    # ---------------------------------------------------------
    print("🔍 正在获取目标 Topic ID...")
    topic_res = requests.get(f"{base_url}/topics",
                             params={"search": "Generative Adversarial Networks and Image Synthesis"}).json()
    topic_id = topic_res["results"][0]["id"].split('/')[-1]
    print(f"✅ 锁定 Topic: {topic_id}")

    # ---------------------------------------------------------
    # 2. 定义两个抓取轨道的 Filters
    # ---------------------------------------------------------
    filter_a = (
        "publication_year:2014-2016,concepts.id:C41008148,"
        "title_and_abstract.search:\"Generative Adversarial\"|\"Variational Autoencoder\"|\"Deep Generative\"|"
        "\"DCGAN\"|\"LAPGAN\"|\"InfoGAN\"|\"BiGAN\"|\"CoGAN\"|\"Normalizing Flow\"|\"Real NVP\"|"
        "\"Sequence to Sequence\"|\"Seq2Seq\"|\"Neural Language Model\"|\"Text Generation\"|\"Neural Machine Translation\""
    )
    filter_b = f"publication_year:2014-2016,topics.id:{topic_id}"

    # ---------------------------------------------------------
    # 3. 核心下载函数（带游标翻页和极简清洗）
    # ---------------------------------------------------------
    def download_set(filters, temp_filename, set_name):
        print(f"\n🚀 开始下载 {set_name} 数据...")
        params = {"filter": filters, "per-page": 200, "cursor": "*"}
        total_fetched = 0

        with open(temp_filename, 'w', encoding='utf-8') as f:
            while params["cursor"]:
                try:
                    res = requests.get(f"{base_url}/works", params=params)
                    res.raise_for_status()
                    data = res.json()
                    results = data.get('results', [])
                    if not results: break

                    for work in results:
                        clean_work = {
                            "work_id": work.get('id'),
                            "title": work.get('title'),
                            "publication_year": work.get('publication_year'),
                            "author_ids": [
                                a.get('author').get('id') for a in work.get('authorships', [])
                                if a.get('author') and a.get('author').get('id')
                            ],
                            "referenced_works": work.get('referenced_works', []),
                            "concepts": [c.get('display_name') for c in work.get('concepts', []) if
                                         c.get('level') in [0, 1]]
                        }
                        f.write(json.dumps(clean_work, ensure_ascii=False) + '\n')

                    total_fetched += len(results)
                    print(f"   已抓取 {total_fetched} 篇...")
                    params["cursor"] = data.get('meta', {}).get('next_cursor')
                    time.sleep(0.1)
                except Exception as e:
                    print(f"❌ {set_name} 下载中断: {e}")
                    break
        print(f"✅ {set_name} 下载完成，共 {total_fetched} 篇。")

    # 执行下载，分别存入临时文件
    temp_a = "temp_set_a.jsonl"
    temp_b = "temp_set_b.jsonl"
    download_set(filter_a, temp_a, "集合A (全拼与模型矩阵)")
    download_set(filter_b, temp_b, "集合B (Topic聚类矩阵)")

    # ---------------------------------------------------------
    # 4. Pandas 终极去重与合并
    # ---------------------------------------------------------
    print("\n🧬 正在使用 Pandas 进行数据合并与去重...")
    df_a = pd.read_json(temp_a, lines=True) if os.path.exists(temp_a) else pd.DataFrame()
    df_b = pd.read_json(temp_b, lines=True) if os.path.exists(temp_b) else pd.DataFrame()

    # 纵向拼接，并根据 work_id 剔除重复项
    df_final = pd.concat([df_a, df_b], ignore_index=True)
    df_final = df_final.drop_duplicates(subset=['work_id'], keep='first')

    # 保存最终大一统文件
    final_filename = "phase1_hybrid_final.jsonl"
    df_final.to_json(final_filename, orient='records', lines=True, force_ascii=False)

    # 清理临时文件 (保持硬盘干净)
    if os.path.exists(temp_a): os.remove(temp_a)
    if os.path.exists(temp_b): os.remove(temp_b)

    print("-" * 50)
    print(f"🎉 大功告成！第一阶段最终纯净数据集已生成。")
    print(f"🏆 去重后最终有效文献量: 【 {len(df_final)} 】 篇")
    print(f"📂 终极数据文件已保存为: {final_filename}")
    print("-" * 50)


# 执行！
fetch_and_merge_hybrid_data()