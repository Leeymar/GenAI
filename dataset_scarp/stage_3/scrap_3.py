import requests
import json
import time
import pandas as pd
import os


def fetch_phase3_heavy_duty():
    print("🚀 启动第三阶段 (2022-2026) 史诗级重装收割机...\n")

    base_url = "https://api.openalex.org/works"
    MY_EMAIL = "leeymar10@gmail.com"  # ⚠️ 务必填入你的邮箱，维持高速通道！
    headers = {"mailto": MY_EMAIL}

    print("🔍 正在锁定 3 大核心聚类 ID...")
    topic_cv_id = requests.get("https://api.openalex.org/topics",
                               params={"search": "Generative Adversarial Networks and Image Synthesis"}).json()[
        "results"][0]["id"].split('/')[-1]
    topic_nlp_id = \
    requests.get("https://api.openalex.org/topics", params={"search": "Natural Language Processing"}).json()["results"][
        0]["id"].split('/')[-1]
    topic_tm_id = \
    requests.get("https://api.openalex.org/topics", params={"search": "Topic Modeling"}).json()["results"][0][
        "id"].split('/')[-1]

    # 第三阶段终极 Filter 矩阵
    filter_a = (
        "publication_year:2022-2026,concepts.id:C41008148,"
        "title_and_abstract.search:\"Transformer\"|\"Transformers\"|\"GPT\"|\"Large Language Model\"|\"LLM\"|\"LLMs\"|"
        "\"Diffusion model\"|\"Stable Diffusion\"|\"Generative Adversarial\"|\"Image Synthesis\"|\"Text Generation\"|"
        "\"ChatGPT\"|\"GPT-4\"|\"LLaMA\"|\"Claude\"|\"Gemini\"|\"Midjourney\"|\"DALL-E\"|\"Sora\"|"
        "\"Prompt Engineering\"|\"In-context learning\"|\"Chain of Thought\"|\"RLHF\"|\"Retrieval-Augmented Generation\"|\"RAG\"|"
        "\"Generative AI\"|\"GenAI\"|\"Foundation Model\"|\"Artificial General Intelligence\""
    )
    filter_b = f"publication_year:2022-2026,topics.id:{topic_cv_id}"
    filter_c = (
        f"publication_year:2022-2026,topics.id:{topic_nlp_id}|{topic_tm_id},"
        "title_and_abstract.search:\"generative\"|\"generation\"|\"decoder\"|\"language model\"|\"seq2seq\"|\"prompt\"|\"chatgpt\"|\"llm\""
    )

    # ---------------------------------------------------------
    # 🛡️ 核心：带断点续传的重装下载引擎
    # ---------------------------------------------------------
    def download_track_safely(filters, track_id, track_name):
        data_file = f"phase3_temp_{track_id}.jsonl"
        cursor_file = f"phase3_temp_{track_id}.cursor"

        cursor = "*"
        fetched_count = 0

        if os.path.exists(cursor_file):
            with open(cursor_file, 'r') as cf:
                saved_cursor = cf.read().strip()
                if saved_cursor == "DONE":
                    print(f"✅ {track_name} 之前已彻底下载完成，直接跳过！")
                    return
                elif saved_cursor:
                    cursor = saved_cursor
                    print(f"🔄 发现断点！{track_name} 将从游标继续下载。")

        if os.path.exists(data_file):
            with open(data_file, 'r', encoding='utf-8') as f:
                fetched_count = sum(1 for _ in f)
            print(f"   本地已有 {fetched_count:,} 篇记录。")

        print(f"🚀 开始/继续下载 {track_name} ...")

        # 为了应对超大数据量，我们只提取绝对必要的字段，极大加快下载和去重速度！
        params = {
            "filter": filters,
            "per-page": 200,
            "cursor": cursor,
            "select": "id,title,publication_year,authorships,referenced_works,concepts"
        }

        with open(data_file, 'a', encoding='utf-8') as f:
            while params.get("cursor"):
                try:
                    res = requests.get(base_url, params=params, headers=headers)
                    if res.status_code == 429:
                        print("⚠️ 触发 OpenAlex 限流保护 (429)，自动冷却 10 秒...")
                        time.sleep(10)
                        continue

                    res.raise_for_status()
                    data = res.json()
                    results = data.get('results', [])

                    if not results: break

                    for work in results:
                        clean_work = {
                            "work_id": work.get('id'),
                            "title": work.get('title'),
                            "publication_year": work.get('publication_year'),
                            "author_ids": [a.get('author', {}).get('id') for a in work.get('authorships', []) if
                                           a.get('author')],
                            "referenced_works": work.get('referenced_works', []),
                            "concepts": [c.get('display_name') for c in work.get('concepts', []) if
                                         c.get('level') in [0, 1]]
                        }
                        f.write(json.dumps(clean_work, ensure_ascii=False) + '\n')

                    fetched_count += len(results)
                    print(f"   [{track_name}] 累计抓取: {fetched_count:,} 篇...")

                    next_cursor = data.get('meta', {}).get('next_cursor')
                    params["cursor"] = next_cursor
                    if next_cursor:
                        with open(cursor_file, 'w') as cf:
                            cf.write(next_cursor)

                    time.sleep(0.1)  # 极速模式，仅休眠 0.1 秒

                except Exception as e:
                    print(f"❌ 网络异常: {e}。等待 5 秒后重试...")
                    time.sleep(5)

        with open(cursor_file, 'w') as cf:
            cf.write("DONE")
        print(f"✅ {track_name} 完美收官，共计 {fetched_count:,} 篇。\n")

    # 执行三轨下载
    download_track_safely(filter_a, "A", "轨道 A (大模型与新范式)")
    download_track_safely(filter_b, "B", "轨道 B (CV视觉生成)")
    download_track_safely(filter_c, "C", "轨道 C (NLP与主题建模提纯)")

    # ---------------------------------------------------------
    # 🧬 融合与去重
    # ---------------------------------------------------------
    print("🧬 正在进行最终的 Pandas 三轨合并与极致去重 (这可能会消耗一些内存，请耐心等待)...")
    dfs = []
    for track_id in ["A", "B", "C"]:
        temp_file = f"phase3_temp_{track_id}.jsonl"
        if os.path.exists(temp_file):
            dfs.append(pd.read_json(temp_file, lines=True))

    if dfs:
        df_final = pd.concat(dfs, ignore_index=True)
        df_final = df_final.drop_duplicates(subset=['work_id'], keep='first')

        final_filename = "phase3_hybrid_cleaned.jsonl"
        df_final.to_json(final_filename, orient='records', lines=True, force_ascii=False)

        # 清理临时文件释放硬盘空间
        for track_id in ["A", "B", "C"]:
            os.remove(f"phase3_temp_{track_id}.jsonl")
            os.remove(f"phase3_temp_{track_id}.cursor")

        print("-" * 64)
        print(f"🎉 终极胜利！第三阶段 (2022-2026) 纯净数据集已生成。")
        print(f"🏆 去重后有效文献总计: 【 {len(df_final):,.0f} 】 篇")
        print(f"📂 已保存为: {final_filename}")
        print("-" * 64)


# 开启这场史诗级的收割吧！
fetch_phase3_heavy_duty()