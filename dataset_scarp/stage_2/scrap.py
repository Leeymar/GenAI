import requests
import json
import time
import pandas as pd
import os


def fetch_phase2_bulletproof():
    base_url = "https://api.openalex.org/works"

    # ⚠️ 填入你的邮箱，这是进入 Polite Pool（高速合法通道）的关键！
    MY_EMAIL = "leeymar10@gmail.com"
    headers = {"mailto": MY_EMAIL}

    print("🔍 正在锁定 CV 和 NLP 核心聚类 ID...")
    topic_cv_id = requests.get("https://api.openalex.org/topics",
                               params={"search": "Generative Adversarial Networks and Image Synthesis"}).json()[
        "results"][0]["id"].split('/')[-1]
    topic_nlp_id = \
    requests.get("https://api.openalex.org/topics", params={"search": "Natural Language Processing"}).json()["results"][
        0]["id"].split('/')[-1]

    # 三轨 Filter 定义 (与之前完全一致)
    filter_a = (
        "publication_year:2017-2021,concepts.id:C41008148,"
        "title_and_abstract.search:\"Transformer\"|\"Transformers\"|\"Pre-trained Language\"|\"Generative Pre-trained\"|"
        "\"GPT\"|\"BERT\"|\"Large Language Model\"|\"Diffusion model\"|\"Denoising diffusion\"|\"VQ-VAE\"|\"VQGAN\"|"
        "\"Generative Adversarial\"|\"StyleGAN\"|\"CycleGAN\"|\"Image Synthesis\"|\"Text Generation\"|\"Seq2Seq\""
    )
    filter_b = f"publication_year:2017-2021,topics.id:{topic_cv_id}"
    filter_c = (
        f"publication_year:2017-2021,topics.id:{topic_nlp_id},"
        "title_and_abstract.search:\"generative\"|\"generation\"|\"decoder\"|\"language model\"|\"seq2seq\""
    )

    # ---------------------------------------------------------
    # 🛡️ 核心：带断点续传的稳健下载引擎
    # ---------------------------------------------------------
    def download_track_safely(filters, track_id, track_name):
        data_file = f"phase2_temp_{track_id}.jsonl"
        cursor_file = f"phase2_temp_{track_id}.cursor"

        cursor = "*"
        fetched_count = 0

        # 1. 检查是否存在游标存档（断点恢复）
        if os.path.exists(cursor_file):
            with open(cursor_file, 'r') as cf:
                saved_cursor = cf.read().strip()
                if saved_cursor == "DONE":
                    print(f"✅ {track_name} 之前已彻底下载完成，直接跳过！")
                    return
                elif saved_cursor:
                    cursor = saved_cursor
                    print(f"🔄 发现断点存档！{track_name} 将从游标 {cursor[:10]}... 继续下载。")

        # 2. 统计本地已下载的数据量（用于正确的日志打印）
        if os.path.exists(data_file):
            with open(data_file, 'r', encoding='utf-8') as f:
                fetched_count = sum(1 for _ in f)
            print(f"   本地已有 {fetched_count} 篇记录。")

        print(f"🚀 开始/继续下载 {track_name} ...")
        params = {"filter": filters, "per-page": 200, "cursor": cursor}

        # 注意：这里使用 'a' (追加模式)，及时落盘，绝对不会覆盖之前的数据
        with open(data_file, 'a', encoding='utf-8') as f:
            while params.get("cursor"):
                try:
                    res = requests.get(base_url, params=params, headers=headers)

                    # 捕捉 API 官方的限流警告
                    if res.status_code == 429:
                        print("⚠️ 触发 OpenAlex 限流保护 (429)，脚本自动冷却 10 秒...")
                        time.sleep(10)
                        continue

                    res.raise_for_status()
                    data = res.json()
                    results = data.get('results', [])

                    if not results: break  # 这一轨抓完了

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
                    print(f"   [{track_name}] 累计抓取: {fetched_count} 篇...")

                    # 更新通行证并实时存档
                    next_cursor = data.get('meta', {}).get('next_cursor')
                    params["cursor"] = next_cursor
                    if next_cursor:
                        with open(cursor_file, 'w') as cf:
                            cf.write(next_cursor)

                    # 温和降频：单次请求后固定休息 0.2 秒，既快又安全
                    time.sleep(0.2)

                except Exception as e:
                    print(f"❌ 网络异常: {e}。等待 5 秒后重试当前页...")
                    time.sleep(5)

        # 标记当前轨道全部完成
        with open(cursor_file, 'w') as cf:
            cf.write("DONE")
        print(f"✅ {track_name} 完美收官，共计 {fetched_count} 篇。\n")

    # 执行三轨下载 (即使中断也能完美接力)
    download_track_safely(filter_a, "A", "轨道 A (硬核大模型基座)")
    download_track_safely(filter_b, "B", "轨道 B (CV视觉生成聚类)")
    download_track_safely(filter_c, "C", "轨道 C (NLP语境结界提纯)")

    # ---------------------------------------------------------
    # 🧬 融合与去重 (只有三个轨道的 .cursor 全是 DONE 时才会走到这里)
    # ---------------------------------------------------------
    print("🧬 正在进行 Pandas 最终三轨合并与去重...")
    dfs = []
    for track_id in ["A", "B", "C"]:
        temp_file = f"phase2_temp_{track_id}.jsonl"
        if os.path.exists(temp_file):
            dfs.append(pd.read_json(temp_file, lines=True))

    if dfs:
        df_final = pd.concat(dfs, ignore_index=True)
        df_final = df_final.drop_duplicates(subset=['work_id'], keep='first')

        final_filename = "phase2_hybrid_cleaned.jsonl"
        df_final.to_json(final_filename, orient='records', lines=True, force_ascii=False)

        print("-" * 64)
        print(f"🎉 终极胜利！第二阶段 (2017-2021) 纯净数据集合并完毕。")
        print(f"🏆 去重后有效文献总计: 【 {len(df_final):,.0f} 】 篇")
        print(f"📂 已保存为: {final_filename}")
        print("-" * 64)
    else:
        print("❓ 没有找到任何临时文件，请检查下载过程。")


fetch_phase2_bulletproof()