import requests
import json
import time
import pandas as pd
import os


def fetch_phase1_ultimate_engine():
    print("🧬 启动【第一阶段 (2014-2016)】创世拓荒者全量打捞引擎 (数据结构对齐版)...")
    base_url = "https://api.openalex.org/works"

    # ⚠️ 维持合法高速通道
    MY_EMAIL = "leeymar10@gmail.com"
    headers = {"mailto": MY_EMAIL}

    # ---------------------------------------------------------
    # 1. 绝对防御：直接焊死 OpenAlex 官方的底层 Topic IDs
    # ---------------------------------------------------------
    topic_cv_id = "T11914"  # Generative Adversarial Networks and Image Synthesis
    topic_nlp_id = "T10181"  # Natural Language Processing
    topic_tm_id = "T11497"  # Topic Modeling (完美对齐你的 Phase 3)

    print(f"   ➤ 锁定 视觉生成大类: {topic_cv_id}")
    print(f"   ➤ 锁定 语言处理大类: {topic_nlp_id}")
    print(f"   ➤ 锁定 主题建模交叉: {topic_tm_id}")

    # ---------------------------------------------------------
    # 2. 四大 AI 核心概念池 (彻底解除 CV 单科死锁，火力全开)
    # ---------------------------------------------------------
    core_concepts = "C41008148|C204321447|C154945302|C119857082"

    # ---------------------------------------------------------
    # 3. 四轨 Filter 定义：原汁原味的专有名词 + 交叉过滤
    # ---------------------------------------------------------
    # 轨道A: 视觉与基础生成主线 (涵盖极早期硬核专有名词)
    v1_keys = (
        "\"generative adversarial\"|\"variational autoencoder\"|\"deep generative\"|"
        "\"dcgan\"|\"lapgan\"|\"infogan\"|\"bigan\"|\"cogan\"|\"normalizing flow\"|\"real nvp\"|"
        "\"image synthesis\"|\"text-to-image\""
    )
    filter_a = f"publication_year:2014-2016,concepts.id:{core_concepts},title_and_abstract.search:{v1_keys}"

    # 轨道B: 语言生成早期主线 (LLM 的启蒙老祖宗，解除锁定后全量释放)
    n1_keys = (
        "\"sequence to sequence\"|\"seq2seq\"|\"neural language model\"|\"text generation\"|"
        "\"neural machine translation\"|\"autoregressive language model\""
    )
    filter_b = f"publication_year:2014-2016,concepts.id:{core_concepts},title_and_abstract.search:{n1_keys}"

    # 轨道C: NLP与Topic Modeling 交叉兜底 (引入 n1_keys 安全锁，防止抓回纯传统 LDA)
    filter_c = f"publication_year:2014-2016,topics.id:{topic_nlp_id}|{topic_tm_id},title_and_abstract.search:{n1_keys}"

    # 轨道D: 纯粹的 CV 官方 Topic 兜底 (利用 OpenAlex 的后向追溯算法抓漏网之鱼)
    filter_d = f"publication_year:2014-2016,topics.id:{topic_cv_id}"

    # ---------------------------------------------------------
    # 4. 核心：带断点续传的重装下载引擎
    # ---------------------------------------------------------
    def download_track_safely(filters, track_id, track_name):
        data_file = f"phase1_temp_{track_id}.jsonl"
        cursor_file = f"phase1_temp_{track_id}.cursor"

        cursor = "*"
        fetched_count = 0

        # 断点检查
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

        print(f"🚀 开始下载 {track_name} ...")

        # ⚠️ 这里的 select 已经完美对齐了 Phase 3，确保信息极其饱满
        params = {
            "filter": filters,
            "per-page": 200,
            "cursor": cursor,
            "select": "id,title,publication_year,authorships,referenced_works,concepts"
        }

        with open(data_file, 'a', encoding='utf-8') as f:
            while params.get("cursor"):
                try:
                    res = requests.get(base_url, params=params, headers=headers, timeout=20)

                    if res.status_code == 429:
                        print("⚠️ 触发 OpenAlex 限流保护 (429)，冷却 10 秒...")
                        time.sleep(10)
                        continue

                    res.raise_for_status()
                    data = res.json()
                    results = data.get('results', [])

                    if not results: break

                    for work in results:
                        # ⚠️ 这里的字典结构已完美对齐 Phase 3，原汁原味
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

                    time.sleep(0.1)

                except Exception as e:
                    print(f"❌ 网络异常: {e}。等待 5 秒后重试...")
                    time.sleep(5)

        with open(cursor_file, 'w') as cf:
            cf.write("DONE")
        print(f"✅ {track_name} 收官，共计 {fetched_count:,} 篇。\n")

    # 启动四轨并行下载
    download_track_safely(filter_a, "A", "轨道 A (大概念池 + 视觉先驱关键字)")
    download_track_safely(filter_b, "B", "轨道 B (大概念池 + 语言生成先驱关键字)")
    download_track_safely(filter_c, "C", "轨道 C (NLP与主题建模安全交叉兜底)")
    download_track_safely(filter_d, "D", "轨道 D (纯视觉生成官方 Topic 兜底)")

    # ---------------------------------------------------------
    # 5. Pandas 终极合并与极致去重
    # ---------------------------------------------------------
    print("🧬 正在进行最终的 Pandas 四轨合并与极致去重...")
    dfs = []
    for track_id in ["A", "B", "C", "D"]:
        temp_file = f"phase1_temp_{track_id}.jsonl"
        if os.path.exists(temp_file):
            try:
                df = pd.read_json(temp_file, lines=True)
                if not df.empty:
                    dfs.append(df)
            except Exception as e:
                pass

    if dfs:
        df_final = pd.concat(dfs, ignore_index=True)
        # 根据 work_id 绝对去重
        df_final = df_final.drop_duplicates(subset=['work_id'], keep='first')

        final_filename = "phase1_hybrid_cleaned.jsonl"
        df_final.to_json(final_filename, orient='records', lines=True, force_ascii=False)

        # 自动清理临时文件，释放硬盘空间
        for track_id in ["A", "B", "C", "D"]:
            if os.path.exists(f"phase1_temp_{track_id}.jsonl"): os.remove(f"phase1_temp_{track_id}.jsonl")
            if os.path.exists(f"phase1_temp_{track_id}.cursor"): os.remove(f"phase1_temp_{track_id}.cursor")

        print("-" * 64)
        print(f"🎉 终极胜利！第一阶段 (2014-2016) 完美对齐版纯净数据重建完毕。")
        print(f"🏆 剔除冗余后，创世文献总计: 【 {len(df_final):,.0f} 】 篇")
        print(f"📂 终极宝藏已安全保存为: {final_filename}")
        print("-" * 64)
    else:
        print("❓ 没有获取到有效数据，请检查网络环境。")


# 启动引擎！
fetch_phase1_ultimate_engine()