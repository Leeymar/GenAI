import requests


def probe_historical_missing_topics():
    print("📡 启动历史阶段 (Phase 1 & 2) 统一口径补漏雷达...\n")

    base_url = "https://api.openalex.org"
    MY_EMAIL = "your_email@example.com"  # 填入邮箱

    # 获取 3 大核心聚类 ID
    try:
        topic_cv_id = requests.get(f"{base_url}/topics",
                                   params={"search": "Generative Adversarial Networks and Image Synthesis"}).json()[
            "results"][0]["id"].split('/')[-1]
        topic_nlp_id = \
        requests.get(f"{base_url}/topics", params={"search": "Natural Language Processing"}).json()["results"][0][
            "id"].split('/')[-1]
        topic_tm_id = \
        requests.get(f"{base_url}/topics", params={"search": "Topic Modeling"}).json()["results"][0]["id"].split('/')[
            -1]
    except Exception as e:
        print("❌ Topic ID 获取失败:", e)
        return

    # ---------------------------------------------------------
    # 历史阶段专属的 Track A (核心词)
    # ---------------------------------------------------------
    # Phase 1 核心词 (2014-2016)
    filter_a_p1 = (
        "publication_year:2014-2016,concepts.id:C41008148,"
        "title_and_abstract.search:\"Generative Adversarial\"|\"Variational Autoencoder\"|\"Deep Generative\"|"
        "\"DCGAN\"|\"LAPGAN\"|\"InfoGAN\"|\"BiGAN\"|\"CoGAN\"|\"Normalizing Flow\"|\"Real NVP\"|"
        "\"Sequence to Sequence\"|\"Seq2Seq\"|\"Neural Language Model\"|\"Text Generation\"|\"Neural Machine Translation\""
    )

    # Phase 2 核心词 (2017-2021)
    filter_a_p2 = (
        "publication_year:2017-2021,concepts.id:C41008148,"
        "title_and_abstract.search:\"Transformer\"|\"Transformers\"|\"Pre-trained Language\"|\"Generative Pre-trained\"|"
        "\"GPT\"|\"BERT\"|\"Large Language Model\"|\"Diffusion model\"|\"Denoising diffusion\"|\"VQ-VAE\"|\"VQGAN\"|"
        "\"Generative Adversarial\"|\"StyleGAN\"|\"CycleGAN\"|\"Image Synthesis\"|\"Text Generation\"|\"Seq2Seq\""
    )

    # ---------------------------------------------------------
    # 统一升级后的 Track B (视觉) 和 Track C (文本双聚类联合)
    # ---------------------------------------------------------
    def build_track_b(years):
        return f"publication_year:{years},topics.id:{topic_cv_id}"

    def build_track_c(years):
        # ⚠️ 关键升级：NLP 和 Topic Modeling 联手提纯！
        return (f"publication_year:{years},topics.id:{topic_nlp_id}|{topic_tm_id},"
                "title_and_abstract.search:\"generative\"|\"generation\"|\"decoder\"|\"language model\"|\"seq2seq\"")

    def get_count(filt):
        res = requests.get(f"{base_url}/works", params={"filter": filt, "mailto": MY_EMAIL}).json()
        return res.get('meta', {}).get('count', 0)

    # ---------------------------------------------------------
    # 测算 Phase 1
    # ---------------------------------------------------------
    print("🔍 正在测算 Phase 1 (2014-2016)...")
    p1_a = get_count(filter_a_p1)
    p1_b = get_count(build_track_b("2014-2016"))
    p1_c = get_count(build_track_c("2014-2016"))
    p1_total = p1_a + p1_b + p1_c

    # ---------------------------------------------------------
    # 测算 Phase 2
    # ---------------------------------------------------------
    print("🔍 正在测算 Phase 2 (2017-2021)...")
    p2_a = get_count(filter_a_p2)
    p2_b = get_count(build_track_b("2017-2021"))
    p2_c = get_count(build_track_c("2017-2021"))
    p2_total = p2_a + p2_b + p2_c

    print("\n================================================================")
    print(" ⚖️ 历史数据口径统一 (补漏) 测试战报")
    print("================================================================")
    print("【第一阶段 Phase 1: 2014-2016】")
    print(f" 之前的记录大约是: 8,954 篇 (去重后)")
    print(f" 新口径三轨毛量为: {p1_total:>8,.0f} 篇 (去重前)")
    print("-" * 64)
    print("【第二阶段 Phase 2: 2017-2021】")
    print(f" 之前的记录大约是: 105,347 篇 (去重后)")
    print(f" 新口径三轨毛量为: {p2_total:>8,.0f} 篇 (去重前)")
    print("================================================================\n")
    print("💡 决策依据：把这两个【新口径毛量】发给我！我们来看看增幅到底有多大。")


probe_historical_missing_topics()