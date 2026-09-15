import requests


def probe_dataset_counts():
    print("🛰️ 启动 OpenAlex 极速探针 (仅查询数量，不下载数据)...\n")
    base_url = "https://api.openalex.org/works"

    # ⚠️ 填入你的邮箱
    headers = {"mailto": "leeymar10@gmail.com"}

    # 1. 抓取必备的 Topic IDs
    print("🔍 正在解析底层 Topic 标签...")
    topic_cv_id = requests.get("https://api.openalex.org/topics",
                               params={"search": "Generative Adversarial Networks and Image Synthesis"}).json()["results"][0]["id"].split(
        '/')[-1]
    topic_nlp_id = requests.get("https://api.openalex.org/topics",
                                params={"search": "Natural Language Processing Techniques"}).json()["results"][0]["id"].split('/')[
        -1]

    core_concepts = "C41008148|C204321447|C154945302|C119857082"

    def get_count(filters, name):
        # 核心秘诀：per-page=1 且只要 id，极其节省网络带宽，瞬间返回！
        params = {
            "filter": filters,
            "per-page": 1,
            "select": "id"
        }
        try:
            res = requests.get(base_url, params=params, headers=headers)
            res.raise_for_status()
            count = res.json().get('meta', {}).get('count', 0)
            print(f"   ➤ {name:<35}: {count:>8,} 篇")
            return count
        except Exception as e:
            print(f"   ❌ 查询 {name} 失败: {e}")
            return 0

    # ==========================================
    # 📊 第一阶段 (2014-2016) 预估盘点
    # ==========================================
    print("\n" + "=" * 50)
    print(" 🏆 第一阶段 (2014-2016) 拓荒期探针报告")
    print("=" * 50)

    v1_keys = "\"generative adversarial\"|\"variational autoencoder\"|\"generative model\"|\"deep generative\"|\"image synthesis\"|\"text-to-image\"|\"dcgan\"|\"infogan\""
    n1_keys = "\"language model\"|\"sequence to sequence\"|\"seq2seq\"|\"neural machine translation\"|\"text generation\"|\"neural language model\"|\"autoregressive language model\""

    f1_a = f"publication_year:2014-2016,concepts.id:{core_concepts},title_and_abstract.search:{v1_keys}"
    f1_b = f"publication_year:2014-2016,concepts.id:{core_concepts},title_and_abstract.search:{n1_keys}"
    f1_c = f"publication_year:2014-2016,topics.id:{topic_cv_id}|{topic_nlp_id}"

    c1_a = get_count(f1_a, "[轨A] 视觉与基础生成 (GAN/VAE)")
    c1_b = get_count(f1_b, "[轨B] 语言生成先驱 (Seq2Seq等)")
    c1_c = get_count(f1_c, "[轨C] 官方聚类兜底 (Topic)")

    # 粗略估算去重后的总数（假设三轨重合度在 20% 左右）
    estimated_total_1 = int((c1_a + c1_b + c1_c) * 0.8)
    print("-" * 50)
    print(f"💡 [Phase 1] 预计去重后总文献量约为: ~{estimated_total_1:,} 篇")

    # ==========================================
    # 📊 第二阶段 (2017-2021) 预估盘点
    # ==========================================
    print("\n" + "=" * 50)
    print(" 🏆 第二阶段 (2017-2021) 大模型前夜探针报告")
    print("=" * 50)

    v2_keys = "\"generative adversarial\"|\"variational autoencoder\"|\"diffusion model\"|\"denoising diffusion\"|\"image synthesis\"|\"text-to-image\"|\"stylegan\"|\"vqgan\""
    n2_keys = "\"transformer\"|\"transformers\"|\"pretrained language model\"|\"pre-trained language\"|\"generative pre-trained\"|\"gpt\"|\"bert\"|\"large language model\"|\"text generation\"|\"seq2seq\""

    f2_a = f"publication_year:2017-2021,concepts.id:{core_concepts},title_and_abstract.search:{v2_keys}"
    f2_b = f"publication_year:2017-2021,concepts.id:{core_concepts},title_and_abstract.search:{n2_keys}"
    f2_c = f"publication_year:2017-2021,topics.id:{topic_cv_id}|{topic_nlp_id}"

    c2_a = get_count(f2_a, "[轨A] 计算机视觉与生成 (GAN/Diffusion)")
    c2_b = get_count(f2_b, "[轨B] NLP与大模型主线 (Transformer/GPT)")
    c2_c = get_count(f2_c, "[轨C] 官方聚类兜底 (Topic)")

    # 第二阶段领域大融合，重合度可能会高一点，我们按 75% 算有效率
    estimated_total_2 = int((c2_a + c2_b + c2_c) * 0.75)
    print("-" * 50)
    print(f"💡 [Phase 2] 预计去重后总文献量约为: ~{estimated_total_2:,} 篇\n")


# 直接执行！
probe_dataset_counts()