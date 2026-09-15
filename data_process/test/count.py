import requests
import xml.etree.ElementTree as ET


def test_cloud_arxiv_extraction():
    # 你的目标文献
    TARGET_WORK_ID = "W4404345312"

    print(f"🎯 启动【云端双穿透狙击】：目标 {TARGET_WORK_ID}\n")

    # ==========================================
    # 第一步：向 OpenAlex 索要 arXiv 身份证号
    # ==========================================
    print(f"📡 [1/2] 正在连接 OpenAlex 官方，查询此人的 arXiv 档案...")
    oa_url = f"https://api.openalex.org/works/{TARGET_WORK_ID}"

    try:
        oa_response = requests.get(oa_url, timeout=10)
        oa_response.raise_for_status()
        oa_data = oa_response.json()

        # 提取 arXiv 链接
        ids_dict = oa_data.get('ids', {})
        arxiv_url = ids_dict.get('arxiv')

        if not arxiv_url:
            print(f"❌ 失败：OpenAlex 官方数据库里也没有这篇论文的 arXiv 链接。它可能不是预印本！")
            return

        # 清洗出纯净的 arXiv ID
        arxiv_id = str(arxiv_url).split('/abs/')[-1].split('v')[0]
        print(f"🔑 破译成功！拿到真实的 arXiv 身份证号: {arxiv_id}")

    except Exception as e:
        print(f"❌ OpenAlex API 请求失败: {e}")
        return

    print("=" * 60)

    # ==========================================
    # 第二步：拿着身份证号去 arXiv 官方拿基因标签
    # ==========================================
    print(f"🌐 [2/2] 正在突入 arXiv 官方服务器，读取原始跨界填报记录...\n")
    arxiv_api_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"

    try:
        arxiv_response = requests.get(arxiv_api_url, timeout=10)
        arxiv_response.raise_for_status()

        root = ET.fromstring(arxiv_response.text)
        ns = {'atom': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}

        for entry in root.findall('atom:entry', ns):
            title = entry.find('atom:title', ns).text.replace('\n', ' ').strip()

            # 🌟 提取标签
            primary_cat_element = entry.find('arxiv:primary_category', ns)
            primary_cat = primary_cat_element.attrib['term'] if primary_cat_element is not None else "Unknown"

            all_cats = [cat.attrib['term'] for cat in entry.findall('atom:category', ns)]

            print("🏆 【云端获取：arXiv 原生基因档案】")
            print("-" * 60)
            print(f"📄 论文标题: {title}")
            print(f"📌 主干分类 (Primary): {primary_cat}")
            print(f"🏷️ 全部交叉分类 (All):   {' | '.join(all_cats)}")
            print("-" * 60)

            if len(all_cats) > 1:
                print("💡 洞察：这篇文章贴了不止一个标签！作者在上传时就具有明确的跨学科动机！")
            else:
                print("💡 洞察：这篇文章只贴了一个标签，是一次极度垂直的纯技术探讨。")

    except Exception as e:
        print(f"❌ arXiv API 请求失败: {e}")


if __name__ == "__main__":
    test_cloud_arxiv_extraction()