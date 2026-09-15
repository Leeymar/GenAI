import pandas as pd
import requests
import xml.etree.ElementTree as ET
import time
import os
from tqdm import tqdm

# ==========================================
# ⚙️ 核心配置区
# ==========================================
BATCH_SIZE = 100  # 每次打包 100 个，平衡速度与稳定性
SLEEP_TIME = 3  # 严格遵守官方 3 秒间隔要求


def run_step2_fetch_arxiv():
    print("📡 启动【第二步：arXiv 官方 API 温和批量抓取引擎】...")
    print("💡 提醒：为了保护你的 IP 不被封禁，我们采用单线程 + 3秒间隔策略。\n")

    PHASES = ['phase1', 'phase2', 'phase3']

    for phase in PHASES:
        keys_csv = f"{phase}_arxiv_keys.csv"
        output_csv = f"{phase}_arxiv_categories.csv"

        if not os.path.exists(keys_csv):
            print(f"⚠️ 找不到钥匙文件 {keys_csv}，跳过该阶段。")
            continue

        print("\n" + "█" * 60)
        print(f"🎯 正在处理 {phase} 的原生基因标签...")

        # 1. 加载第一步跑出来的钥匙串
        df_keys = pd.read_csv(keys_csv, dtype=str)
        w_to_arxiv_map = dict(zip(df_keys['arxiv_id'], df_keys['work_id']))
        arxiv_ids = list(w_to_arxiv_map.keys())

        # 2. 断点续传逻辑
        processed_ids = set()
        if os.path.exists(output_csv):
            try:
                done_df = pd.read_csv(output_csv, dtype=str)
                processed_ids = set(done_df['arxiv_id'].tolist())
                print(f"📂 发现历史进度，已跳过 {len(processed_ids)} 篇。")
            except:
                pass

        todo_arxiv_ids = [aid for aid in arxiv_ids if aid not in processed_ids]

        if not todo_arxiv_ids:
            print(f"🎉 {phase} 已经全量抓取完毕！")
            continue

        print(f"📥 剩余需联网抓取: {len(todo_arxiv_ids)} 篇。")

        # 3. 开始批量迭代
        # 如果文件不存在，先写表头
        if not os.path.exists(output_csv):
            pd.DataFrame(
                columns=['work_id', 'arxiv_id', 'primary_category', 'all_categories', 'category_count']).to_csv(
                output_csv, index=False, encoding='utf-8-sig')

        batches = [todo_arxiv_ids[i:i + BATCH_SIZE] for i in range(0, len(todo_arxiv_ids), BATCH_SIZE)]

        with open(output_csv, 'a', encoding='utf-8-sig', newline='') as f_out:
            import csv
            writer = csv.writer(f_out)

            for batch in tqdm(batches, desc=f"{phase} 抓取进度"):
                id_query = ','.join(batch)
                url = f'http://export.arxiv.org/api/query?id_list={id_query}&max_results={BATCH_SIZE}'

                success = False
                retries = 3
                while retries > 0:
                    try:
                        # 加上基础的 User-Agent 伪装
                        headers = {'User-Agent': 'Academic-Research-Bot/1.0 (mailto:your_email@example.com)'}
                        response = requests.get(url, headers=headers, timeout=30)

                        if response.status_code == 200:
                            root = ET.fromstring(response.text)
                            ns = {'atom': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}

                            for entry in root.findall('atom:entry', ns):
                                entry_id_url = entry.find('atom:id', ns).text
                                entry_arxiv_id = entry_id_url.split('/abs/')[-1].split('v')[0]

                                matched_w_id = w_to_arxiv_map.get(entry_arxiv_id, "Unknown")

                                primary_cat_elem = entry.find('arxiv:primary_category', ns)
                                primary_cat = primary_cat_elem.attrib[
                                    'term'] if primary_cat_elem is not None else "Unknown"

                                all_cats = [cat.attrib['term'] for cat in entry.findall('atom:category', ns)]

                                writer.writerow(
                                    [matched_w_id, entry_arxiv_id, primary_cat, "|".join(all_cats), len(all_cats)])

                            f_out.flush()
                            success = True
                            break
                        elif response.status_code == 429:
                            print("🚨 触发 arXiv 限流，正在强制休眠 30 秒...")
                            time.sleep(30)
                            retries -= 1
                        else:
                            retries -= 1
                            time.sleep(5)
                    except Exception:
                        retries -= 1
                        time.sleep(5)

                # 🌟 灵魂核心：无论成功失败，每次请求后必须强制休息 3 秒！
                time.sleep(SLEEP_TIME)

        print(f"✅ {phase} 任务圆满完成！")


if __name__ == "__main__":
    run_step2_fetch_arxiv()