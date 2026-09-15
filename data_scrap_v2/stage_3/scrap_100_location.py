import json
import random
import pandas as pd
import requests
import time


def sample_and_fetch_locations_from_api():
    file_path = 'phase3_hybrid_cleaned.jsonl'
    sample_size = 100
    sampled_lines = []

    print(f"🎲 启动【云端溯源探针】：正在从本地随机抽取 {sample_size} 篇文献 ID...")

    # 1. 蓄水池抽样算法 (只抽 ID)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if not line.strip(): continue
                if len(sampled_lines) < sample_size:
                    sampled_lines.append(line)
                else:
                    j = random.randint(0, i)
                    if j < sample_size:
                        sampled_lines[j] = line
    except FileNotFoundError:
        print(f"❌ 找不到文件: {file_path}")
        return

    print("☁️ 抽样完成！正在向 OpenAlex 云端请求这 100 篇文献的真实发表阵地...")

    results = []
    session = requests.Session()

    # 2. 遍历抽出的 100 条记录，去云端查 Location
    for idx, line in enumerate(sampled_lines, 1):
        try:
            work_local = json.loads(line)
            work_id_url = work_local.get('work_id')
            if not work_id_url: continue

            # 提取纯粹的 ID (例如 W4320013936)
            short_id = work_id_url.split('/')[-1]
            api_url = f"https://api.openalex.org/works/{short_id}"

            # 向云端发起请求
            res = session.get(api_url, timeout=10)
            if res.status_code == 200:
                cloud_data = res.json()

                title = cloud_data.get('title') or 'Unknown Title'
                short_title = title[:50] + '...' if len(title) > 50 else title

                # 从云端数据中扒出发表位置
                primary_loc = cloud_data.get('primary_location') or {}
                source = primary_loc.get('source') or {}

                venue_name = source.get('display_name', 'Unknown Venue')
                venue_type = source.get('type', 'Unknown Type')
                publisher = source.get('host_organization_name', 'Unknown Publisher')

                results.append({
                    'Work_ID': short_id,
                    'Paper Title': short_title,
                    'Venue Name': venue_name,
                    'Venue Type': venue_type,
                    'Publisher': publisher
                })

            # 加上一点微小的延迟，防止被 API 踢下线
            time.sleep(0.1)

            # 打印进度条
            if idx % 20 == 0:
                print(f"   ➤ 已溯源 {idx}/100 篇...")

        except Exception as e:
            pass

    # 3. 数据分析与展示 (去掉了引发报错的 to_markdown)
    df_results = pd.DataFrame(results)

    print("\n================================================================")
    print(" 🎯 Phase 3 (爆发期) 随机 100 篇文献【云端真实阵地】战报")
    print("================================================================")

    # 统计这 100 篇里最常出现的阵地
    top_venues = df_results['Venue Name'].value_counts().head(5)
    print("🏆 这 100 篇中出现频率最高的 Top 5 阵地：")
    for venue, count in top_venues.items():
        print(f"   ➤ {str(venue)[:50]:<50} : {count} 篇")

    print("\n----------------------------------------------------------------")
    print("📄 随机截取前 10 篇文献的溯源结果：")

    # 直接使用 Pandas 的原生打印，安全可靠
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df_results.head(10).to_string(index=False))
    print("================================================================\n")

    # 保存结果
    output_csv = 'Sampled_100_Cloud_Locations_Phase3.csv'
    df_results.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"💾 完整的 100 条溯源档案已保存至: {output_csv}")


# 启动引擎！
sample_and_fetch_locations_from_api()