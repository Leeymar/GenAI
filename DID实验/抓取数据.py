import asyncio
import aiohttp
import json
import time
import os

# ================= 配置区 =================
EMAIL = "2212408015@stmail.ujs.edu.cn"
API_KEY = "Qiz03Bw0vZAm6TTp4VeKfz"

# 提速配置
RATE_LIMIT = 50  # Premium 支持每秒 50 次并发请求
API_BATCH_SIZE = 50  # OpenAlex 单次 filter 允许的最大 ID 数量 (硬性限制50)

# 文件路径 (请根据实际情况微调)
INPUT_FILE = r"Missing_Authors_for_OpenAlex.txt"
OUTPUT_FILE = r"Fetched_Missing_Authors.jsonl"


# ==========================================

async def fetch_batch(session, ids_batch, retries=3):
    """发送单次批量查询请求"""
    filter_str = "|".join(ids_batch)
    # 使用 select 参数仅返回需要的字段，大幅加快传输和解析速度
    url = f"https://api.openalex.org/authors?filter=openalex:{filter_str}&select=id,last_known_institutions,display_name&per-page={API_BATCH_SIZE}"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "User-Agent": f"mailto:{EMAIL}"
    }

    for attempt in range(retries):
        try:
            async with session.get(url, headers=headers, timeout=15) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("results", [])
                elif response.status == 429:
                    # 触发限流，退避重试
                    await asyncio.sleep(2 ** attempt)
                else:
                    print(f"    [警告] 请求失败，状态码: {response.status}")
                    break
        except Exception as e:
            if attempt == retries - 1:
                print(f"    [错误] 批次请求异常: {e}")
            await asyncio.sleep(1)
    return []


async def main():
    print("🚀 启动【OpenAlex Premium 高并发极速拉取引擎】...")

    # --- 1. 读取缺失名单 ---
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到输入文件: {INPUT_FILE}")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        # 清洗出纯净的 A... 格式 ID
        raw_ids = [line.strip().split('/')[-1] for line in f if line.strip()]

    # 去重
    unique_ids = list(set(raw_ids))
    print(f"   📂 1. 成功加载并去重，共需获取 {len(unique_ids):,} 位作者。")

    # --- 2. 切片为 API 允许的每组 50 个 ID ---
    id_batches = [unique_ids[i:i + API_BATCH_SIZE] for i in range(0, len(unique_ids), API_BATCH_SIZE)]
    print(f"   🗂️ 2. 已切分为 {len(id_batches):,} 个 API 请求批次。")

    fetched_count = 0
    start_time = time.time()

    # --- 3. 异步并发请求 ---
    print("   ⚡ 3. 开始执行高并发拉取 (每秒 2500 个 ID)...")

    # 使用 aiohttp 异步会话
    async with aiohttp.ClientSession() as session:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as fout:
            # 按照 RATE_LIMIT (50次请求/秒) 来派发任务
            for i in range(0, len(id_batches), RATE_LIMIT):
                batch_tasks = id_batches[i:i + RATE_LIMIT]

                # 记录这一秒的开始时间
                sec_start = time.time()

                # 瞬间派发这一秒内的所有请求 (最高 50 个)
                tasks = [fetch_batch(session, b) for b in batch_tasks]
                results_list = await asyncio.gather(*tasks)

                # 解析并立刻写入文件
                for results in results_list:
                    for author in results:
                        aid = str(author.get("id", "")).split("/")[-1]
                        if not aid: continue

                        insts = author.get("last_known_institutions", [])
                        institution = insts[0].get("display_name", "") if insts else ""
                        country_code = insts[0].get("country_code", "") if insts else ""

                        record = {
                            "author_id": aid,
                            "display_name": author.get("display_name", ""),
                            "institution": institution,
                            "country_code": country_code
                        }
                        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                        fetched_count += 1

                # 打印进度
                print(
                    f"      [进度] 已发送 {min(i + RATE_LIMIT, len(id_batches))}/{len(id_batches)} 批次 | 已落盘作者: {fetched_count:,}")

                # 严格限流保护：如果处理这批请求的时间不到 1 秒，强制休眠补足 1 秒
                elapsed = time.time() - sec_start
                if elapsed < 1.0:
                    await asyncio.sleep(1.0 - elapsed)

    total_time = time.time() - start_time
    print("\n" + "=" * 50)
    print(f"🎉 极速拉取任务圆满完成！")
    print(f"⏱️ 总耗时: {total_time:.2f} 秒")
    print(f"🎯 成功抓取并落盘: {fetched_count:,} 条作者信息")
    print(f"💾 数据已保存至: {OUTPUT_FILE}")
    print("=" * 50)
    print("💡 接下来，你可以直接读取这个 JSONL 文件，用刚才的方法把它合并回你的 DID 面板数据中。")


if __name__ == '__main__':
    # Windows 平台下防止 asyncio 报错的常见设置
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())