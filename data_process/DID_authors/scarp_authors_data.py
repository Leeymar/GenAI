import json
import asyncio
import httpx
import os
import time
from tqdm.asyncio import tqdm

# ================= 配置区 =================
EMAIL = "2212408015@stmail.ujs.edu.cn"  # 填你的邮箱
API_KEY = "teHSRQPDMAveyma48Q0UTA"  # 【必填】你的 Premium API Key
INPUT_FILE = r'All_96k_Authors_Basic.jsonl'
OUTPUT_FILE = r'Author_Global_Metadata_90k.jsonl'

# 提速配置：高级密钥支持高并发，设为每秒 50 次请求
RATE_LIMIT = 50.0
BATCH_SIZE = 2000  # 批次也可以稍微调大一点


# ==========================================

class TokenBucketLimiter:
    """严格的令牌桶限流器"""

    def __init__(self, rate: float):
        self.rate = rate
        self.tokens = rate
        self.last_update = time.monotonic()
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            while True:
                now = time.monotonic()
                elapsed = now - self.last_update
                self.tokens = min(self.rate, self.tokens + elapsed * self.rate)
                self.last_update = now

                if self.tokens >= 1:
                    self.tokens -= 1
                    return

                wait_time = (1 - self.tokens) / self.rate
                await asyncio.sleep(wait_time)


async def fetch_author(client, author_info, limiter):
    await limiter.acquire()

    author_id = author_info['author_id'].split('/')[-1]
    url = f"https://api.openalex.org/authors/{author_id}"

    headers = {"User-Agent": f"mailto:{EMAIL}"}
    if API_KEY:
        # 高级密钥建议放在 Authorization 里面，或者作为 api_key 参数
        headers["Authorization"] = f"Bearer {API_KEY}"

    # 加入简单的重试机制，防止极个别请求失败导致丢数据
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = await client.get(url, headers=headers, timeout=20.0)

            # 如果触发了 429 报错，稍微睡一会儿再重试
            if response.status_code == 429:
                await asyncio.sleep(2 ** attempt)
                continue

            if response.status_code == 200:
                data = response.json()
                counts_by_year = data.get('counts_by_year', [])
                years_active = [c['year'] for c in counts_by_year]
                first_pub_year = min(years_active) if years_active else None

                last_inst = data.get('last_known_institutions', [{}])
                last_inst_dict = last_inst[0] if last_inst else {}

                return {
                    "author_id": author_info['author_id'],
                    "is_treated": author_info.get('is_treated'),
                    "event_year": author_info.get('event_year'),
                    "display_name": data.get('display_name'),
                    "institution": last_inst_dict.get('display_name', 'Unknown'),
                    "country_code": last_inst_dict.get('country_code', 'Unknown'),
                    "h_index": data.get('summary_stats', {}).get('h_index', 0),
                    "first_pub_year": first_pub_year,
                    "counts_by_year": counts_by_year
                }
            else:
                return None
        except Exception:
            if attempt == max_retries - 1:
                return None
            await asyncio.sleep(1)
    return None


def dynamic_slice_generator(filepath, processed_ids, batch_size):
    batch = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                author_info = json.loads(line)
                if author_info['author_id'] not in processed_ids:
                    batch.append(author_info)
                    if len(batch) == batch_size:
                        yield batch
                        batch = []
            except json.JSONDecodeError:
                continue
    if batch:
        yield batch


async def main():
    processed_ids = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    processed_ids.add(json.loads(line)['author_id'])
                except:
                    pass
    print(f"🔄 检测到已成功抓取 {len(processed_ids)} 人，准备跳过...")

    total_to_fetch = sum(1 for line in open(INPUT_FILE, 'r', encoding='utf-8')
                         if json.loads(line)['author_id'] not in processed_ids)
    print(f"🚀 本次实际需要向 API 发起请求: {total_to_fetch} 人")

    if total_to_fetch == 0:
        return

    limiter = TokenBucketLimiter(rate=RATE_LIMIT)

    # 【提速关键】放宽连接池大小，允许最多 100 个 TCP 握手连接
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=100, max_keepalive_connections=50)) as client:
        generator = dynamic_slice_generator(INPUT_FILE, processed_ids, BATCH_SIZE)

        # 把打开文件的操作提到最外面，避免反复打开关闭
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f_out:
            for batch_index, batch in enumerate(generator):
                tasks = [fetch_author(client, a, limiter) for a in batch]

                # 【核心改动】使用 as_completed，谁先完成就先存谁，绝不等待！
                for coro in tqdm.as_completed(tasks, desc=f"抓取批次 {batch_index + 1}"):
                    res = await coro
                    if res:
                        f_out.write(json.dumps(res, ensure_ascii=False) + '\n')
                        f_out.flush()  # 【双保险】强制立刻刷入硬盘，绝不停留在内存缓冲区

                # 缩短休眠时间，保持高效推进
                await asyncio.sleep(0.1)

    print("\n✅ 全量全时空特征采集完成！")


if __name__ == "__main__":
    import sys

    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())