import pandas as pd
import json
import os

# ================= 配置区域 (Windows 路径) =================
# 1. 你的 DID 面板数据 (截图中的那个文件)
CSV_PATH = r"DID_Panel_Ready.csv"

# 2. 包含机构和国家信息的 JSONL 字典文件
JSONL_PATH = r"E:\PythonProject\GenAI\data_process\DID_authors\Author_Global_Metadata_90k.jsonl"

# 3. 输出文件 1：拼接成功后的新 DID 面板数据
OUTPUT_CSV = r"DID_Panel_Ready_with_Meta.csv"

# 4. 输出文件 2：需要重新去 OpenAlex 获取的缺失作者名单
MISSING_LIST_PATH = r"Missing_Authors_for_OpenAlex.txt"


# =========================================================

def match_author_metadata():
    print("🚀 启动【作者机构与国家信息匹配引擎】...")

    # --- 步骤 1：解析 JSONL，建立哈希字典 ---
    print(f"   📂 1. 正在加载元数据字典: {JSONL_PATH}")
    meta_dict = {}

    if os.path.exists(JSONL_PATH):
        with open(JSONL_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    data = json.loads(line.strip())
                    # 清洗 ID：把 "https://openalex.org/A5052435039" 变成 "A5052435039"
                    raw_id = data.get("author_id", "")
                    clean_id = str(raw_id).split("/")[-1]

                    if clean_id:
                        meta_dict[clean_id] = {
                            "institution": data.get("institution") or "",
                            "country_code": data.get("country_code") or ""
                        }
                except json.JSONDecodeError:
                    continue
        print(f"      ✅ 成功提取了 {len(meta_dict):,} 位作者的元数据。")
    else:
        print(f"      ❌ 找不到 JSONL 文件，请检查路径。")
        return

    # --- 步骤 2：加载 CSV 并比对 ---
    print(f"   📊 2. 正在加载 DID 面板数据: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    unique_authors = df['author_id'].unique()
    print(f"      🔍 CSV 中共有 {len(unique_authors):,} 位独立作者需要匹配。")

    # 找出缺失的作者：
    # 1. 压根不在 JSONL 字典里
    # 2. 在 JSONL 里，但是 institution 和 country_code 全是空的
    missing_authors = set()
    for aid in unique_authors:
        if aid not in meta_dict:
            missing_authors.add(aid)
        else:
            if not meta_dict[aid]["institution"] and not meta_dict[aid]["country_code"]:
                missing_authors.add(aid)

    print(f"      ⚠️ 发现 {len(missing_authors):,} 位作者数据缺失，将写入补全名单。")

    # --- 步骤 3：拼接数据并导出 ---
    print("   💾 3. 正在将匹配到的数据并入 DataFrame...")

    # 映射 institution 和 country_code 到新的列
    df['institution'] = df['author_id'].map(lambda x: meta_dict[x]['institution'] if x in meta_dict else None)
    df['country_code'] = df['author_id'].map(lambda x: meta_dict[x]['country_code'] if x in meta_dict else None)

    # 导出包含新列的 CSV
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"      ✅ 增强版面板数据已保存至: {OUTPUT_CSV}")

    # 导出缺失名单
    with open(MISSING_LIST_PATH, 'w', encoding='utf-8') as f:
        for aid in missing_authors:
            # 还原为 OpenAlex 长链接格式方便你爬虫，如果你只需要短 ID，删掉前面这段前缀即可
            f.write(f"https://openalex.org/{aid}\n")

    print(f"      ✅ 缺失名单已保存至: {MISSING_LIST_PATH}")
    print("\n🎉 匹配任务全部完成！")


if __name__ == '__main__':
    match_author_metadata()