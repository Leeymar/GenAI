import pandas as pd
import json
import os
import numpy as np

# ================= 配置区域 (Windows 路径) =================
# 1. 第一步生成的“半成品” CSV (里面有部分缺失的 institution 和 country_code)
# 如果你没用那个半成品，直接用原始的 DID_Panel_Ready.csv 也可以，代码会自动兼容！
CSV_PATH = r"DID_Panel_Ready_with_Meta.csv"

# 2. 刚才用并发脚本抓回来的新鲜数据
NEW_JSONL_PATH = r"Fetched_Missing_Authors.jsonl"

# 3. 终极输出文件（完美无缺的最终版）
FINAL_OUTPUT_CSV = r"DID_Panel_Ready_Final.csv"


# =========================================================

def patch_missing_data():
    print("🚀 启动【终极数据缝合与补齐引擎】...")

    # --- 步骤 1：加载刚抓回来的增量字典 ---
    print(f"   📂 1. 正在加载最新抓取的增量数据: {NEW_JSONL_PATH}")
    fresh_data = {}
    if os.path.exists(NEW_JSONL_PATH):
        with open(NEW_JSONL_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    data = json.loads(line.strip())
                    aid = data.get("author_id", "")
                    if aid:
                        fresh_data[aid] = {
                            "institution": data.get("institution", ""),
                            "country_code": data.get("country_code", "")
                        }
                except json.JSONDecodeError:
                    continue
        print(f"      ✅ 成功加载 {len(fresh_data):,} 位作者的最新数据。")
    else:
        print(f"      ❌ 找不到增量数据文件，请检查路径。")
        return

    # --- 步骤 2：加载待修补的 CSV ---
    print(f"   📊 2. 正在加载待修补的面版数据: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    # 兼容性处理：如果 CSV 压根就没有这两列，先建出空列
    if 'institution' not in df.columns:
        df['institution'] = np.nan
    if 'country_code' not in df.columns:
        df['country_code'] = np.nan

    # 统计修补前的缺失情况
    missing_inst_before = df['institution'].isna().sum() + (df['institution'] == '').sum()
    print(f"      🔍 修补前，共有 {missing_inst_before:,} 条记录缺乏机构信息。")

    # --- 步骤 3：定义修补逻辑并执行 ---
    print("   💉 3. 正在执行精准无缝注射（填补空缺）...")

    def fill_institution(row):
        # 如果当前行是空的，并且该作者在我们刚抓的数据里，就填进去
        if pd.isna(row['institution']) or str(row['institution']).strip() == '':
            return fresh_data.get(row['author_id'], {}).get('institution', row['institution'])
        return row['institution']

    def fill_country(row):
        if pd.isna(row['country_code']) or str(row['country_code']).strip() == '':
            return fresh_data.get(row['author_id'], {}).get('country_code', row['country_code'])
        return row['country_code']

    # 应用修补函数
    df['institution'] = df.apply(fill_institution, axis=1)
    df['country_code'] = df.apply(fill_country, axis=1)

    # 统计修补后的情况
    missing_inst_after = df['institution'].isna().sum() + (df['institution'] == '').sum()
    fixed_count = missing_inst_before - missing_inst_after

    # --- 步骤 4：保存终极成果 ---
    print(f"   💾 4. 正在保存终极完全体数据...")
    df.to_csv(FINAL_OUTPUT_CSV, index=False, encoding='utf-8-sig')

    print("\n" + "=" * 50)
    print("🎉 数据缝合大功告成！")
    print(f"🛠️ 成功修补了 {fixed_count:,} 条记录的空缺！")
    print(f"🏆 目前仍缺失机构信息的记录数: {missing_inst_after:,} (部分作者可能在 OpenAlex 本身就没有机构记录)")
    print(f"💾 你的终极武器已保存至: {FINAL_OUTPUT_CSV}")
    print("=" * 50)
    print("💡 下一步：你可以直接把这个 Final CSV 扔进 Stata/Python 里跑 DID 回归，或者画那个惊艳的跨国合作大迁徙图了！")


if __name__ == '__main__':
    patch_missing_data()