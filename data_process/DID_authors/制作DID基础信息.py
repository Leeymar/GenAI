import pandas as pd
import json
import re
import warnings

warnings.filterwarnings('ignore')

# ================= 配置区 =================
FILE_FORTUNE_500 = r'500.csv'
FILE_ACADEMIC_100 = r'清洗后_终极名单.xlsx'  # 引入你刚洗好的名校表
FILE_JSONL = r'Author_Global_Metadata_90k.jsonl'
FILE_MATCHED_CSV = r'Matched_Author_IDs_Ultra_Strict.csv'
OUTPUT_FINAL_AUTHOR_META = r'Final_Author_Metadata_for_DID.csv'

COUNTRY_MAP = {
    '美国': 'US', '中国': 'CN', '英国': 'GB', '德国': 'DE', '法国': 'FR',
    '日本': 'JP', '韩国': 'KR', '印度': 'IN', '加拿大': 'CA', '瑞士': 'CH',
    '荷兰': 'NL', '沙特阿拉伯': 'SA', '意大利': 'IT', '西班牙': 'ES',
    '澳大利亚': 'AU', '瑞典': 'SE', '芬兰': 'FI', '新加坡': 'SG'
}


# ==========================================

def load_fortune_500_with_country(filepath):
    print("🏢 正在加载《财富》世界500强名单及国家信息...")
    fortune_patterns = []
    df_500 = None

    encodings_to_try = ['gb18030', 'gbk', 'utf-8', 'ansi']
    for enc in encodings_to_try:
        try:
            df_500 = pd.read_csv(filepath, header=None, names=['en_name', 'country'], encoding=enc)
            break
        except Exception:
            continue

    if df_500 is not None:
        for _, row in df_500.iterrows():
            en_name = str(row['en_name']).strip()
            cn_country = str(row['country']).strip()

            if en_name.lower() != 'nan' and len(en_name) > 2:
                # 工业界大多是全称，继续用严格边界
                pattern = re.compile(r'\b' + re.escape(en_name) + r'\b', re.IGNORECASE)
                country_code = COUNTRY_MAP.get(cn_country, cn_country)
                fortune_patterns.append((pattern, country_code))

    return fortune_patterns


def load_academic_elites(filepath):
    print("🏛️ 正在加载《终极名校名单》...")
    aca_patterns = []
    try:
        df_aca = pd.read_excel(filepath)
        for _, row in df_aca.iterrows():
            univ = str(row['University']).strip()
            country = str(row['Country']).strip()

            if univ.lower() != 'nan' and len(univ) > 1:
                safe_univ = re.escape(univ)
                # 【核心智能匹配逻辑】
                if len(univ) <= 5:
                    # 短名称（如 MIT, UCL）：严格单词边界，防误伤
                    pattern = re.compile(r'\b' + safe_univ + r'\b', re.IGNORECASE)
                else:
                    # 长名称：只要字符串包含了名校名字就算！（解决 Chinese Academy of Sciences 问题）
                    pattern = re.compile(safe_univ, re.IGNORECASE)

                country_code = COUNTRY_MAP.get(country, country)
                aca_patterns.append((pattern, country_code))

        print(f"✅ 成功加载 {len(aca_patterns)} 所名校及国家归属！")
        return aca_patterns
    except Exception as e:
        print(f"❌ 加载名校数据时出错: {e}")
        return []


def process_and_label_matched_authors():
    # 1. 加载 500 强
    fortune_patterns = load_fortune_500_with_country(FILE_FORTUNE_500)

    # 2. 预定义 AI 巨头补丁
    tech_giants = {
        r'\bgoogle\b': 'US', r'\bdeepmind\b': 'GB', r'\bmicrosoft\b': 'US',
        r'\bmeta\b': 'US', r'\bfacebook\b': 'US', r'\bopenai\b': 'US',
        r'\banthropic\b': 'US', r'\bapple\b': 'US', r'\bamazon\b': 'US', r'\baws\b': 'US',
        r'\btencent\b': 'CN', r'\balibaba\b': 'CN', r'\bbaidu\b': 'CN',
        r'\bbytedance\b': 'CN', r'\bhuawei\b': 'CN', r'\bnvidia\b': 'US', r'\bibm\b': 'US'
    }
    tech_giant_patterns = [(re.compile(k, re.IGNORECASE), v) for k, v in tech_giants.items()]
    all_industry_patterns = fortune_patterns + tech_giant_patterns

    # 3. 加载你的动态名校表
    academic_patterns = load_academic_elites(FILE_ACADEMIC_100)

    # 【名校超级补丁包】：专门解决加州系统、德州系统和特定机构
    academic_patches = [
        (re.compile(r'University of California', re.IGNORECASE), 'US'),  # 统杀 Berkeley, LA 等所有 UC 分校
        (re.compile(r'University of Texas', re.IGNORECASE), 'US'),  # 统杀 Austin, MD Anderson 等所有 UT 分校
        (re.compile(r'State University of New York', re.IGNORECASE), 'US'),  # 统杀所有纽约州立系统
        (re.compile(r'Max Planck', re.IGNORECASE), 'DE'),  # 马普所的各种分支
        (re.compile(r'CNRS', re.IGNORECASE), 'FR')  # 法国国家科研中心
    ]
    all_academic_patterns = academic_patterns + academic_patches

    # 4. 解析 JSONL
    print("\n⏳ 正在从 JSONL 解析全量作者元数据...")
    records = []
    with open(FILE_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            records.append({
                'author_id': data.get('author_id'),
                'institution': data.get('institution', 'Unknown'),
                'country_code': data.get('country_code', 'Unknown')
            })
    df_meta = pd.DataFrame(records)

    # 5. 合并数据
    print("🤝 正在与严格匹配名单合并...")
    df_matched = pd.read_csv(FILE_MATCHED_CSV)
    df_final = df_matched.merge(df_meta, on='author_id', how='left')
    df_final['institution'] = df_final['institution'].fillna('Unknown').astype(str)

    # 6. 【核心模块】：打标与双重地域抢救
    print("🚀 启动终极打标与地域信息挽救引擎...")

    def evaluate_author(row):
        inst = row['institution']
        curr_country = row['country_code']
        is_ind, is_aca = 0, 0

        # 判断工业界 (并尝试抢救国家)
        for pattern, country in all_industry_patterns:
            if pattern.search(inst):
                is_ind = 1
                if curr_country == 'Unknown' or pd.isna(curr_country):
                    curr_country = country
                break

        # 判断学术界 (并尝试抢救国家)
        for pattern, country in all_academic_patterns:
            if pattern.search(inst):
                is_aca = 1
                if curr_country == 'Unknown' or pd.isna(curr_country):
                    curr_country = country
                break

        return pd.Series([is_ind, is_aca, curr_country])

    # 应用函数
    df_final[['is_industry_elite', 'is_academic_elite', 'country_code']] = df_final.apply(evaluate_author, axis=1)
    df_final['is_elite_overall'] = (
                df_final['is_industry_elite'].fillna(0).astype(int) | df_final['is_academic_elite'].fillna(0).astype(
            int))

    # 7. 统计
    total = len(df_final)
    elite_ind = df_final['is_industry_elite'].sum()
    elite_aca = df_final['is_academic_elite'].sum()
    elite_total = df_final['is_elite_overall'].sum()
    original_countries = df_meta.set_index('author_id').loc[df_final['author_id']]['country_code'].values
    rescued = (df_final['country_code'] != original_countries).sum()

    print("\n=========================================================")
    print("🎯 DID 最终静态属性库构建完成！统计结果：")
    print("=========================================================")
    print(f"匹配存活总作者: {total:,} 人")
    print(f"🏭 工业巨头/500强作者: {elite_ind:,} 人 ({elite_ind / total * 100:.1f}%)")
    print(f"🏛️ Top100 名校作者: {elite_aca:,} 人 ({elite_aca / total * 100:.1f}%)")
    print(f"✨ 拥有【精英光环】总人数: {elite_total:,} 人 ({elite_total / total * 100:.1f}%)")
    print(f"🚑 成功挽救缺失国家信息的作者: {rescued:,} 人")
    print("=========================================================")

    df_final.to_csv(OUTPUT_FINAL_AUTHOR_META, index=False, encoding='utf-8-sig')
    print(f"💾 附带静态变量及修正国家的最终名单已保存至: {OUTPUT_FINAL_AUTHOR_META}")


if __name__ == "__main__":
    process_and_label_matched_authors()