import pandas as pd
import re
import warnings

warnings.filterwarnings('ignore')

# ================= 配置区 =================
FILE_FORTUNE_500 = r'500.csv'
FILE_ACADEMIC_100 = r'清洗后_终极名单.xlsx'
FILE_DID_PANEL = r'DID_Panel_Ready_Final.csv'  # 你的原始面板数据
OUTPUT_ULTIMATE_PANEL = r'Ultimate_DID_Panel.csv'  # 最终生成的超级面板

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
                if len(univ) <= 5:
                    pattern = re.compile(r'\b' + safe_univ + r'\b', re.IGNORECASE)
                else:
                    pattern = re.compile(safe_univ, re.IGNORECASE)

                country_code = COUNTRY_MAP.get(country, country)
                aca_patterns.append((pattern, country_code))

        print(f"✅ 成功加载 {len(aca_patterns)} 所名校及国家归属！")
        return aca_patterns
    except Exception as e:
        print(f"❌ 加载名校数据时出错: {e}")
        return []


def enrich_did_panel():
    # 1. 准备所有匹配规则
    fortune_patterns = load_fortune_500_with_country(FILE_FORTUNE_500)

    tech_giants = {
        r'\bgoogle\b': 'US', r'\bdeepmind\b': 'GB', r'\bmicrosoft\b': 'US',
        r'\bmeta\b': 'US', r'\bfacebook\b': 'US', r'\bopenai\b': 'US',
        r'\banthropic\b': 'US', r'\bapple\b': 'US', r'\bamazon\b': 'US', r'\baws\b': 'US',
        r'\btencent\b': 'CN', r'\balibaba\b': 'CN', r'\bbaidu\b': 'CN',
        r'\bbytedance\b': 'CN', r'\bhuawei\b': 'CN', r'\bnvidia\b': 'US', r'\bibm\b': 'US'
    }
    tech_giant_patterns = [(re.compile(k, re.IGNORECASE), v) for k, v in tech_giants.items()]
    all_industry_patterns = fortune_patterns + tech_giant_patterns

    academic_patterns = load_academic_elites(FILE_ACADEMIC_100)
    academic_patches = [
        (re.compile(r'University of California', re.IGNORECASE), 'US'),
        (re.compile(r'University of Texas', re.IGNORECASE), 'US'),
        (re.compile(r'State University of New York', re.IGNORECASE), 'US'),
        (re.compile(r'Max Planck', re.IGNORECASE), 'DE'),
        (re.compile(r'CNRS', re.IGNORECASE), 'FR')
    ]
    all_academic_patterns = academic_patterns + academic_patches

    # 2. 读取面板数据
    print("\n⏳ 正在读取 DID 面板数据...")
    df_panel = pd.read_csv(FILE_DID_PANEL)

    # 3. 【性能优化】提取唯一的作者列表
    print("🤝 正在提取去重后的唯一作者名单进行极速打标...")
    # 这里我们假定同一个人 institution 和 country_code 是稳定的，直接取第一条即可
    df_authors = df_panel[['author_id', 'institution', 'country_code']].drop_duplicates(subset=['author_id']).copy()
    df_authors['institution'] = df_authors['institution'].fillna('Unknown').astype(str)

    def evaluate_author(row):
        inst = row['institution']
        curr_country = row['country_code']
        is_ind, is_aca = 0, 0

        # 判断工业界
        for pattern, country in all_industry_patterns:
            if pattern.search(inst):
                is_ind = 1
                if curr_country == 'Unknown' or pd.isna(curr_country):
                    curr_country = country
                break

        # 判断学术界
        for pattern, country in all_academic_patterns:
            if pattern.search(inst):
                is_aca = 1
                if curr_country == 'Unknown' or pd.isna(curr_country):
                    curr_country = country
                break

        return pd.Series([is_ind, is_aca, curr_country])

    # 4. 对唯一作者进行打标
    df_authors[['is_industry_elite', 'is_academic_elite', 'updated_country_code']] = df_authors.apply(evaluate_author,
                                                                                                      axis=1)

    # 5. 映射回原始面板数据
    print("🚀 正在将精英标签与挽救后的地区数据映射回完整面板...")
    # 为了避免重名列，先丢弃面板中旧的 country_code
    df_panel = df_panel.drop(columns=['country_code'])

    # 将包含新标签的字典表合并回去
    df_final = df_panel.merge(
        df_authors[['author_id', 'is_industry_elite', 'is_academic_elite', 'updated_country_code']],
        on='author_id',
        how='left'
    )

    # 整理列名
    df_final.rename(columns={'updated_country_code': 'country_code'}, inplace=True)

    # 生成综合精英标志
    df_final['is_elite_overall'] = (
                df_final['is_industry_elite'].fillna(0).astype(int) | df_final['is_academic_elite'].fillna(0).astype(
            int))

    # 6. 统计打印
    total_authors = len(df_authors)
    elite_ind = df_authors['is_industry_elite'].sum()
    elite_aca = df_authors['is_academic_elite'].sum()
    # 先填充空值，强制转为整型，然后再执行位运算 OR
    elite_total = (df_authors['is_industry_elite'].fillna(0).astype(int) | df_authors['is_academic_elite'].fillna(
        0).astype(int)).sum()

    # 计算被挽救的作者数
    rescued = (df_authors['country_code'] != df_authors['updated_country_code']).sum()

    print("\n=========================================================")
    print("🎯 面板数据终极构建完成！基于【唯一作者维度】的统计：")
    print("=========================================================")
    print(f"总计覆盖作者: {total_authors:,} 人")
    print(f"🏭 工业巨头/500强作者: {elite_ind:,} 人 ({elite_ind / total_authors * 100:.1f}%)")
    print(f"🏛️ Top100 名校作者: {elite_aca:,} 人 ({elite_aca / total_authors * 100:.1f}%)")
    print(f"✨ 拥有【精英光环】总人数: {elite_total:,} 人 ({elite_total / total_authors * 100:.1f}%)")
    print(f"🚑 成功挽救缺失国家信息的作者: {rescued:,} 人")
    print("=========================================================")
    print(f"📊 最终面板数据总行数: {len(df_final):,} 行")

    # 7. 保存文件
    df_final.to_csv(OUTPUT_ULTIMATE_PANEL, index=False, encoding='utf-8-sig')
    print(f"💾 大功告成！完美附带异质性标签的超级面板已保存至: {OUTPUT_ULTIMATE_PANEL}")


if __name__ == "__main__":
    enrich_did_panel()