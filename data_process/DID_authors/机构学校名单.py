import pandas as pd
import re
import warnings

warnings.filterwarnings('ignore')


def clean_university_data():
    print("⏳ 正在加载原始数据...")
    file_path = '待处理名单.xlsx'
    try:
        # 没有表头
        df = pd.read_excel(file_path, header=None)
    except FileNotFoundError:
        print(f"❌ 找不到文件：{file_path}，请确认文件名和路径！")
        return

    # 【修正点】第一列是学校(0)，第二列是国家(1)
    try:
        df = df[[0, 1]].copy()
    except KeyError:
        print("❌ 表格格式有误，请确保至少包含两列数据！")
        return

    df.columns = ['University', 'Country']

    # ==========================================
    # 第一步：清洗国家/地区信息
    # ==========================================
    print("🌍 正在标准化国家/地区代码...")
    df['Country'] = df['Country'].fillna('Unknown').astype(str).str.strip()

    # 国家映射字典
    country_map = {
        'United States of America': 'US',
        'United States': 'US',
        'China (Mainland)': 'CN',
        'China': 'CN',
        'United Kingdom': 'GB',
        'Singapore': 'SG',
        'Switzerland': 'CH',
        'Australia': 'AU',
        'Canada': 'CA',
        'Japan': 'JP',
        'South Korea': 'KR',
        'Germany': 'DE',
        'France': 'FR'
    }
    df['Country'] = df['Country'].map(lambda x: country_map.get(x, x))

    # ==========================================
    # 第二步：跨行抢救缺失的国家信息
    # ==========================================
    valid_countries = df[df['Country'] != 'Unknown'].groupby('University')['Country'].first().to_dict()
    df['Country'] = df.apply(lambda row: valid_countries.get(row['University'], row['Country']), axis=1)

    # ==========================================
    # 第三步：核心拆分逻辑（括号与逗号）
    # ==========================================
    print("✂️ 正在拆分括号缩写与逗号截断...")
    cleaned_records = []

    for _, row in df.iterrows():
        orig_name = str(row['University']).strip()
        country = row['Country']

        if orig_name.lower() in ['nan', 'none', '']:
            continue

        names_to_process = []

        # 处理括号：如 Massachusetts Institute of Technology (MIT)
        match = re.search(r'^(.*?)\s*\((.*?)\)(.*)$', orig_name)
        if match:
            outside_text = (match.group(1) + " " + match.group(3)).strip()
            inside_text = match.group(2).strip()
            names_to_process.append(outside_text)
            names_to_process.append(inside_text)
        else:
            names_to_process.append(orig_name)

        # 处理逗号：只保留第一个逗号前面的内容
        for name in names_to_process:
            if ',' in name:
                name = name.split(',')[0].strip()

            if name:
                cleaned_records.append({'University': name, 'Country': country})

    # ==========================================
    # 第四步：去重并保存
    # ==========================================
    print("🧹 正在进行最终去重...")
    final_df = pd.DataFrame(cleaned_records)

    # 排序确保有效国家排在 Unknown 前面，去重时优先保留有国家的记录
    final_df = final_df.sort_values(by=['University', 'Country'])
    final_df = final_df.drop_duplicates(subset=['University'], keep='first').reset_index(drop=True)

    output_file = '清洗后_终极名单.xlsx'
    final_df.to_excel(output_file, index=False)

    print("=========================================")
    print(f"🎉 大功告成！原始数据处理完毕。")
    print(f"💾 最终保留了 {len(final_df)} 条唯一的高校记录。")
    print(f"👉 干净的名单已保存至：{output_file}")
    print("=========================================")


if __name__ == "__main__":
    clean_university_data()