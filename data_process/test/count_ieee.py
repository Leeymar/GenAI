import pandas as pd
import os


def count_specific_venue(target_venue="IEEE Access"):
    print(f"🔍 启动定向追踪：统计【{target_venue}】在三大阶段的真实发文量...\n")
    print("=" * 40)

    FILES = {
        'Phase 1 (Pre-Shock)': 'phase1_venues_mapping.csv',
        'Phase 2 (Eruption)': 'phase2_venues_mapping.csv',
        'Phase 3 (Democratization)': 'phase3_venues_mapping.csv'
    }

    total_count = 0

    for phase_name, file_path in FILES.items():
        if os.path.exists(file_path):
            try:
                # 读取数据，强制转换为字符串防报错
                df = pd.read_csv(file_path, usecols=['venue'], dtype=str)

                # 清洗字符串：转小写、去首尾空格，保证 100% 匹配
                df['clean_venue'] = df['venue'].fillna('').str.strip().str.lower()

                # 统计目标期刊
                count = len(df[df['clean_venue'] == target_venue.lower()])
                total_count += count

                print(f"📍 {phase_name:<25}: {count:5d} 篇")
            except Exception as e:
                print(f"⚠️ 读取 {file_path} 失败: {e}")
        else:
            print(f"⚠️ 找不到文件 {file_path}")

    print("-" * 40)
    print(f"🌍 【{target_venue}】 跨界总收录量: {total_count} 篇")
    print("=" * 40)


if __name__ == "__main__":
    count_specific_venue()