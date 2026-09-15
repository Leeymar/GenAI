import pandas as pd
import re


def create_ultimate_elite_variables():
    print("🚀 启动【终极版 QS Top 100 + AI 学科 Top 100 + 工业界】打标引擎...\n")

    # 1. 加载你的机构数据
    try:
        df = pd.read_csv('Author_Institutions_Map.csv')
        df['institution'] = df['institution'].fillna('Unknown').astype(str)
    except FileNotFoundError:
        print("❌ 找不到 Author_Institutions_Map.csv，请确认路径！")
        return

    # 2. 工业界 AI 巨头名单 (保持不变，极其重要)
    industry_keywords = [
        r'\bgoogle\b', r'\bdeepmind\b', r'\bmicrosoft\b', r'\bmeta\b', r'\bfacebook\b',
        r'\bopenai\b', r'\banthropic\b', r'\bapple\b', r'\bamazon\b', r'\baws\b',
        r'\btencent\b', r'\balibaba\b', r'\bbaidu\b', r'\bbytedance\b', r'\bhuawei\b',
        r'\bnvidia\b', r'\bibm\b'
    ]

    # 3. 终极学术殿堂 (QS 综合 100 ∪ CS 100 ∪ AI 100 去重合并版)
    academic_keywords = [
        # --- 北美区 (US & Canada) ---
        r'\bmit\b', r'massachusetts institute of technology', r'stanford', r'harvard', r'caltech',
        r'california institute of technology',
        r'carnegie mellon', r'\bcmu\b', r'princeton', r'yale', r'columbia', r'cornell', r'university of chicago',
        r'university of pennsylvania', r'\bupenn\b', r'johns hopkins', r'northwestern', r'duke', r'brown',
        r'university of washington', r'new york university', r'\bnyu\b', r'university of michigan',
        r'\bucla\b', r'\bucsd\b', r'uc san diego', r'\bucb\b', r'berkeley', r'university of california, irvine',
        r'uc irvine',
        r'illinois at urbana', r'\buiuc\b', r'texas at austin', r'\but austin\b', r'georgia institute of technology',
        r'georgia tech',
        r'purdue', r'university of maryland', r'penn state', r'pennsylvania state', r'boston university',
        r'university of southern california', r'\busc\b',
        r'university of toronto', r'mcgill', r'british columbia', r'\bubc\b', r'waterloo', r'university of alberta',
        r'universite de montreal',

        # --- 欧洲区 (UK & EU) ---
        r'oxford', r'cambridge', r'imperial college', r'university college london', r'\bucl\b',
        r'university of edinburgh',
        r'university of manchester', r'king\'s college london', r'\bkcl\b', r'london school of economics', r'\blse\b',
        r'university of bristol', r'university of warwick', r'university of birmingham', r'university of southampton',
        r'university of leeds', r'university of sheffield', r'university of nottingham', r'queen mary',
        r'durham university', r'university of glasgow',
        r'eth zurich', r'epfl', r'ecole polytechnique federale de lausanne', r'psl university', r'universite psl',
        r'paris-saclay', r'sorbonne', r'institut polytechnique de paris', r'delft', r'university of amsterdam',
        r'ku leuven',
        r'technical university of munich', r'\btum\b', r'ludwig-maximilians', r'\blmu\b', r'kth royal',
        r'politecnico di milano',
        r'sapienza', r'university of bologna', r'tu berlin', r'aalto university', r'lund university',
        r'uppsala university',
        r'heidelberg', r'freie universitat berlin', r'tu wien', r'universitat de barcelona', r'university of zurich',
        r'\buzh\b', r'lomonosov',
        r'trinity college dublin',

        # --- 亚洲及澳洲区 (Asia & Australia) ---
        r'\bnus\b', r'national university of singapore', r'\bntu\b', r'nanyang technological',
        r'tsinghua', r'peking university', r'fudan', r'shanghai jiao tong', r'\bsjtu\b', r'zhejiang university',
        r'ustc', r'university of science and technology of china', r'nanjing university',
        r'beijing institute of technology', r'\bbit\b', r'harbin institute of technology', r'\bhit\b',
        r'tongji university',
        r'university of tokyo', r'kyoto university', r'osaka university', r'institute of science tokyo',
        r'tokyo institute of technology',
        r'seoul national', r'yonsei', r'korea university', r'\bkaist\b', r'\bpostech\b',
        r'university of hong kong', r'\bhku\b', r'chinese university of hong kong', r'\bcuhk\b',
        r'hong kong university of science', r'\bhkust\b',
        r'city university of hong kong', r'\bcityuhk\b', r'hong kong polytechnic',
        r'national taiwan university', r'universiti malaya',
        r'university of melbourne', r'university of sydney', r'new south wales', r'\bunsw\b', r'monash',
        r'university of queensland', r'australian national university', r'\banu\b', r'adelaide university',
        r'university of adelaide',
        r'western australia', r'\buwa\b', r'university of technology sydney', r'\buts\b', r'rmit university',
        r'university of auckland',

        # --- 其他地区 (Middle East, Latin America, India) ---
        r'iit bombay', r'iit delhi', r'iit kanpur', r'iit kharagpur', r'iit madras', r'iisc bangalore',
        r'vellore institute', r'\bvit\b',
        r'universidade de sao paulo', r'universidad de buenos aires', r'tecnologico de monterrey', r'\bitesm\b',
        r'universidad nacional autonoma de mexico', r'\bunam\b',
        r'university of jordan', r'king fahd university', r'\bkfupm\b'
    ]

    # 4. 编译正则
    ind_pattern = re.compile('|'.join(industry_keywords), re.IGNORECASE)
    aca_pattern = re.compile('|'.join(academic_keywords), re.IGNORECASE)

    # 5. 打标函数
    df['is_industry_elite'] = df['institution'].apply(lambda x: 1 if ind_pattern.search(x) else 0)
    df['is_academic_elite'] = df['institution'].apply(lambda x: 1 if aca_pattern.search(x) else 0)

    # 终极光环变量
    df['is_elite_overall'] = (df['is_industry_elite'] | df['is_academic_elite']).astype(int)

    # 6. 统计战果
    total = len(df)
    elite_ind = df['is_industry_elite'].sum()
    elite_aca = df['is_academic_elite'].sum()
    elite_total = df['is_elite_overall'].sum()

    print("=========================================================")
    print("🎯 终极三合一精英光环打标完成！统计结果：")
    print("=========================================================")
    print(f"总计扫描作者: {total:,} 人")
    print(f"🏭 工业巨头作者: {elite_ind:,} 人 ({elite_ind / total * 100:.1f}%)")
    print(f"🏛️ Top100 名校作者: {elite_aca:,} 人 ({elite_aca / total * 100:.1f}%)")
    print(f"✨ 拥有【精英光环】总人数: {elite_total:,} 人 ({elite_total / total * 100:.1f}%)")
    print(f"👥 非精英/普通机构: {total - elite_total:,} 人 ({(total - elite_total) / total * 100:.1f}%)")
    print("=========================================================")

    # 7. 保存结果
    output_file = 'Author_Institutions_Map_Top100.csv'
    df.to_csv(output_file, index=False)
    print(f"💾 附带终极 Top100 光环变量的新表已保存至: {output_file}")


create_ultimate_elite_variables()