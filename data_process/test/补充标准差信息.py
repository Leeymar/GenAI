import pandas as pd
import json
import ast
import os

# ==========================================
# 1. 路径配置 (请替换为你本地的实际路径)
# ==========================================
JSONL_FILES = [
    r"E:\PythonProject\GenAI\data_scrap_v2\stage_1\phase1_hybrid_cleaned.jsonl",
    r"E:\PythonProject\GenAI\data_scrap_v2\stage_2\phase2_hybrid_cleaned.jsonl",
    r"E:\PythonProject\GenAI\data_scrap_v2\stage_3\phase3_hybrid_cleaned.jsonl"
]
AUTHOR_PROFILE_PATH = r"E:\PythonProject\GenAI\data_process\process\MASTER_AUTHOR_PROFILES_12YEARS.csv"
BASE_REGRESSION_PATH = r"Ultimate_Regression_Base.csv"  # 你的基准表路径
OUTPUT_PATH = r"Ultimate_Regression_Base_with_Diversity.csv"

# 典型的 Scopus 人文社科 (HSS) 领域大类名称，请确保与你之前的分类一致
HSS_DOMAINS = [
    'Social Sciences',
    'Psychology',
    'Economics, Econometrics and Finance',
    'Arts and Humanities',
    'Business, Management and Accounting',
    'Decision Sciences'
]

# ==========================================
# 2. 从 JSONL 中提取【论文 - 作者 - 年份】映射关系
# ==========================================
print("🚀 正在从 JSONL 文件中提取论文与作者的映射桥梁...")
mapping_records = []

for file_path in JSONL_FILES:
    if not os.path.exists(file_path):
        continue
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                # 统一提取 ID 后缀，去掉 https://openalex.org/ 以防匹配失败
                w_id = data.get('work_id', '').split('/')[-1]
                year = data.get('publication_year')
                author_urls = data.get('author_ids', [])

                for a_url in author_urls:
                    a_id = a_url.split('/')[-1]
                    mapping_records.append({
                        'work_id': w_id,
                        'T0_year': year,  # 与作者画像表的年份对齐
                        'author_id': a_id
                    })
            except Exception as e:
                continue

df_mapping = pd.DataFrame(mapping_records)
print(f"✅ 成功提取映射记录: {len(df_mapping)} 条")

# ==========================================
# 3. 读取作者画像表并计算个人 HSS Score
# ==========================================
print("🚀 正在读取作者画像表并计算个人的 HSS 浓度...")
df_authors = pd.read_csv(AUTHOR_PROFILE_PATH)

# 清理 author_id (去掉 URL 前缀)
df_authors['author_id'] = df_authors['author_id'].astype(str).apply(lambda x: x.split('/')[-1])


def calculate_hss_score(weight_str):
    if pd.isna(weight_str):
        return 0.0
    try:
        # 解析字典字符串 (处理 JSON 格式或单引号字典)
        weights_dict = ast.literal_eval(weight_str) if "'" in weight_str else json.loads(weight_str)
        # 累加属于 HSS 领域的权重
        hss_score = sum(weights_dict.get(domain, 0.0) for domain in HSS_DOMAINS)
        return float(hss_score)
    except:
        return 0.0


df_authors['individual_hss_score'] = df_authors['scopus_dna_weights'].apply(calculate_hss_score)

# 我们只需要 id, 年份 和 算出来的分数
df_authors_clean = df_authors[['author_id', 'T0_year', 'individual_hss_score']]

# ==========================================
# 4. 组装明细，计算标准差 (HSS_Diversity)
# ==========================================
print("🚀 正在计算团队级别的 Cognitive Diversity (标准差)...")
# 根据 作者ID 和 发表年份 进行精确匹配
df_merged = pd.merge(df_mapping, df_authors_clean, on=['author_id', 'T0_year'], how='inner')

# 按 work_id 分组，计算团队内部个人得分的标准差 (ddof=0表示总体标准差)
# 填补 0 是为了处理单作者论文 (算不出标准差会变成 NaN)
team_diversity = df_merged.groupby('work_id')['individual_hss_score'].std(ddof=0).fillna(0).reset_index()
team_diversity.rename(columns={'individual_hss_score': 'HSS_Diversity'}, inplace=True)

# ==========================================
# 5. 拼接到最终的回归主表
# ==========================================
print("🚀 正在合并到最终的回归主表...")
df_base = pd.read_csv(BASE_REGRESSION_PATH)

# 清理回归表里的 work_id (确保格式一致)
df_base['clean_work_id'] = df_base['work_id'].astype(str).apply(lambda x: x.split('/')[-1])

# 左连接！把刚刚算好的 HSS_Diversity 拼进去
df_final = pd.merge(df_base, team_diversity, left_on='clean_work_id', right_on='work_id', how='left')

# 删除合并产生多余的 work_id 列，填补匹配不到的空值为 0
if 'work_id_y' in df_final.columns:
    df_final.drop(columns=['work_id_y'], inplace=True)
    df_final.rename(columns={'work_id_x': 'work_id'}, inplace=True)

df_final['HSS_Diversity'] = df_final['HSS_Diversity'].fillna(0)

# ==========================================
# 6. 保存导出
# ==========================================
df_final.to_csv(OUTPUT_PATH, index=False)
print(f"🎉 搞定！带有 HSS_Diversity 的最终版数据集已保存至: {OUTPUT_PATH}")