import json
import os
import pandas as pd


def calculate_specific_discipline_flow_5bins():
    print("🚀 启动【起点基因 (5大梯队) -> 具体目标学科：微观流向统计引擎】...\n")

    # 1. 加载作者画像 DNA
    print("📖 正在加载作者画像基因库...")
    try:
        master_df = pd.read_csv('MASTER_AUTHOR_PROFILES_12YEARS.csv')
        id_to_dna = {}
        for _, row in master_df.iterrows():
            a_id = str(row['author_id']).strip()
            try:
                id_to_dna[a_id] = json.loads(row['scopus_dna_weights'])
            except:
                id_to_dna[a_id] = {}
    except Exception as e:
        print(f"❌ 读取作者库失败: {e}，请确认文件没有被打开！")
        return

    hss_fields = {'Social Sciences', 'Arts and Humanities', 'Psychology',
                  'Business, Management and Accounting', 'Economics, Econometrics and Finance', 'Decision Sciences'}

    SOURCE_PHASE_FILES = [
        r'E:\PythonProject\GenAI\data_scrap_v2\stage_1\phase1_hybrid_cleaned.jsonl',
        r'E:\PythonProject\GenAI\data_scrap_v2\stage_2\phase2_hybrid_cleaned.jsonl',
        r'E:\PythonProject\GenAI\data_scrap_v2\stage_3\phase3_hybrid_cleaned.jsonl'
    ]
    METRICS_CSV = 'dynamic_diffusion_full_metrics.csv'

    # 2. 计算源文献的浓度
    print("🧬 正在计算源文献文科浓度...")
    work_hss_map = {}
    for phase_file in SOURCE_PHASE_FILES:
        if not os.path.exists(phase_file): continue
        with open(phase_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    work = json.loads(line)
                    raw_id = work.get('work_id') or work.get('id')
                    clean_id = str(raw_id).split('/')[-1]

                    a_ids = [str(aid).strip() for aid in work.get('author_ids', []) if aid]
                    if not a_ids: continue

                    team_dna_sum = {}
                    valid_authors = 0
                    for aid in a_ids:
                        author_dna = id_to_dna.get(aid, {})
                        if author_dna:
                            valid_authors += 1
                            for field, weight in author_dna.items():
                                team_dna_sum[field] = team_dna_sum.get(field, 0.0) + weight

                    if valid_authors > 0:
                        hss_score = sum(team_dna_sum.get(field, 0.0) for field in hss_fields)
                        work_hss_map[clean_id] = (hss_score / valid_authors) * 100
                except:
                    pass

    # 3. 爆炸拆解，严格应用 5 个浓度梯队
    print("💥 正在拆解 OOD 流向，记录具体的接收学科...")
    df_metrics = pd.read_csv(METRICS_CSV)
    flow_records = []

    for _, row in df_metrics.iterrows():
        wid = str(row['work_id'])
        if wid not in work_hss_map: continue

        hss_score = work_hss_map[wid]

        # 🌟 强硬切分：5大浓度梯队
        if hss_score == 0:
            source_bin = '1. Pure_STEM (0%)'
        elif hss_score <= 20:
            source_bin = '2. Low_HSS (1-20%)'
        elif hss_score <= 40:
            source_bin = '3. Moderate_HSS (21-40%)'
        elif hss_score <= 60:
            source_bin = '4. High_HSS (41-60%)'
        else:
            source_bin = '5. HSS_Dominant (>60%)'

        stage = str(row['stage']).replace(' ', '').capitalize()

        try:
            ood_dist = json.loads(row['ood_distribution'])
            for target_discipline, count in ood_dist.items():
                flow_records.append({
                    'Phase': stage,
                    'Source_Team': source_bin,
                    'Target_Discipline': target_discipline,
                    'Citations': count
                })
        except:
            pass

    flow_df = pd.DataFrame(flow_records)

    # 4. 分组统计并提取 Top 5 流向
    print("\n🏆 【跨界知识具体流向揭秘：5大浓度梯队的 Top 5 目标阵地】")
    print("=" * 80)

    grouped = flow_df.groupby(['Phase', 'Source_Team', 'Target_Discipline'])['Citations'].sum().reset_index()

    phases = ['Phase1', 'Phase2', 'Phase3']
    teams = [
        '1. Pure_STEM (0%)',
        '2. Low_HSS (1-20%)',
        '3. Moderate_HSS (21-40%)',
        '4. High_HSS (41-60%)',
        '5. HSS_Dominant (>60%)'
    ]

    for phase in phases:
        phase_data = grouped[grouped['Phase'] == phase]
        if phase_data.empty: continue

        print(f"\n📍 历史阶段: 【{phase}】")
        print("-" * 60)

        for team in teams:
            team_data = phase_data[phase_data['Source_Team'] == team]
            if team_data.empty: continue

            total_team_citations = team_data['Citations'].sum()
            top5_targets = team_data.nlargest(5, 'Citations')

            print(f" 👨‍🔬 源头基因: {team} (总跨界输出: {total_team_citations} 次)")
            for idx, row in top5_targets.iterrows():
                target = row['Target_Discipline']
                count = row['Citations']
                pct = (count / total_team_citations) * 100
                print(f"    --> 流向 [{target}]: {count} 次 ({pct:.1f}%)")
            print("")


if __name__ == "__main__":
    calculate_specific_discipline_flow_5bins()