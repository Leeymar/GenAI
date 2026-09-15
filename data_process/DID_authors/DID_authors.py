import json
import os
import pandas as pd
import ast
from collections import defaultdict


def extract_and_match_authors():
    print("🚀 启动【跨界作者雷达】与【基础身份无损提取】流水线...\n")

    # ==========================================
    # 阶段 1：圈定那 9 万个作者 (带破冰年份)
    # ==========================================
    hss_disciplines = {
        "Psychology", "Social Sciences", "Economics", "Arts and Humanities",
        "Business, Management and Accounting", "Decision Sciences",
        "Neuroscience"  # 请根据你实际情况微调
    }

    author_profile_file = r'E:\PythonProject\GenAI\data_process\process\MASTER_AUTHOR_PROFILES_12YEARS.csv'
    print(f"📖 阶段1：正在读取作者底层基因库: {author_profile_file} ...")

    author_hss_baseline = {}
    try:
        df_authors = pd.read_csv(author_profile_file)
        for _, row in df_authors.iterrows():
            a_id = str(row['author_id']).strip()
            dna_str = str(row.get('scopus_dna_weights', '{}'))
            try:
                dna_dict = ast.literal_eval(dna_str)
                hss_sum = sum(weight for subj, weight in dna_dict.items() if subj in hss_disciplines)
                author_hss_baseline[a_id] = hss_sum
            except:
                author_hss_baseline[a_id] = 0.0
    except Exception as e:
        print(f"❌ 读取作者基因库失败，请检查路径: {e}")
        return

    jsonl_files = [
        r'E:\PythonProject\GenAI\data_scrap_v2\stage_1\phase1_hybrid_cleaned.jsonl',
        r'E:\PythonProject\GenAI\data_scrap_v2\stage_2\phase2_hybrid_cleaned.jsonl',
        r'E:\PythonProject\GenAI\data_scrap_v2\stage_3\phase3_hybrid_cleaned.jsonl'
    ]

    print("📂 阶段1：正在扫描论文合作网络，判定个人时间线...")
    author_timelines = defaultdict(list)

    for file_path in jsonl_files:
        if not os.path.exists(file_path): continue
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                paper = json.loads(line)
                pub_year = paper.get('publication_year')
                if not pub_year: continue

                authors_in_paper = paper.get('author_ids', [])
                if not authors_in_paper: continue

                # ================= 加上这层防弹过滤 =================
                # 剔除掉列表里的 None 值，并且立刻执行 strip()
                valid_authors = [a.strip() for a in authors_in_paper if a and isinstance(a, str)]
                if not valid_authors: continue

                team_hss_dnas = [author_hss_baseline.get(a, 0.0) for a in valid_authors]
                paper_is_cross_disciplinary = sum(team_hss_dnas) > 0.05

                for a_id in valid_authors:
                    author_timelines[a_id].append({
                        'year': pub_year,
                        'is_cross': paper_is_cross_disciplinary
                    })
                # ====================================================

    print("🔍 阶段1：正在定位【破冰者】与【老顽固】...")
    treated_authors = {}  # {author_id: event_year}
    control_authors = set()  # set(author_id)

    for a_id, papers in author_timelines.items():
        my_dna = author_hss_baseline.get(a_id, 0.0)
        if my_dna > 0.05:
            continue  # 本身是文科生，跳过

        papers.sort(key=lambda x: x['year'])
        if len(papers) < 3:
            continue  # 发文太少，跳过

        has_crossed = False
        event_year = None

        for p in papers:
            if p['is_cross']:
                has_crossed = True
                event_year = p['year']
                break

        if has_crossed:
            papers_before = [p for p in papers if p['year'] < event_year]
            if len(papers_before) >= 1:
                treated_authors[a_id] = event_year
        else:
            control_authors.add(a_id)

    print(f"✅ 阶段1完成！拿到 Treated: {len(treated_authors)} 人, Control: {len(control_authors)} 人。")

    # ==========================================
    # 阶段 2：直接无损保存所有 9 万人的基本身份
    # ==========================================
    output_basic_jsonl = r'E:\PythonProject\GenAI\data_process\DID\All_96k_Authors_Basic.jsonl'
    os.makedirs(os.path.dirname(output_basic_jsonl), exist_ok=True)

    print("\n🚀 阶段2：正在无损保存全部作者的基础身份牌...")
    with open(output_basic_jsonl, 'w', encoding='utf-8') as f_out:
        # 保存实验组
        for a_id, event_year in treated_authors.items():
            record = {"author_id": a_id, "is_treated": 1, "event_year": event_year}
            f_out.write(json.dumps(record, ensure_ascii=False) + '\n')

        # 保存控制组
        for a_id in control_authors:
            record = {"author_id": a_id, "is_treated": 0, "event_year": None}
            f_out.write(json.dumps(record, ensure_ascii=False) + '\n')

    print("=" * 60)
    print("🏆 【身份牌提取战报】")
    print("=" * 60)
    print(f"🎯 成功提取并无损保存: {len(treated_authors) + len(control_authors)} 人")
    print(f"📂 文件已存入: {output_basic_jsonl}")
    print("=" * 60)


if __name__ == "__main__":
    extract_and_match_authors()