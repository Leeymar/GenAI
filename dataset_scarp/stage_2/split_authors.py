import pandas as pd
import json
import os


def split_veterans_and_newcomers():
    print("🧬 启动【老兵档案继承】与【新兵清单剥离】引擎...\n")

    csv_file = 'phase2_active_authors_with_historical_T0.csv'
    phase1_jsonl = 'E:\PythonProject\GenAI\data_process\stage_1\phase1_authors_full_census_final_with_chinese_tags_safe.jsonl'

    out_veterans = 'phase2_veterans_full_profiles.jsonl'
    out_newcomers = 'phase2_newcomers_to_fetch.jsonl'

    # ---------------------------------------------------------
    # 1. 加载第二阶段全量活跃名单 (包含 T0 和 最新累加战绩)
    # ---------------------------------------------------------
    print(f"📥 正在读取第二阶段总名单: {csv_file}")
    df_all = pd.read_csv(csv_file)

    # 构建一个字典，方便极速查找。键为完整的 URL ID
    # 结构: {'https://openalex.org/A5086198262': {'T0_year': 2014, 'paper_count': 57}}
    active_authors_dict = {}
    for _, row in df_all.iterrows():
        a_id = str(row['authors_id']).strip()
        active_authors_dict[a_id] = {
            'T0_year': int(row['T0_year']),
            'paper_count': int(row['paper_count'])
        }
    print(f"   ➤ 共加载 {len(active_authors_dict):,} 位第二阶段活跃学者。\n")

    # ---------------------------------------------------------
    # 2. 扫描第一阶段旧档案，提取老兵并更新战绩
    # ---------------------------------------------------------
    print(f"📥 正在扫描第一阶段旧档案库: {phase1_jsonl}")
    veterans_count = 0

    with open(phase1_jsonl, 'r', encoding='utf-8') as f_in, \
            open(out_veterans, 'w', encoding='utf-8') as f_vet:

        for line in f_in:
            if not line.strip(): continue
            try:
                profile = json.loads(line)
                a_id = profile.get('author_ids')

                # 如果这个第一阶段的老兵，出现在了第二阶段的活跃名单里
                if a_id in active_authors_dict:
                    veterans_count += 1

                    # 🎯 核心更新：把旧档案里的发文量，替换为包含第一+第二阶段的最新总战绩！
                    profile['paper_count'] = active_authors_dict[a_id]['paper_count']
                    # 确保 T0 是第一阶段的年份
                    profile['T0_year'] = active_authors_dict[a_id]['T0_year']

                    # 写入老兵最终档案库
                    f_vet.write(json.dumps(profile, ensure_ascii=False) + '\n')

                    # 🛑 从活跃名单里将他剔除，剩下的就全是纯新兵了！
                    del active_authors_dict[a_id]

            except Exception as e:
                continue

    print(f"   ➤ 成功继承并更新了 {veterans_count:,} 位老兵的详尽档案！")
    print(f"   💾 老兵数据已安全保存至: {out_veterans}\n")

    # ---------------------------------------------------------
    # 3. 将剩下的纯新兵剥离为待查询清单
    # ---------------------------------------------------------
    print("🧮 正在将剩余的纯新兵打包为待查询清单...")
    newcomers_count = 0

    with open(out_newcomers, 'w', encoding='utf-8') as f_new:
        # 现在 active_authors_dict 里剩下的，全是没有 Phase 1 档案的新人
        for a_id, stats in active_authors_dict.items():
            newcomer_record = {
                "author_ids": a_id,
                "T0_year": stats['T0_year'],
                "paper_count": stats['paper_count']
                # 其他字段留空，等下一步去 OpenAlex 查完补齐
            }
            f_new.write(json.dumps(newcomer_record, ensure_ascii=False) + '\n')
            newcomers_count += 1

    print("================================================================")
    print(" 🎯 第二阶段人员分流战报")
    print("================================================================")
    print(f"🎖️ 继承老兵 (无需再查 API):  {veterans_count:>8,.0f} 人")
    print(f"👶 剥离新兵 (下一步需查 API): {newcomers_count:>8,.0f} 人")
    print(f"🔥 总计无缝对齐:          {veterans_count + newcomers_count:>8,.0f} 人")
    print("================================================================\n")
    print(f"💾 新兵待查清单已保存至: {out_newcomers}")
    print("💡 接下来，我们只需要对着这个 newcormers 文件，疯狂扫街查询他们的历史学科即可！")


# 运行这台分流手术机！
split_veterans_and_newcomers()