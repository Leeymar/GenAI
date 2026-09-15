import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def analyze_cs_hss_collaboration():
    print("🤝 启动【CS 与 人文社科 (HSS) 联姻】专属追踪引擎...\n")

    # 1. 加载字典：ID -> 学科基因
    master_df = pd.read_csv('MASTER_AUTHOR_PROFILES_12YEARS.csv')
    id_to_field = dict(zip(master_df['author_id'].astype(str), master_df['primary_origin_field']))

    # 2. 严格定义 HSS 文科阵营
    hss_fields = {
        'Social Sciences',
        'Arts and Humanities',
        'Psychology',
        'Business, Management and Accounting',
        'Economics, Econometrics and Finance',
        'Decision Sciences'
    }

    phase_files = {
        'Phase 1\n(2014-2016)': 'E:\PythonProject\GenAI\data_scrap_v2\stage_1\phase1_hybrid_cleaned.jsonl',
        'Phase 2\n(2017-2021)': 'E:\PythonProject\GenAI\data_scrap_v2\stage_2\phase2_hybrid_cleaned.jsonl',
        'Phase 3\n(2022-2026)': 'E:\PythonProject\GenAI\data_scrap_v2\stage_3\phase3_hybrid_cleaned.jsonl'
    }

    results = []

    # ---------------------------------------------------------
    # 3. 逐篇扫描论文团队基因
    # ---------------------------------------------------------
    for phase, path in phase_files.items():
        if not os.path.exists(path): continue
        clean_phase_name = phase.replace('\n', ' ')
        print(f"🧐 正在提纯 {clean_phase_name} 的文理联姻数据...")

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    work = json.loads(line)
                    a_ids = [str(aid).strip() for aid in work.get('author_ids', []) if aid]
                    if not a_ids: continue

                    has_cs = False
                    has_hss = False

                    # 遍历团队每个人的学科
                    for aid in a_ids:
                        field = id_to_field.get(aid, 'Unknown')
                        if field == 'Computer Science':
                            has_cs = True
                        elif field in hss_fields:
                            has_hss = True

                        # 如果都已经找到了，就可以提前结束循环，提升速度
                        if has_cs and has_hss:
                            break

                    # 归类这篇论文的“联姻状态”
                    if has_cs and has_hss:
                        collab_type = "CS + HSS Collaboration"
                    elif has_cs and not has_hss:
                        collab_type = "CS (Without HSS)"
                    elif not has_cs and has_hss:
                        collab_type = "HSS Independent (No CS)"
                    else:
                        collab_type = "Others (No CS or HSS)"

                    results.append({'Phase': phase, 'Collaboration Type': collab_type})
                except:
                    pass

    df_comp = pd.DataFrame(results)

    # ---------------------------------------------------------
    # 4. 统计占比与绘制极具冲击力的图表
    # ---------------------------------------------------------
    # 计算每个阶段各类别的百分比
    comp_pivot = pd.crosstab(df_comp['Phase'], df_comp['Collaboration Type'], normalize='index') * 100

    # 强制排序，让 "CS + HSS Collaboration" 在最显眼的位置 (比如最底部或最顶部)
    order = ["CS + HSS Collaboration", "HSS Independent (No CS)", "CS (Without HSS)", "Others (No CS or HSS)"]
    comp_pivot = comp_pivot[[c for c in order if c in comp_pivot.columns]]

    print("\n================================================================")
    print(" 🎯 CS & HSS 跨界联姻比例战报 (%)")
    print("================================================================")
    print(comp_pivot.round(2))
    print("================================================================\n")

    # 绘图：我们用极具反差的颜色来突出 CS+HSS
    colors = ['#D62828', '#F77F00', '#003049', '#EAE2B7']  # 红色突出联姻，橙色突出纯文科，深蓝代表纯CS

    fig, ax = plt.subplots(figsize=(10, 7))
    comp_pivot.plot(kind='bar', stacked=True, color=colors, ax=ax, edgecolor='white', width=0.6, alpha=0.9)

    plt.title('The Rise of CS & HSS Collaboration in Generative AI (2014-2026)', fontsize=15, fontweight='bold', pad=20)
    plt.ylabel('Proportion of Research Papers (%)', fontsize=12, fontweight='bold')
    plt.xlabel('Evolutionary Phases', fontsize=12, fontweight='bold')
    plt.xticks(rotation=0)

    # 图例配置
    plt.legend(title="Team Collaboration Typology", bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)

    # 核心高光：在柱子上标注红色的 "CS + HSS" 的具体数字！
    for i, phase in enumerate(comp_pivot.index):
        cs_hss_val = comp_pivot.loc[phase, 'CS + HSS Collaboration']
        # 标注在红色柱子的正中间
        ax.text(i, cs_hss_val / 2, f"{cs_hss_val:.1f}%", ha='center', va='center', color='white', fontweight='bold',
                fontsize=11)

        # 顺便标注一下纯文科独立发文的比例
        hss_ind_val = comp_pivot.loc[phase, 'HSS Independent (No CS)']
        if hss_ind_val > 2:  # 太小就不标了，防重叠
            ax.text(i, cs_hss_val + (hss_ind_val / 2), f"{hss_ind_val:.1f}%", ha='center', va='center', color='black',
                    fontweight='bold', fontsize=10)

    plt.tight_layout()
    output_img = 'CS_HSS_Collaboration_Evolution.png'
    plt.savefig(output_img, dpi=300, bbox_inches='tight')

    print(f"💾 绝杀图表已保存为: {output_img}")
    plt.show()


# 引爆真相！
analyze_cs_hss_collaboration()