import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns


def plot_diffusion_direction_stacked_bar():
    print("🚀 启动【跨界方向·100%堆叠柱状图】绘制引擎...\n")

    dna_file = 'Ultimate_ArXiv_DNA_Master.csv'
    diffusion_file = 'dynamic_diffusion_full_metrics.csv'

    if not os.path.exists(dna_file) or not os.path.exists(diffusion_file):
        print("❌ 找不到数据文件，请确保两张表在同目录下！")
        return

    # 1. 缝合数据
    df_dna = pd.read_csv(dna_file)
    df_diff = pd.read_csv(diffusion_file)
    df = pd.merge(df_dna, df_diff, on='work_id', how='inner')

    # 2. 生成 HSS_Bin 浓度分箱
    bins = [-0.1, 0.1, 20.0, 40.0, 60.0, 100.1]
    labels = ['1. Pure STEM (0%)', '2. Low HSS (1-20%)', '3. Mod HSS (21-40%)', '4. High HSS (41-60%)',
              '5. Dominant HSS (>60%)']
    df['HSS_Bin'] = pd.cut(df['team_hss_dna_pct'], bins=bins, labels=labels)

    # 3. 定义宏观学科聚合字典
    macro_mapping = {
        'Medicine': 'Medicine & Biology', 'Biology': 'Medicine & Biology', 'Neuroscience': 'Medicine & Biology',
        'Psychology': 'Social Sci & Humanities', 'Sociology': 'Social Sci & Humanities',
        'Philosophy': 'Social Sci & Humanities', 'Political science': 'Social Sci & Humanities',
        'Art': 'Social Sci & Humanities',
        'Economics': 'Business & Econ', 'Business': 'Business & Econ', 'Management': 'Business & Econ',
        'Finance': 'Business & Econ',
        'Mathematics': 'Hard Sciences (Math/Phys/Eng)', 'Physics': 'Hard Sciences (Math/Phys/Eng)',
        'Engineering': 'Hard Sciences (Math/Phys/Eng)',
        'Materials science': 'Hard Sciences (Math/Phys/Eng)', 'Chemistry': 'Hard Sciences (Math/Phys/Eng)'
    }

    # 4. 统计每个分箱的流向总数
    flow_data = []

    for _, row in df.iterrows():
        hss_bin = str(row['HSS_Bin'])
        if pd.isna(hss_bin) or hss_bin == 'nan': continue

        # 流向本领域的引用量 (内卷池)
        host_count = int(row.get('host_count', 0))
        if host_count > 0:
            flow_data.append(
                {'HSS_Bin': hss_bin, 'Target_Discipline': 'Computer Science (Host)', 'Citations': host_count})

        # 解析流向外领域的引用量 (破圈池)
        try:
            ood_dict = json.loads(row['ood_distribution'])
            for disc, count in ood_dict.items():
                macro_disc = macro_mapping.get(disc, 'Other Disciplines')
                flow_data.append({'HSS_Bin': hss_bin, 'Target_Discipline': macro_disc, 'Citations': count})
        except:
            pass

    df_flow = pd.DataFrame(flow_data)

    # 5. 按照 HSS_Bin 和 Target_Discipline 分组求和
    grouped = df_flow.groupby(['HSS_Bin', 'Target_Discipline'])['Citations'].sum().unstack(fill_value=0)

    # 6. 转换为 100% 百分比
    # 每一行的和归一化为 1 (100%)
    grouped_percentage = grouped.div(grouped.sum(axis=1), axis=0) * 100

    # 确保特定的顺序以优化视觉效果 (把计算机放在最下面当基石，破圈的放在上面)
    target_order = [
        'Computer Science (Host)',
        'Hard Sciences (Math/Phys/Eng)',
        'Medicine & Biology',
        'Business & Econ',
        'Social Sci & Humanities',
        'Other Disciplines'
    ]
    # 过滤掉不存在的列
    target_order = [col for col in target_order if col in grouped_percentage.columns]
    grouped_percentage = grouped_percentage[target_order]

    # 7. 渲染高级学术堆叠图
    print("🎨 正在绘制流向分布 100% 堆叠柱状图...")

    # 设定高对比度的配色方案
    colors = ['#BDC3C7', '#3498DB', '#2ECC71', '#F1C40F', '#E74C3C', '#95A5A6']

    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

    grouped_percentage.plot(kind='bar', stacked=True, color=colors, ax=ax, width=0.7, edgecolor='white', linewidth=1.5)

    plt.title('Knowledge Diffusion Directions by Interdisciplinary Team DNA\n(Where does the impact flow?)',
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Team HSS DNA Concentration', fontsize=14, fontweight='bold')
    plt.ylabel('Proportion of Receiving Disciplines (%)', fontsize=14, fontweight='bold')

    # 旋转 X 轴标签，使其更容易阅读
    plt.xticks(rotation=15, ha='right', fontsize=11)

    # 将图例放在图表右侧外围
    plt.legend(title='Absorbing Disciplines', title_fontsize='13', fontsize='11',
               loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True, shadow=True)

    # 在图上标注具体数值（仅标注大于 3% 的区块，保持图面整洁）
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy()
        if height > 3.0:  # 只有占比超过 3% 的方向才显示数字
            ax.text(x + width / 2,
                    y + height / 2,
                    f'{height:.1f}%',
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=10, color='black' if height > 20 else 'white', fontweight='bold')

    plt.tight_layout()
    output_filename = 'Diffusion_Directions_Stacked.png'
    plt.savefig(output_filename)
    print(f"💾 极其直观的【跨学科流向分布图】已保存为: {output_filename}")
    plt.show()


if __name__ == "__main__":
    plot_diffusion_direction_stacked_bar()