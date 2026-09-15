import pandas as pd
import json
import ast
from collections import defaultdict
import os
import plotly.graph_objects as go


def generate_hss_focused_analysis():
    print("🚀 启动【人文社科高亮版·桑基图与统计表双引擎】...")

    INPUT_CSV = 'dynamic_diffusion_full_metrics.csv'
    OUTPUT_HTML = 'ai_hss_highlight_sankey.html'
    OUTPUT_STATS = 'hss_diffusion_statistics.csv'

    if not os.path.exists(INPUT_CSV):
        print(f"❌ 找不到文件 {INPUT_CSV}！请确保它在当前目录。")
        return

    # ==========================================
    # 🧠 学科阵营硬核划分 (基于 OpenAlex Level 0)
    # ==========================================
    HSS_DISCIPLINES = {
        "Economics", "Sociology", "Political science", "Psychology",
        "Business", "Geography", "Philosophy", "Art", "History"
    }

    df = pd.read_csv(INPUT_CSV)
    df['clean_stage'] = df['stage'].astype(str).str.replace(' ', '').str.capitalize()
    PHASE_ORDER = ["Phase1", "Phase2", "Phase3"]

    # 用于统计的二维字典：stats_dict["Economics"]["Phase1"] = 100
    stats_dict = defaultdict(lambda: {"Phase1": 0, "Phase2": 0, "Phase3": 0, "Total": 0})

    print("🧠 正在进行全量聚合与文科标记...")
    for _, row in df.iterrows():
        phase = row['clean_stage']
        if phase not in PHASE_ORDER: continue

        raw_dist = row.get('ood_distribution', '{}')
        if pd.isna(raw_dist) or raw_dist == '{}': continue

        try:
            if isinstance(raw_dist, str):
                try:
                    distribution = json.loads(raw_dist)
                except json.JSONDecodeError:
                    distribution = ast.literal_eval(raw_dist)
            else:
                distribution = raw_dist

            for discipline, count in distribution.items():
                stats_dict[discipline][phase] += count
                stats_dict[discipline]["Total"] += count
        except:
            pass

    # ==========================================
    # 📊 任务一：生成极其严谨的统计数据表
    # ==========================================
    print("\n📊 正在生成全局学科演化统计表...")
    records = []
    all_target_disciplines = []

    # 按照总火力从大到小排序，保证表格的美观
    for disc, counts in sorted(stats_dict.items(), key=lambda x: x[1]["Total"], reverse=True):
        all_target_disciplines.append(disc)
        category = "人文社科 (HSS)" if disc in HSS_DISCIPLINES else "理工农医 (STEM)"

        records.append({
            "学科名称": disc,
            "学科分类": category,
            "Phase 1 引用量": counts["Phase1"],
            "Phase 2 引用量": counts["Phase2"],
            "Phase 3 引用量": counts["Phase3"],
            "总跨界引用量": counts["Total"]
        })

    df_stats = pd.DataFrame(records)
    df_stats.to_csv(OUTPUT_STATS, index=False, encoding='utf-8-sig')
    print(f"✅ 统计表生成成功！已保存为: {OUTPUT_STATS}")

    # ==========================================
    # 🎨 任务二：绘制文科高亮版桑基图
    # ==========================================
    print("\n🎨 正在绘制【文科高亮版】桑基图...")

    all_nodes = list(PHASE_ORDER)
    # 阶段节点的颜色 (左侧)
    node_colors = [
        "rgba(31, 119, 180, 0.9)",  # Phase 1: 蓝
        "rgba(255, 127, 14, 0.9)",  # Phase 2: 橙
        "rgba(44, 160, 44, 0.9)"  # Phase 3: 绿
    ]

    node_x = [0.01, 0.01, 0.01]
    node_y = [0.1, 0.5, 0.9]

    discipline_to_idx = {}
    current_idx = 3

    # 分配目标学科节点的颜色 (右侧)
    for disc in all_target_disciplines:
        all_nodes.append(disc)
        discipline_to_idx[disc] = current_idx

        # 🌟 核心视觉引擎：文科高亮，理科低调
        if disc in HSS_DISCIPLINES:
            # 给文科分配极其亮眼的红色系/暖色系
            node_colors.append("rgba(214, 39, 40, 0.95)")  # 亮红色
        else:
            # 理科统一使用带点透明度的低调灰
            node_colors.append("rgba(200, 200, 200, 0.6)")

        current_idx += 1

    # 均匀分布右侧节点的 Y 坐标
    target_count = len(all_target_disciplines)
    node_x.extend([0.99] * target_count)
    node_y.extend([i / max(1, target_count - 1) for i in range(target_count)])

    sankey_links = []
    link_colors = []

    # 定义流速带的基础颜色
    LINK_COLOR_MAP = {
        "Phase1": "rgba(31, 119, 180, 0.2)",
        "Phase2": "rgba(255, 127, 14, 0.2)",
        "Phase3": "rgba(44, 160, 44, 0.2)"
    }
    # 定义流向文科的高亮流速带颜色 (加深不透明度，极其惹眼)
    LINK_HIGHLIGHT_MAP = {
        "Phase1": "rgba(31, 119, 180, 0.6)",
        "Phase2": "rgba(255, 127, 14, 0.6)",
        "Phase3": "rgba(44, 160, 44, 0.6)"
    }

    for phase in PHASE_ORDER:
        source_idx = PHASE_ORDER.index(phase)
        for discipline in all_target_disciplines:
            count = stats_dict[discipline][phase]
            if count == 0: continue

            target_idx = discipline_to_idx[discipline]
            sankey_links.append({
                "source": source_idx,
                "target": target_idx,
                "value": count,
                "label": f"{phase} -> {discipline}"
            })

            # 🌟 流速带如果指向文科，颜色变深、变清晰！
            if discipline in HSS_DISCIPLINES:
                link_colors.append(LINK_HIGHLIGHT_MAP[phase])
            else:
                link_colors.append(LINK_COLOR_MAP[phase])

    df_links = pd.DataFrame(sankey_links)

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=15,
            thickness=25,
            line=dict(color="white", width=1),
            label=all_nodes,
            color=node_colors,
            x=node_x,
            y=node_y
        ),
        link=dict(
            source=df_links['source'],
            target=df_links['target'],
            value=df_links['value'],
            color=link_colors
        ))])

    fig.update_layout(
        title_text="🕸️ AI 的跨学科演化图：人文社科(红)与理工农医(灰)的全面渗透",
        font_size=13,
        height=900  # 加大画幅，容纳所有19个学科
    )

    fig.write_html(OUTPUT_HTML)
    print(f"🎉 桑基图生成成功！已保存为: {OUTPUT_HTML}")
    print("\n👉 现在你可以双击打开网页看图，同时去 Excel 里检查那份严谨的统计表了！")


if __name__ == "__main__":
    generate_hss_focused_analysis()