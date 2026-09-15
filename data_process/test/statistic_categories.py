import pandas as pd
import os
import collections


def analyze_arxiv_tags():
    print("📊 启动【arXiv 原生基因全景透视引擎】...\n")

    PHASES = {
        'Phase1': 'phase1_arxiv_categories.csv',
        'Phase2': 'phase2_arxiv_categories.csv',
        'Phase3': 'phase3_arxiv_categories.csv'
    }

    df_list = []
    for phase_name, file_path in PHASES.items():
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, dtype=str)
                df['stage'] = phase_name
                df_list.append(df)
            except Exception as e:
                print(f"⚠️ 读取 {file_path} 失败: {e}")

    if not df_list:
        print("❌ 没有找到任何 arXiv 提取结果文件！")
        return

    full_df = pd.concat(df_list, ignore_index=True)
    total_papers = len(full_df)
    print(f"✅ 成功加载 {total_papers} 篇带有 arXiv 原生标签的文献。\n")

    # 1. 拆解所有标签 (把 cs.AI|econ.GN 拆成两行)
    # 过滤掉空值
    full_df = full_df.dropna(subset=['all_categories'])

    # 统计精细标签 (如 cs.AI, econ.GN)
    all_tags = []
    # 统计大类领域 (如 cs, econ, q-bio)
    broad_domains = []

    for cats in full_df['all_categories']:
        tags = str(cats).split('|')
        all_tags.extend(tags)
        for tag in tags:
            # 提取点号前面的部分作为大类，比如 q-fin.ST -> q-fin
            domain = tag.split('.')[0] if '.' in tag else tag
            broad_domains.append(domain)

    # 2. 统计精细标签 Top 20
    print("🏆 【精细标签 (Sub-categories) 全局 Top 20】")
    print("=" * 60)
    tag_counter = collections.Counter(all_tags)
    for rank, (tag, count) in enumerate(tag_counter.most_common(20), 1):
        pct = (count / total_papers) * 100
        print(f" {rank:2d}. {tag:<15} | {count:6d} 次出现 (覆盖了 {pct:.1f}% 的论文)")

    # 3. 统计大类领域 (Broad Domains) 分布
    print("\n🌍 【宽泛学科大类 (Broad Domains) 势力版图】")
    print("=" * 60)
    domain_counter = collections.Counter(broad_domains)
    for rank, (domain, count) in enumerate(domain_counter.most_common(15), 1):
        # 排除掉不规范的 Unknown
        if domain == "Unknown": continue
        pct = (count / total_papers) * 100
        print(f" {rank:2d}. {domain:<15} | {count:6d} 次出现 (覆盖了 {pct:.1f}% 的论文)")

    # 4. 导出完整词典，供你人工审查
    output_summary = 'ArXiv_Tags_Summary.csv'
    tag_df = pd.DataFrame(tag_counter.most_common(), columns=['ArXiv_Tag', 'Frequency'])
    tag_df['Broad_Domain'] = tag_df['ArXiv_Tag'].apply(lambda x: x.split('.')[0] if '.' in x else x)
    tag_df.to_csv(output_summary, index=False, encoding='utf-8-sig')

    print("\n" + "=" * 60)
    print(f"💾 完整的标签频率字典已保存至: {output_summary}")
    print("💡 接下来，我们可以根据这个字典，定义哪些属于'文科跨界'，哪些属于'理科内卷'！")
    print("=" * 60)


if __name__ == "__main__":
    analyze_arxiv_tags()