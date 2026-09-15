import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_evolution_with_overall():
    # 提取自你的三大阶段 + 全量 OLS 回归表系数 (Coef)
    data = {
        'HSS_Concentration': ['0% (Pure STEM)', '1-20%', '21-40%', '41-60%', '>60%'],
        'Phase_1': [0, 0.0352, -0.3112, 0.0091, -0.2168],
        'Phase_2': [0, -0.0403, -0.0392, -0.0107, 0.0458],
        'Phase_3': [0, 0.0087, 0.0557, 0.1292, 0.1854],
        'Overall': [0, 0.0010, 0.0465, 0.1202, 0.1768]  # 全量数据的系数
    }

    df = pd.DataFrame(data)

    plt.figure(figsize=(12, 7.5), dpi=300)
    sns.set_theme(style="whitegrid", context="talk")

    # 1. 绘制三条历史演进折线
    plt.plot(df['HSS_Concentration'], df['Phase_1'], marker='o', linestyle='-.',
             linewidth=2.5, markersize=8, label='Phase 1: Early AI Era', color='gray', alpha=0.5)

    plt.plot(df['HSS_Concentration'], df['Phase_2'], marker='s', linestyle='--',
             linewidth=2.5, markersize=9, label='Phase 2: Deep Learning Era', color='darkorange', alpha=0.8)

    plt.plot(df['HSS_Concentration'], df['Phase_3'], marker='^', linestyle='-',
             linewidth=3.5, markersize=12, label='Phase 3: LLM Era (Paradigm Shift)', color='crimson')

    # 2. 🌟 绘制全量数据的 整体趋势线
    plt.plot(df['HSS_Concentration'], df['Overall'], marker='D', linestyle=':',
             linewidth=4, markersize=10, label='Overall (12 Years Pooled)', color='black')

    # 绘制 0 基准线 (纯理科水位线)
    plt.axhline(0, color='black', linestyle='-', linewidth=1.5, zorder=1)

    # 背景区域涂色
    plt.fill_between(df['HSS_Concentration'], 0, -0.35, color='gray', alpha=0.1)
    plt.text(0.1, -0.25, 'Penalty Zone\n(Negative Impact)', color='gray', fontsize=14, fontweight='bold', alpha=0.7)

    plt.fill_between(df['HSS_Concentration'], 0, 0.25, color='green', alpha=0.05)
    plt.text(0.1, 0.18, 'Premium Zone\n(Positive Reward)', color='green', fontsize=14, fontweight='bold', alpha=0.6)

    # 标签优化
    plt.title('The Awakening of Humanities & Social Sciences in AI:\nOverall Premium vs. Historical Evolution',
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('HSS Knowledge Concentration in Team DNA', fontsize=14, fontweight='bold')
    plt.ylabel('Citation Premium / Penalty (Regression Coef)', fontsize=14, fontweight='bold')

    plt.ylim(-0.35, 0.25)

    # 将图例移到左上角，防止挡住右侧飙升的曲线
    plt.legend(fontsize=12, loc='upper left', frameon=True, shadow=True)
    plt.tight_layout()

    # 保存
    output_filename = 'Premium_Evolution_with_Overall.png'
    plt.savefig(output_filename)
    print(f"📊 图表已生成并保存为: {output_filename}")
    plt.show()


if __name__ == "__main__":
    plot_evolution_with_overall()