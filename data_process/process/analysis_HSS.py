import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
from scipy import stats


def analyze_segmented_trend_by_team_dna():
    print("🌋 启动【Team DNA 浓度提取 & 分段趋势分析】引擎 (学术定稿版)...\n")

    # 1. 加载 MASTER 表，构建 ID 到 DNA 的映射字典
    print("🧬 正在加载 12 年全量作者的 DNA 图谱...")
    master_df = pd.read_csv('MASTER_AUTHOR_PROFILES_12YEARS.csv')

    id_to_dna = {}
    for _, row in master_df.iterrows():
        a_id = str(row['author_id'])
        # 安全解析存储的 JSON 字符串
        try:
            dna = json.loads(row['scopus_dna_weights'])
        except:
            dna = {}
        id_to_dna[a_id] = dna

    hss_fields = {
        'Social Sciences', 'Arts and Humanities', 'Psychology',
        'Business, Management and Accounting', 'Economics, Econometrics and Finance', 'Decision Sciences'
    }

    phase_files = [
        'E:\PythonProject\GenAI\data_scrap_v2\stage_1\phase1_hybrid_cleaned.jsonl',
        'E:\PythonProject\GenAI\data_scrap_v2\stage_2\phase2_hybrid_cleaned.jsonl',
        'E:\PythonProject\GenAI\data_scrap_v2\stage_3\phase3_hybrid_cleaned.jsonl'
    ]

    paper_records = []

    # ---------------------------------------------------------
    # 2. 逐篇提纯“团队全基因组”的文科浓度
    # ---------------------------------------------------------
    print("🧮 正在计算几十万篇论文的【真实文科浓度】与【发表时序】...")
    for path in phase_files:
        if not os.path.exists(path): continue

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    work = json.loads(line)

                    # 尝试获取精确的发表日期，如果没有则退化到年份
                    pub_date_str = work.get('publication_date')
                    if pub_date_str:
                        pub_date = pd.to_datetime(pub_date_str, errors='coerce')
                    else:
                        pub_year = work.get('publication_year')
                        if not pub_year: continue
                        pub_date = pd.to_datetime(f"{pub_year}-06-30")  # 默认年中

                    if pd.isna(pub_date) or pub_date.year < 2014 or pub_date.year > 2026:
                        continue

                    a_ids = [str(aid).strip() for aid in work.get('author_ids', []) if aid]
                    if not a_ids: continue

                    # 💡 核心算法：Team DNA 完美融合
                    team_dna_sum = {}
                    valid_authors = 0

                    for aid in a_ids:
                        author_dna = id_to_dna.get(aid, {})
                        if author_dna:
                            valid_authors += 1
                            for field, weight in author_dna.items():
                                team_dna_sum[field] = team_dna_sum.get(field, 0.0) + weight

                    if valid_authors == 0: continue

                    # 算出这篇论文的总文科浓度 (0.00 ~ 1.00)
                    hss_concentration = 0.0
                    for field in hss_fields:
                        hss_concentration += team_dna_sum.get(field, 0.0)

                    hss_concentration = hss_concentration / valid_authors

                    paper_records.append({
                        'date': pub_date,
                        'hss_concentration': hss_concentration * 100  # 转为百分比
                    })
                except Exception:
                    pass

    df_papers = pd.DataFrame(paper_records)

    # ---------------------------------------------------------
    # 3. 按季度聚合 (Quarterly Aggregation)
    # ---------------------------------------------------------
    print("📈 正在构建季度时间序列面板数据...")
    df_papers['Quarter'] = df_papers['date'].dt.to_period('Q')

    # 算出每个季度所有论文的平均文科浓度
    ts_data = df_papers.groupby('Quarter')['hss_concentration'].mean().reset_index()
    # 将 Period 转换回该季度的中间日期，方便画图
    ts_data['Plot_Date'] = ts_data['Quarter'].dt.to_timestamp() + pd.Timedelta(days=45)

    # 过滤掉 2026 年不完整的数据，避免尾部失真
    ts_data = ts_data[ts_data['Plot_Date'].dt.year <= 2025]

    # ---------------------------------------------------------
    # 4. 画图：分段线性回归 (Segmented Trend Analysis)
    # ---------------------------------------------------------
    print("🎨 正在生成【分段趋势分析】顶刊级学术图...")

    # 设定技术断点：2022年11月 (ChatGPT 发布) -> 划定在 2022 Q4
    breakpoint_date = pd.to_datetime('2022-11-01')

    # 拆分断点前 (Pre-Shock) 和 断点后 (Post-Shock) 的数据
    pre_shock = ts_data[ts_data['Plot_Date'] < breakpoint_date]
    post_shock = ts_data[ts_data['Plot_Date'] >= breakpoint_date]

    fig, ax = plt.subplots(figsize=(12, 7))

    # 画散点 (标签学术化)
    ax.scatter(ts_data['Plot_Date'], ts_data['hss_concentration'], color='#8D99AE', alpha=0.6, s=60, edgecolors='black',
               label='Quarterly Mean HSS Weight')

    # 计算并画出 Pre-Shock 的回归线 (标签学术化)
    if len(pre_shock) > 1:
        x_pre = mdates.date2num(pre_shock['Plot_Date'])
        y_pre = pre_shock['hss_concentration']
        z_pre = np.polyfit(x_pre, y_pre, 1)  # 1阶线性拟合
        p_pre = np.poly1d(z_pre)
        ax.plot(pre_shock['Plot_Date'], p_pre(x_pre), color='#003049', linewidth=3, linestyle='-',
                label='Pre-shock Trend')

    # 计算并画出 Post-Shock 的回归线 (标签学术化)
    if len(post_shock) > 1:
        x_post = mdates.date2num(post_shock['Plot_Date'])
        y_post = post_shock['hss_concentration']
        z_post = np.polyfit(x_post, y_post, 1)
        p_post = np.poly1d(z_post)
        ax.plot(post_shock['Plot_Date'], p_post(x_post), color='#D62828', linewidth=4, linestyle='-',
                label='Post-shock Trend')

    # 画下那条改变历史的垂直断点线
    ax.axvline(x=breakpoint_date, color='#F77F00', linestyle='--', linewidth=2, zorder=0)

    # 标注 ChatGPT 发布
    ax.annotate('ChatGPT\nReleased\n(Nov 2022)',
                xy=(breakpoint_date, ax.get_ylim()[1] * 0.85),
                xytext=(breakpoint_date - pd.Timedelta(days=300), ax.get_ylim()[1] * 0.9),
                arrowprops=dict(facecolor='#F77F00', shrink=0.05, width=1.5, headwidth=8),
                fontsize=12, fontweight='bold', color='#F77F00', ha='right')

    # 坐标轴学术化改造
    ax.set_ylabel('Mean HSS Disciplinary Weight (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Publication Quarter', fontsize=14, fontweight='normal')

    # 设置 X 轴的年份显示格式
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper left', fontsize=12, frameon=True)

    plt.tight_layout()
    # 输出文件名学术化
    output_img = 'Segmented_Trend_Team_DNA_Evolution.png'
    plt.savefig(output_img, dpi=300, bbox_inches='tight')

    print(f"💾 严谨版分段趋势图已高清保存为: {output_img}")
    plt.show()

    # -------------------------------------------------------
    # 5. 正式统计检验：提取斜率、标准误、p 值
    # -------------------------------------------------------
    print("\n📊 正式分段回归统计检验结果：")

    # 将日期转换为以 "第几个季度" 为单位的数值（更易解读）
    ts_data = ts_data.reset_index(drop=True)
    ts_data['t'] = range(len(ts_data))

    # 断点索引（pre/post 数据点数量）
    breakpoint_idx = len(pre_shock)

    pre_data = ts_data[ts_data['t'] < breakpoint_idx].copy()
    post_data = ts_data[ts_data['t'] >= breakpoint_idx].copy()

    # --- Pre-shock 分段回归 ---
    X_pre = sm.add_constant(pre_data['t'])
    model_pre = sm.OLS(pre_data['hss_concentration'], X_pre).fit()
    slope_pre = model_pre.params['t']
    pval_pre = model_pre.pvalues['t']
    print(f"  Pre-shock  斜率: {slope_pre:.4f} pp/季度  |  p = {pval_pre:.4f}")

    # --- Post-shock 分段回归 ---
    X_post = sm.add_constant(post_data['t'])
    model_post = sm.OLS(post_data['hss_concentration'], X_post).fit()
    slope_post = model_post.params['t']
    pval_post = model_post.pvalues['t']
    print(f"  Post-shock 斜率: {slope_post:.4f} pp/季度  |  p = {pval_post:.4f}")

    # --- 斜率差异的 Chow Test（结构突变检验）---
    # 构造完整 ITS 模型: Y = β0 + β1·t + β2·D + β3·D·t + ε
    #   D = 0 for pre, 1 for post
    ts_data['D'] = (ts_data['t'] >= breakpoint_idx).astype(int)
    ts_data['D_t'] = ts_data['D'] * ts_data['t']

    X_full = sm.add_constant(ts_data[['t', 'D', 'D_t']])
    model_full = sm.OLS(ts_data['hss_concentration'], X_full).fit()

    slope_diff = model_full.params['D_t']
    pval_diff = model_full.pvalues['D_t']
    level_shift = model_full.params['D']
    pval_level = model_full.pvalues['D']

    print(f"\n  完整 ITS 模型结果：")
    print(f"  水平突变 (Level Shift):  {level_shift:.4f} pp  |  p = {pval_level:.4f}")
    print(f"  斜率突变 (Slope Change): {slope_diff:.4f} pp/季度  |  p = {pval_diff:.4f}")
    print(f"\n  模型 R²: {model_full.rsquared:.4f}")
    print(model_full.summary())

# 引爆核心逻辑！
analyze_segmented_trend_by_team_dna()