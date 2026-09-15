import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── 数据来自 tab:twfe_event_study 列(1)全样本 ────────
# 参考年 T=-1 手动插入（coef=0, SE=0）
events = [-3,    -2,    -1,    0,     1,     2,      3    ]
coeffs = [0.0434, 0.0021, 0.000, 0.0333, 0.0463, -0.0228, -0.0324]
ses    = [0.0217, 0.0170, 0.000, 0.0134,  0.0169,  0.0216,  0.0270]

lo = [c - 1.96 * s for c, s in zip(coeffs, ses)]
hi = [c + 1.96 * s for c, s in zip(coeffs, ses)]

# 显著性标记（对照表格）
# T=-3: * (p<0.05), T=0: * (p<0.05), T=+1: ** (p<0.01)
sig_labels = {-3: '*', 0: '*', 1: '**'}

# ── 画图 ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5.5))

# 背景色块
ax.axvspan(-3.5, -1.5, alpha=0.07, color='gray',  zorder=0, label='_nolegend_')
ax.axvspan(-1.5,  3.5, alpha=0.09, color='green', zorder=0, label='_nolegend_')

# 零线
ax.axhline(0, color='black', lw=0.9, ls='--', alpha=0.45, zorder=1)

# 处理节点竖线
ax.axvline(-0.5, color='crimson', lw=1.8, ls='--',
           alpha=0.85, zorder=3, label='Treatment onset (T = 0)')

# CI 色带
ax.fill_between(events, lo, hi,
                alpha=0.18, color='steelblue', zorder=2)

# 系数折线 + 点
ax.plot(events, coeffs, 'o-',
        color='steelblue', lw=2.2, ms=7,
        zorder=5, label='TWFE ATT [95% CI]')

# 参考年标注
ax.annotate('Reference\n(T = −1)',
            xy=(-1, 0),
            xytext=(-1.6, 0.055),
            fontsize=8.5, color='dimgray',
            arrowprops=dict(arrowstyle='->', color='dimgray', lw=0.9))

# 显著性标记
for e, c in zip(events, coeffs):
    if e in sig_labels:
        offset = 11 if c >= 0 else -16
        ax.annotate(sig_labels[e],
                    xy=(e, c),
                    xytext=(0, offset), textcoords='offset points',
                    ha='center', fontsize=13,
                    color='crimson', fontweight='bold')

# 轴标签与标题
ax.set_title(
    'Dynamic Effects of Cross-Disciplinary Collaboration on Citation Impact\n'
    'TWFE Event Study (Full Sample, Reference Year: T = −1)',
    fontsize=11, fontweight='bold', pad=10)
ax.set_xlabel('Years relative to first cross-boundary collaboration (T = 0)',
              fontsize=11)
ax.set_ylabel('Estimated ATT: Δ log(1 + RCR)', fontsize=11)
ax.set_xticks(events)
ax.set_xticklabels([f'T = {e:+d}' for e in events], fontsize=10)
ax.set_xlim(-3.6, 3.6)

# 图例
pre_patch  = mpatches.Patch(color='gray',  alpha=0.25,
                             label='Pre-treatment window')
post_patch = mpatches.Patch(color='green', alpha=0.25,
                             label='Post-treatment window')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles + [pre_patch, post_patch],
          fontsize=9, loc='upper right', framealpha=0.9)

ax.grid(axis='y', alpha=0.28)
plt.tight_layout()
plt.savefig('twfe_event_study_main.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ 图已保存：twfe_event_study_main.png")
