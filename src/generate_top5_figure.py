import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

targets = [
    'FliM\n(1shu/1t6b)',
    'Carboxyp. Y\n(1ysc/1wpx)',
    'Barnase\n(1lni/1ay7)',
    'Ran GTPase\n(1om0/1ta3)',
    'TAXI-I\n(1t6e/2b42)'
]

b_aupr = [0.9150, 0.8742, 0.8439, 0.7979, 0.7848]
u_aupr = [0.3598, 0.2838, 0.1568, 0.2314, 0.3128]

b_auc = [0.9829, 0.9647, 0.9460, 0.9761, 0.9487]
u_auc = [0.8025, 0.7399, 0.3746, 0.7664, 0.7080]

b_p10 = [100.0, 100.0, 90.0, 80.0, 100.0]
u_p10 = [50.0, 40.0, 0.0, 20.0, 50.0]

u_f1_fixed = [0.2703, 0.1429, 0.0000, 0.1951, 0.3103]
u_f1_opt = [0.4727, 0.3200, 0.2466, 0.3600, 0.3590]

fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
x = np.arange(len(targets))
width = 0.35

# 1. AUPR (Novelty 4 Conformational Shift)
ax1 = axes[0, 0]
ax1.bar(x - width/2, u_aupr, width, label='Unbound / Apo (Free)', color='#ff7f0e', alpha=0.85, edgecolor='black', lw=0.8)
ax1.bar(x + width/2, b_aupr, width, label='Bound / Holo (Complex)', color='#1f77b4', alpha=0.85, edgecolor='black', lw=0.8)
ax1.set_ylabel('PR-AUC (AUPR)', fontsize=11, fontweight='bold')
ax1.set_title('Novelty 4: AUPR Conformational Shift (Bound vs. Unbound)', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(targets, fontsize=10)
ax1.set_ylim(0, 1.05)
ax1.legend(loc='upper right', frameon=True)
for i in range(len(targets)):
    ax1.text(x[i] + width/2, b_aupr[i] + 0.02, f'{b_aupr[i]:.3f}', ha='center', fontsize=9, fontweight='bold')

# 2. Precision@10 (Novelty 3 Candidate Prioritization)
ax2 = axes[0, 1]
ax2.bar(x - width/2, u_p10, width, label='Unbound P@10', color='#ff7f0e', alpha=0.85, edgecolor='black', lw=0.8)
ax2.bar(x + width/2, b_p10, width, label='Bound P@10 (Screening Purity)', color='#2ca02c', alpha=0.85, edgecolor='black', lw=0.8)
ax2.axhline(15.0, color='gray', linestyle=':', label='Random Baseline (~15%)')
ax2.set_ylabel('Precision @ Top-10 (%)', fontsize=11, fontweight='bold')
ax2.set_title('Novelty 3: Experimental Screening Precision@10 (P@10)', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(targets, fontsize=10)
ax2.set_ylim(0, 115)
ax2.legend(loc='upper right', frameon=True)
for i in range(len(targets)):
    ax2.text(x[i] + width/2, b_p10[i] + 2, f'{b_p10[i]:.0f}%', ha='center', fontsize=9, fontweight='bold')

# 3. AUROC (Overall Fold Discrimination)
ax3 = axes[1, 0]
ax3.bar(x - width/2, u_auc, width, label='Unbound AUROC', color='#ff7f0e', alpha=0.85, edgecolor='black', lw=0.8)
ax3.bar(x + width/2, b_auc, width, label='Bound AUROC', color='#9467bd', alpha=0.85, edgecolor='black', lw=0.8)
ax3.set_ylabel('ROC-AUC (AUROC)', fontsize=11, fontweight='bold')
ax3.set_title('Global Structural Discrimination: Bound vs. Unbound ROC-AUC', fontsize=12, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(targets, fontsize=10)
ax3.set_ylim(0.2, 1.05)
ax3.legend(loc='lower right', frameon=True)
for i in range(len(targets)):
    ax3.text(x[i] + width/2, b_auc[i] + 0.02, f'{b_auc[i]:.3f}', ha='center', fontsize=9, fontweight='bold')

# 4. F1 Recovery via Threshold Optimization (Novelty 1 & 4)
ax4 = axes[1, 1]
ax4.bar(x - width/2, u_f1_fixed, width, label='Unbound at Fixed tau=0.33', color='#d62728', alpha=0.75, edgecolor='black', lw=0.8)
ax4.bar(x + width/2, u_f1_opt, width, label='Unbound at Adaptive tau_opt', color='#17becf', alpha=0.85, edgecolor='black', lw=0.8)
ax4.set_ylabel('F1-Score', fontsize=11, fontweight='bold')
ax4.set_title('Novelty 1 & 4: Unbound Performance Recovery via Adaptive Threshold', fontsize=12, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(targets, fontsize=10)
ax4.set_ylim(0, 0.6)
ax4.legend(loc='upper left', frameon=True)
for i in range(len(targets)):
    gain = ((u_f1_opt[i] - u_f1_fixed[i]) / max(0.001, u_f1_fixed[i])) * 100 if u_f1_fixed[i] > 0 else 0
    ax4.text(x[i] + width/2, u_f1_opt[i] + 0.015, f'+{u_f1_opt[i]-u_f1_fixed[i]:.2f}', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
out_path = 'results/case_study/top5_novelty_showcase.png'
plt.savefig(out_path, dpi=300)
plt.close()
print(f"Saved {out_path}")

art_dir = r'C:\Users\jeetu\Desktop\Machine learning\EGCPPIS\results\case_study'
art_user_dir = r'C:\Users\jeetu\.gemini\antigravity\brain\9cd13324-f55e-45a0-b757-58b9442b8b56'
shutil.copy2(out_path, os.path.join(art_user_dir, 'top5_novelty_showcase.png'))
print("Copied to artifact dir.")

