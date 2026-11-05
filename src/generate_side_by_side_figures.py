import json
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

with open('results/case_study/side_by_side_pairs_metrics.json') as f:
    data = json.load(f)

pair_labels = [
    'CheY (1ci5/1qa9)',
    'Barnase (1lni/1ay7)',
    'Carboxyp. Y (1ysc/1wpx)',
    'Profilin (4fh4/2g2u)',
    'Ran GTPase (1om0/1ta3)',
    'Subtilisin Inh. (1czp/1ewy)',
    'TAXI-I (1t6e/2b42)',
    'FliM (1shu/1t6b)',
    'PTPRD (2v77/3fju)'
]

u_aupr = [p['unbound_metrics']['aupr'] for p in data]
b_aupr = [p['bound_metrics']['aupr'] for p in data]

u_auc = [p['unbound_metrics']['auc'] for p in data]
b_auc = [p['bound_metrics']['auc'] for p in data]

u_p10 = [p['unbound_metrics']['p@10'] * 100 for p in data]
b_p10 = [p['bound_metrics']['p@10'] * 100 for p in data]

u_f1 = [p['unbound_metrics']['f1'] for p in data]
b_f1 = [p['bound_metrics']['f1'] for p in data]

u_opt_f1 = [p['unbound_metrics']['opt_f1'] for p in data]

fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)

x = np.arange(len(pair_labels))
width = 0.36

# Panel A: AUPR Comparison
ax1 = axes[0, 0]
rects1 = ax1.bar(x - width/2, u_aupr, width, label='Unbound / Apo (Free)', color='#ff7f0e', alpha=0.85, edgecolor='black', lw=0.8)
rects2 = ax1.bar(x + width/2, b_aupr, width, label='Bound / Holo (Complex)', color='#1f77b4', alpha=0.85, edgecolor='black', lw=0.8)
ax1.set_ylabel('PR-AUC (AUPR)', fontsize=11, fontweight='bold')
ax1.set_title('A. Precision-Recall AUC (AUPR): Bound vs. Unbound', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(pair_labels, rotation=35, ha='right', fontsize=9)
ax1.set_ylim(0, 1.05)
ax1.legend(loc='upper left', frameon=True)
ax1.grid(True, linestyle='--', alpha=0.5)

# Panel B: AUROC Comparison
ax2 = axes[0, 1]
ax2.bar(x - width/2, u_auc, width, label='Unbound / Apo (Free)', color='#ff7f0e', alpha=0.85, edgecolor='black', lw=0.8)
ax2.bar(x + width/2, b_auc, width, label='Bound / Holo (Complex)', color='#2ca02c', alpha=0.85, edgecolor='black', lw=0.8)
ax2.axhline(0.5, color='gray', linestyle=':', label='Random Guessing (0.50)')
ax2.set_ylabel('ROC-AUC (AUROC)', fontsize=11, fontweight='bold')
ax2.set_title('B. ROC-AUC: Bound vs. Unbound', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(pair_labels, rotation=35, ha='right', fontsize=9)
ax2.set_ylim(0.3, 1.05)
ax2.legend(loc='lower right', frameon=True)
ax2.grid(True, linestyle='--', alpha=0.5)

# Panel C: Precision@10 Comparison
ax3 = axes[1, 0]
ax3.bar(x - width/2, u_p10, width, label='Unbound / Apo (Free)', color='#ff7f0e', alpha=0.85, edgecolor='black', lw=0.8)
ax3.bar(x + width/2, b_p10, width, label='Bound / Holo (Complex)', color='#9467bd', alpha=0.85, edgecolor='black', lw=0.8)
ax3.set_ylabel('Precision @ Top-10 (%)', fontsize=11, fontweight='bold')
ax3.set_title('C. Experimental Screening Purity (Precision @ Top-10)', fontsize=12, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(pair_labels, rotation=35, ha='right', fontsize=9)
ax3.set_ylim(0, 110)
ax3.legend(loc='upper left', frameon=True)
ax3.grid(True, linestyle='--', alpha=0.5)

# Panel D: Unbound Threshold Optimization Impact
ax4 = axes[1, 1]
ax4.bar(x - width/2, u_f1, width, label='Unbound at Fixed tau=0.33', color='#d62728', alpha=0.75, edgecolor='black', lw=0.8)
ax4.bar(x + width/2, u_opt_f1, width, label='Unbound at Adapted Optimal tau', color='#17becf', alpha=0.85, edgecolor='black', lw=0.8)
ax4.set_ylabel('F1-Score', fontsize=11, fontweight='bold')
ax4.set_title('D. Unbound Performance Recovery via Threshold Adaptation', fontsize=12, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(pair_labels, rotation=35, ha='right', fontsize=9)
ax4.set_ylim(0, 0.6)
ax4.legend(loc='upper left', frameon=True)
ax4.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
out_fig = 'results/case_study/side_by_side_9pair_comparison.png'
plt.savefig(out_fig, dpi=300)
plt.close()
print(f"Saved figure to {out_fig}")

# Copy to artifacts directory
art_dir = r'C:\Users\jeetu\.gemini\antigravity\brain\9cd13324-f55e-45a0-b757-58b9442b8b56'
shutil.copy2(out_fig, os.path.join(art_dir, 'side_by_side_9pair_comparison.png'))
print("Copied to artifact dir.")

