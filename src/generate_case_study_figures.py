import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

out_dir = 'results/case_study'
art_dir = r'C:\Users\jeetu\.gemini\antigravity\brain\9cd13324-f55e-45a0-b757-58b9442b8b56'

df_res = pd.read_csv(os.path.join(out_dir, 'case_study_residue_predictions.csv'))
df_metrics = pd.read_csv(os.path.join(out_dir, 'case_study_metrics_summary.csv'))

colors = {
    '4fq0_B': '#1f77b4',
    '3uvj_A': '#2ca02c',
    '6kip_A': '#9467bd',
    '1t6e_X': '#ff7f0e'
}

# 1. Individual Profile Plots for each protein
for pid in ['4fq0_B', '3uvj_A', '6kip_A', '1t6e_X']:
    sub = df_res[df_res['protein'] == pid]
    m = df_metrics[df_metrics['protein'] == pid].iloc[0]
    
    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=300)
    
    # Prob line
    ax.plot(sub['residue_idx'], sub['predicted_prob'], color=colors[pid], lw=2, label='EGCPPIS Predicted Probability', zorder=2)
    
    # Fill area under probability
    ax.fill_between(sub['residue_idx'], 0, sub['predicted_prob'], color=colors[pid], alpha=0.15)
    
    # Ground truth interface points
    pos_sub = sub[sub['ground_truth'] == 1]
    ax.scatter(pos_sub['residue_idx'], pos_sub['predicted_prob'], color='#d62728', s=35, zorder=4, 
               label=f'Ground Truth Interface (N={len(pos_sub)})', edgecolors='black', linewidths=0.5)
    
    # Non-interface points (semi-transparent small dots)
    neg_sub = sub[sub['ground_truth'] == 0]
    ax.scatter(neg_sub['residue_idx'], neg_sub['predicted_prob'], color='gray', s=10, alpha=0.3, zorder=3,
               label='Non-Interface Residue')
    
    # Threshold line
    ax.axhline(0.33, color='black', linestyle='--', lw=1.2, label='Classification Threshold (0.33)', zorder=1)
    
    # Annotate Top 3 residues
    top3 = sub.sort_values('rank').head(3)
    for _, r in top3.iterrows():
        aa_label = str(r['amino_acid']) + str(int(r['residue_idx']))
        ax.annotate(aa_label,
                    (r['residue_idx'], r['predicted_prob']),
                    textcoords='offset points', xytext=(0, 10), ha='center',
                    fontsize=8, fontweight='bold', color='#111111',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7, edgecolor='none'))
    
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(1, len(sub))
    ax.set_xlabel('Residue Sequence Index (N- to C-terminus)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Interaction Probability', fontsize=11, fontweight='bold')
    ax.set_title(f'Case Study Prediction Profile: {pid} ({m["description"]})\n'
                 f'Length: {m["length"]} aa | True Interface: {m["n_pos"]} | ROC-AUC: {m["auc"]:.4f} | AUPR: {m["aupr"]:.4f} | MCC: {m["mcc_0.33"]:.4f} | P@10: {m["precision_at_10"]*100:.0f}%',
                 fontsize=12, fontweight='bold', pad=12)
    ax.legend(loc='upper right', frameon=True, fontsize=9)
    plt.tight_layout()
    
    save_path = os.path.join(out_dir, f'case_study_profile_{pid}.png')
    plt.savefig(save_path)
    plt.close()
    shutil.copy(save_path, os.path.join(art_dir, f'case_study_profile_{pid}.png'))

# 2. Master 4-Panel Comparison Figure
fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)

# (A) ROC Curves
ax_roc = axes[0, 0]
for pid in ['4fq0_B', '3uvj_A', '6kip_A', '1t6e_X']:
    sub = df_res[df_res['protein'] == pid]
    fpr, tpr, _ = roc_curve(sub['ground_truth'], sub['predicted_prob'])
    roc_val = auc(fpr, tpr)
    ax_roc.plot(fpr, tpr, color=colors[pid], lw=2.2, label=f'{pid} (AUC = {roc_val:.3f})')
ax_roc.plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.6, label='Random Chance (0.500)')
ax_roc.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold')
ax_roc.set_ylabel('True Positive Rate (Sensitivity)', fontsize=11, fontweight='bold')
ax_roc.set_title('(A) Receiver Operating Characteristic (ROC) Curves', fontsize=12, fontweight='bold')
ax_roc.legend(loc='lower right', frameon=True, fontsize=10)
ax_roc.grid(True, linestyle=':', alpha=0.6)

# (B) Precision-Recall Curves
ax_pr = axes[0, 1]
for pid in ['4fq0_B', '3uvj_A', '6kip_A', '1t6e_X']:
    sub = df_res[df_res['protein'] == pid]
    prec_curve, rec_curve, _ = precision_recall_curve(sub['ground_truth'], sub['predicted_prob'])
    aupr_val = average_precision_score(sub['ground_truth'], sub['predicted_prob'])
    prev = sub['ground_truth'].mean()
    ax_pr.plot(rec_curve, prec_curve, color=colors[pid], lw=2.2, label=f'{pid} (AUPR = {aupr_val:.3f}, Prev = {prev:.2f})')
ax_pr.set_xlabel('Recall (Sensitivity)', fontsize=11, fontweight='bold')
ax_pr.set_ylabel('Precision (Positive Predictive Value)', fontsize=11, fontweight='bold')
ax_pr.set_title('(B) Precision-Recall (PR) Curves', fontsize=12, fontweight='bold')
ax_pr.legend(loc='upper right', frameon=True, fontsize=10)
ax_pr.grid(True, linestyle=':', alpha=0.6)

# (C) Screening Budget Precision@K
ax_k = axes[1, 0]
k_cols = ['precision_at_5', 'precision_at_10', 'precision_at_20']
k_labels = ['Top-5', 'Top-10', 'Top-20']
x = np.arange(len(k_labels))
width = 0.18
prot_keys = ['4fq0_B', '3uvj_A', '6kip_A', '1t6e_X']

for idx, pid in enumerate(prot_keys):
    row = df_metrics[df_metrics['protein'] == pid].iloc[0]
    vals = [row[c] for c in k_cols]
    ax_k.bar(x + (idx - 1.5) * width, vals, width, label=pid, color=colors[pid], alpha=0.88, edgecolor='black', linewidth=0.5)

ax_k.set_xticks(x)
ax_k.set_xticklabels(k_labels, fontsize=11, fontweight='bold')
ax_k.set_ylabel('Candidate Precision @ K', fontsize=11, fontweight='bold')
ax_k.set_ylim(0, 1.1)
ax_k.set_title('(C) Experimental Screening Candidate Precision @ Budget K', fontsize=12, fontweight='bold')
ax_k.legend(loc='upper right', frameon=True, fontsize=10)
ax_k.grid(True, linestyle=':', alpha=0.6, axis='y')

# (D) Multi-metric Radar/Bar Summary
ax_m = axes[1, 1]
metrics_to_plot = ['acc_0.33', 'f1_0.33', 'mcc_0.33', 'auc', 'aupr']
metric_names = ['Accuracy', 'F1-Score', 'MCC', 'ROC-AUC', 'AUPR']
x_m = np.arange(len(metric_names))

for idx, pid in enumerate(prot_keys):
    row = df_metrics[df_metrics['protein'] == pid].iloc[0]
    vals = [row[c] for c in metrics_to_plot]
    ax_m.bar(x_m + (idx - 1.5) * width, vals, width, label=pid, color=colors[pid], alpha=0.88, edgecolor='black', linewidth=0.5)

ax_m.set_xticks(x_m)
ax_m.set_xticklabels(metric_names, fontsize=11, fontweight='bold')
ax_m.set_ylabel('Metric Value', fontsize=11, fontweight='bold')
ax_m.set_ylim(0, 1.1)
ax_m.set_title('(D) Comprehensive Benchmark Performance Metrics', fontsize=12, fontweight='bold')
ax_m.legend(loc='upper right', frameon=True, fontsize=10)
ax_m.grid(True, linestyle=':', alpha=0.6, axis='y')

plt.tight_layout()
summary_fig_path = os.path.join(out_dir, 'case_study_4protein_comparison.png')
plt.savefig(summary_fig_path)
plt.close()
shutil.copy(summary_fig_path, os.path.join(art_dir, 'case_study_4protein_comparison.png'))

print('All 5 case study figures created and copied to artifacts directory successfully.')

