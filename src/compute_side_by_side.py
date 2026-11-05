import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, f1_score, matthews_corrcoef

pairs = [
    ('1ci5_A', '1qa9_B', 'CheY protein (chemotaxis response regulator)'),
    ('1lni_A', '1ay7_A', 'Barnase (ribonuclease)'),
    ('1ysc_A', '1wpx_A', 'Carboxypeptidase Y'),
    ('4fh4_A', '2g2u_A', 'Profilin (actin-binding protein)'),
    ('1om0_A', '1ta3_A', 'Ran GTPase'),
    ('1czp_A', '1ewy_C', 'Subtilisin Inhibitor'),
    ('1t6e_X', '2b42_A', 'TAXI-I (Wheat xylanase inhibitor)'),
    ('1shu_X', '1t6b_Y', 'FliM (flagellar switch rotor)'),
    ('2v77_A', '3fju_A', 'PTPRD phosphatase')
]

with open('results/baseline/test60_protein_ids.json') as f:
    t60_ids = json.load(f)
with open('results/baseline/ubtest_protein_ids.json') as f:
    ub_ids = json.load(f)

def load_fasta(path):
    seqs, labels = {}, {}
    with open(path) as f:
        lines = f.readlines()
    for i in range(0, len(lines), 3):
        pid = lines[i].strip()[1:]
        seq = lines[i+1].strip()
        lbl_str = lines[i+2].strip()
        seqs[pid] = seq
        labels[pid] = np.array([int(c) for c in lbl_str])
    return seqs, labels

t60_seqs, t60_labels = load_fasta('Datasets/Test_60.fa')
ub_seqs, ub_labels = load_fasta('Datasets/GraphPPIS/UBtest_31-6.fa')

t60_prob = np.load('results/baseline/test60_y_prob.npy')
ub_prob = np.load('results/baseline/ubtest_y_prob.npy')

# Compute slices
def get_slices(pids, seqs):
    slices = {}
    idx = 0
    for pid in pids:
        L = len(seqs[pid])
        slices[pid] = (idx, idx + L)
        idx += L
    return slices

t60_slices = get_slices(t60_ids, t60_seqs)
ub_slices = get_slices(ub_ids, ub_seqs)

def evaluate_pred(prob, true, thresh=0.33):
    pred = (prob >= thresh).astype(int)
    auc = roc_auc_score(true, prob) if len(np.unique(true)) > 1 else 0.5
    aupr = average_precision_score(true, prob) if len(np.unique(true)) > 1 else 0.0
    f1 = f1_score(true, pred, zero_division=0)
    mcc = matthews_corrcoef(true, pred) if len(np.unique(pred)) > 1 and len(np.unique(true)) > 1 else 0.0
    prec = np.sum((pred == 1) & (true == 1)) / max(1, np.sum(pred == 1))
    rec = np.sum((pred == 1) & (true == 1)) / max(1, np.sum(true == 1))
    acc = np.mean(pred == true)
    tn, fp, fn, tp = confusion_matrix(true, pred, labels=[0, 1]).ravel()
    
    # Precision@K
    ranked = np.argsort(-prob)
    p_at_5 = np.mean(true[ranked[:min(5, len(true))]])
    p_at_10 = np.mean(true[ranked[:min(10, len(true))]])
    p_at_20 = np.mean(true[ranked[:min(20, len(true))]])
    
    # Optimal threshold for F1
    best_f1, best_tau = 0.0, 0.33
    for tau in np.linspace(0.05, 0.95, 91):
        p_tau = (prob >= tau).astype(int)
        score = f1_score(true, p_tau, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_tau = tau
            
    return {
        "len": len(true), "n_pos": int(np.sum(true)), "prevalence": float(np.mean(true)),
        "auc": float(auc), "aupr": float(aupr), "f1": float(f1), "mcc": float(mcc),
        "prec": float(prec), "rec": float(rec), "acc": float(acc),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "p@5": float(p_at_5), "p@10": float(p_at_10), "p@20": float(p_at_20),
        "opt_tau": float(best_tau), "opt_f1": float(best_f1)
    }

results = []
for u, b, name in pairs:
    u_id = u.replace('_', '')
    b_id = b.replace('_', '')
    
    u_s, u_e = ub_slices[u_id]
    b_s, b_e = t60_slices[b_id]
    
    u_p = ub_prob[u_s:u_e]
    u_y = ub_labels[u_id]
    
    b_p = t60_prob[b_s:b_e]
    b_y = t60_labels[b_id]
    
    u_m = evaluate_pred(u_p, u_y)
    b_m = evaluate_pred(b_p, b_y)
    
    results.append({
        "pair_name": name,
        "unbound_id": u, "unbound_internal": u_id,
        "bound_id": b, "bound_internal": b_id,
        "unbound_metrics": u_m,
        "bound_metrics": b_m
    })

print(json.dumps(results, indent=2))

with open('results/case_study/side_by_side_pairs_metrics.json', 'w') as f:
    json.dump(results, f, indent=2)

