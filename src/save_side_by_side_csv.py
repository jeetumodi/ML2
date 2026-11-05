import json
import pandas as pd

with open('results/case_study/side_by_side_pairs_metrics.json') as f:
    data = json.load(f)

rows = []
for p in data:
    u = p['unbound_metrics']
    b = p['bound_metrics']
    rows.append({
        "pair_name": p['pair_name'],
        "unbound_id": p['unbound_id'],
        "bound_id": p['bound_id'],
        "unbound_len": u['len'],
        "bound_len": b['len'],
        "unbound_pos": u['n_pos'],
        "bound_pos": b['n_pos'],
        "unbound_prevalence": round(u['prevalence'], 4),
        "bound_prevalence": round(b['prevalence'], 4),
        "unbound_aupr": round(u['aupr'], 4),
        "bound_aupr": round(b['aupr'], 4),
        "delta_aupr": round(b['aupr'] - u['aupr'], 4),
        "unbound_auc": round(u['auc'], 4),
        "bound_auc": round(b['auc'], 4),
        "delta_auc": round(b['auc'] - u['auc'], 4),
        "unbound_f1_033": round(u['f1'], 4),
        "bound_f1_033": round(b['f1'], 4),
        "unbound_opt_f1": round(u['opt_f1'], 4),
        "unbound_opt_tau": round(u['opt_tau'], 2),
        "bound_opt_f1": round(b['opt_f1'], 4),
        "bound_opt_tau": round(b['opt_tau'], 2),
        "unbound_mcc_033": round(u['mcc'], 4),
        "bound_mcc_033": round(b['mcc'], 4),
        "delta_mcc": round(b['mcc'] - u['mcc'], 4),
        "unbound_p10": round(u['p@10'] * 100, 1),
        "bound_p10": round(b['p@10'] * 100, 1),
        "unbound_p5": round(u['p@5'] * 100, 1),
        "bound_p5": round(b['p@5'] * 100, 1),
        "unbound_p20": round(u['p@20'] * 100, 1),
        "bound_p20": round(b['p@20'] * 100, 1)
    })

df = pd.DataFrame(rows)
df.to_csv('results/case_study/side_by_side_pairs_summary.csv', index=False)
print("Saved results/case_study/side_by_side_pairs_summary.csv")
print(df[['pair_name', 'unbound_id', 'bound_id', 'unbound_aupr', 'bound_aupr', 'delta_aupr', 'unbound_p10', 'bound_p10']])

