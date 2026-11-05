import pandas as pd

df = pd.read_csv('results/case_study/case_study_residue_predictions.csv')
for prot in ['4fq0_B', '3uvj_A', '6kip_A', '1t6e_X']:
    pdf = df[df['protein'] == prot].sort_values('predicted_prob', ascending=False)
    print(f'=== {prot} Top 20 ===')
    for i, (_, row) in enumerate(pdf.head(20).iterrows(), 1):
        print(f"{i:2d} | pos={int(row['residue_idx']):3d} | aa={row['amino_acid']} | prob={row['predicted_prob']:.4f} | gt={int(row['ground_truth'])} | status={row['classification']}")

