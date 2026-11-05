import json
import numpy as np
import pandas as pd

with open('results/baseline/test60_protein_ids.json') as f:
    t60_ids = json.load(f)

with open('results/baseline/ubtest_protein_ids.json') as f:
    ub_ids = json.load(f)

print(f"Total test60 proteins: {len(t60_ids)}")
print(f"Total ubtest proteins: {len(ub_ids)}")

t60_prob = np.load('results/baseline/test60_y_prob.npy')
t60_true = np.load('results/baseline/test60_y_true.npy')
ub_prob = np.load('results/baseline/ubtest_y_prob.npy')
ub_true = np.load('results/baseline/ubtest_y_true.npy')

print(f"t60_prob shape: {t60_prob.shape}, t60_true shape: {t60_true.shape}")
print(f"ub_prob shape: {ub_prob.shape}, ub_true shape: {ub_true.shape}")

# Let's inspect Datasets/Test_60.fa and Datasets/GraphPPIS/UBtest_31-6.fa sequence lengths
def load_fasta_lens(path):
    lens = {}
    with open(path) as f:
        lines = f.readlines()
    for i in range(0, len(lines), 3):
        pid = lines[i].strip()[1:]
        seq = lines[i+1].strip()
        lens[pid] = len(seq)
    return lens

t60_lens = load_fasta_lens('Datasets/Test_60.fa')
ub_lens = load_fasta_lens('Datasets/GraphPPIS/UBtest_31-6.fa')

print("Loaded lengths successfully.")

