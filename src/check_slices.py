import json
import numpy as np

with open('results/baseline/test60_protein_ids.json') as f:
    t60_ids = json.load(f)

with open('results/baseline/ubtest_protein_ids.json') as f:
    ub_ids = json.load(f)

# Let's verify by loading dataset sequences
def get_seqs(path):
    seqs = {}
    with open(path) as f:
        lines = f.readlines()
    for i in range(0, len(lines), 3):
        pid = lines[i].strip()[1:]
        seqs[pid] = lines[i+1].strip()
    return seqs

t60_seqs = get_seqs('Datasets/Test_60.fa')
ub_seqs = get_seqs('Datasets/GraphPPIS/UBtest_31-6.fa')

t60_prob = np.load('results/baseline/test60_y_prob.npy')
ub_prob = np.load('results/baseline/ubtest_y_prob.npy')

# Check slice lengths match
idx = 0
t60_slices = {}
for pid in t60_ids:
    L = len(t60_seqs[pid])
    t60_slices[pid] = (idx, idx + L)
    idx += L
print(f"t60 total matched: {idx} == {len(t60_prob)}")

idx = 0
ub_slices = {}
for pid in ub_ids:
    L = len(ub_seqs[pid])
    ub_slices[pid] = (idx, idx + L)
    idx += L
print(f"ub total matched: {idx} == {len(ub_prob)}")

