import os
import re
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from dotenv import load_dotenv
from neuprint import Client, fetch_neurons, NeuronCriteria as NC, fetch_adjacencies

# 1. Load credentials securely from .env
load_dotenv()
token = os.getenv("NEUPRINT_TOKEN")
if not token:
    raise ValueError("NEUPRINT_TOKEN missing in .env file.")

client = Client('https://neuprint.janelia.org', dataset='male-cns:v1.0', token=token)
print(f"Connected to {client.dataset}...")

# 2. Fetch all 4 neuron groups across BOTH hemispheres
groups_queries = {
    "PN": NC(type='.*PN.*', regex=True),
    "KC": NC(type='KC.*', regex=True),
    "MBON": NC(type='MBON.*', regex=True),
    "DAN": NC(type='(PAM|PPL1|PPL2).*', regex=True)
}

neuron_dfs = []
for group_name, criteria in groups_queries.items():
    df, _ = fetch_neurons(criteria, client=client)
    df = df.dropna(subset=['type']).copy()
    df['group'] = group_name
    neuron_dfs.append(df)
    print(f"Fetched {group_name}: {len(df)} neurons")

# Combine and remove any duplicate bodyIds (keeping first assigned group)
neurons = pd.concat(neuron_dfs, ignore_index=True).drop_duplicates(subset=['bodyId']).reset_index(drop=True)

# 3. Extract Hemisphere ('L', 'R', or 'Unknown') from instance/type/roiInfo
def infer_hemisphere(row):
    # Check 'instance' or 'type' suffix like _L or _R or (L)/(R)
    for col in ['instance', 'type']:
        val = str(row.get(col, ''))
        if re.search(r'(_L|\(L\))$', val):
            return 'L'
        if re.search(r'(_R|\(R\))$', val):
            return 'R'
    # Fallback: check roiInfo string
    roi_str = str(row.get('roiInfo', ''))
    has_l = '(L)' in roi_str
    has_r = '(R)' in roi_str
    if has_l and not has_r:
        return 'L'
    if has_r and not has_l:
        return 'R'
    return 'Unknown'

neurons['hemisphere'] = neurons.apply(infer_hemisphere, axis=1)

# Assign contiguous 0-indexed matrix IDs
neurons['matrix_idx'] = np.arange(len(neurons))
neuron_table = neurons[['matrix_idx', 'bodyId', 'type', 'group', 'hemisphere']].rename(
    columns={'bodyId': 'id'}
)

print("\nNeuron Table Breakdown:")
print(neuron_table.groupby(['group', 'hemisphere']).size())

# 4. Fetch Synaptic Adjacencies across all extracted neurons
body_ids = neuron_table['id'].tolist()
print("\nFetching synaptic connections across all extracted neurons...")
_, conn_df = fetch_adjacencies(sources=body_ids, targets=body_ids, client=client)

# Apply minimum synapse weight threshold to drop weak noise connections
MIN_WEIGHT = 2
conn_df = conn_df[conn_df['weight'] >= MIN_WEIGHT].copy()
print(f"Filtered connections (weight >= {MIN_WEIGHT}): {len(conn_df)}")

# Map bodyId -> matrix_idx
id_to_idx = dict(zip(neuron_table['id'], neuron_table['matrix_idx']))
rows = conn_df['bodyId_pre'].map(id_to_idx).values
cols = conn_df['bodyId_post'].map(id_to_idx).values
weights = conn_df['weight'].values.astype(np.float32)

# 5. Build Sparse Adjacency Matrix (N x N)
num_neurons = len(neuron_table)
adj_csr = sp.csr_matrix((weights, (rows, cols)), shape=(num_neurons, num_neurons))

# 6. Save everything locally to fly_project/data/
save_dir = "fly_project/data"
os.makedirs(save_dir, exist_ok=True)

# Save neuron table
neuron_table.to_parquet(os.path.join(save_dir, "neurons.parquet"), index=False)
neuron_table.to_csv(os.path.join(save_dir, "neurons.csv"), index=False)
# Save SciPy sparse matrix (.npz)
sp.save_npz(os.path.join(save_dir, "connectome_adj.npz"), adj_csr)

# Also save PyTorch sparse COO tensor (.pt) directly for fast model loading
adj_coo = adj_csr.tocoo()
indices = torch.tensor(np.vstack((adj_coo.row, adj_coo.col)), dtype=torch.long)
values = torch.tensor(adj_coo.data, dtype=torch.float32)
torch_sparse_adj = torch.sparse_coo_tensor(indices, values, size=(num_neurons, num_neurons)).coalesce()
torch.save(torch_sparse_adj, os.path.join(save_dir, "connectome_adj.pt"))

print(f"\nSaved successfully to {save_dir}/:")
print(f"  - neurons.parquet ({len(neuron_table)} rows)")
print(f"  - connectome_adj.npz (shape: {adj_csr.shape}, nnz: {adj_csr.nnz})")
print(f"  - connectome_adj.pt (PyTorch sparse COO tensor ready for training)")