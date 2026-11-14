import os
from dotenv import load_dotenv
from neuprint import Client, fetch_neurons, NeuronCriteria as NC, fetch_adjacencies
import pandas as pd

load_dotenv()
token = os.getenv("NEUPRINT_TOKEN")
client = Client('https://neuprint.janelia.org', dataset='male-cns:v1.0', token=token)

print("Fetching neuron groups for BOTH hemispheres...")

# 1. Fetch neurons across both hemispheres (removing single-hemisphere ROI restrictions)
pn_df, _ = fetch_neurons(NC(type='.*PN.*', regex=True))
kc_df, _ = fetch_neurons(NC(type='KC.*', regex=True))
mbon_df, _ = fetch_neurons(NC(type='MBON.*', regex=True))
dan_df, _ = fetch_neurons(NC(type='(PAM|PPL1|PPL2).*', regex=True))

print(f"Fetched (Both Hemispheres) -> PNs: {len(pn_df)}, KCs: {len(kc_df)}, MBONs: {len(mbon_df)}, DANs: {len(dan_df)}")

pn_df['group']   = 'PN'
kc_df['group']   = 'KC'
mbon_df['group'] = 'MBON'
dan_df['group']  = 'DAN'

# Combine and drop any accidental duplicates by bodyId
all_neurons = pd.concat([pn_df, kc_df, mbon_df, dan_df], ignore_index=True).drop_duplicates(subset=['bodyId'])

print("Fetched right-hemisphere neurons (by ROI innervation):")
print(all_neurons['group'].value_counts())

body_ids = all_neurons['bodyId'].tolist()

print("\nFetching synaptic connections between neurons...")
neuron_df, conn_df = fetch_adjacencies(sources=body_ids, targets=body_ids, client=client)
adj_df = conn_df.groupby(['bodyId_pre', 'bodyId_post'], as_index=False)['weight'].sum()

print(f"Total unique neuron-to-neuron connections found: {len(adj_df)}")

# No threshold yet (min_weight = 1)
min_weight = 2
filtered_adj = adj_df[adj_df['weight'] >= min_weight].copy()
print(f"Connections after filtering (weight >= {min_weight}): {len(filtered_adj)}")

os.makedirs("fly_project/data", exist_ok=True)
filtered_adj.to_parquet("fly_project/data/subgraph_edges.parquet", index=False)
all_neurons.to_parquet("fly_project/data/subgraph_neurons.parquet", index=False)
print("Subgraph with DANs saved successfully to fly_project/data/!")