from neuprint import Client, fetch_neurons,fetch_all_rois, NeuronCriteria as NC
from dotenv import load_dotenv
import os

load_dotenv()
token=os.getenv("NEUPRINT_TOKEN")
client = Client('https://neuprint.janelia.org', dataset='male-cns:v1.0', token=token)

print("Connected to dataset:", client.dataset)
all_rois = fetch_all_rois()
print("Sample MB/AL ROIs:", [r for r in all_rois if any(k in r for k in ['AL(', 'CA(', 'PED(', 'MB(', 'aL(', 'bL(', 'gL('])])

queries = {
    "PNs (AL output)": NC(type='.*PN.*', rois=['AL(L)', 'AL(R)'], regex=True),
    "KCs": NC(type='KC.*', regex=True),
    "MBONs": NC(type='MBON.*', regex=True),
    "MB DANs": NC(type='(PAM|PPL1|PPL2).*', regex=True)
}

for group_name, criteria in queries.items():
    neurons_df, _ = fetch_neurons(criteria)
    valid_types = neurons_df.dropna(subset=['type'])
    type_counts = valid_types['type'].value_counts()
    
    print(f"\n{'='*55}")
    print(f"{group_name}: {len(type_counts)} unique types ({len(valid_types)} neurons)")
    print(f"{'='*55}")
    print(type_counts.to_string())


