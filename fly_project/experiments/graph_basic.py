import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp

# 1. Load cached data
data_dir = "fly_project/data"
neurons_df = pd.read_parquet(os.path.join(data_dir, "neurons.parquet"))
adj_csr = sp.load_npz(os.path.join(data_dir, "connectome_adj.npz"))

print(
    f"Loaded {len(neurons_df)} neurons and adjacency matrix of shape {adj_csr.shape}"
)

# 2. Map matrix index to group
idx_to_group = dict(zip(neurons_df["matrix_idx"], neurons_df["group"]))

# 3. Aggregate edges at the group-to-group level
group_edges = {}
coo = adj_csr.tocoo()
for pre_idx, post_idx, weight in zip(coo.row, coo.col, coo.data):
  pre_group = idx_to_group.get(pre_idx, "Unknown")
  post_group = idx_to_group.get(post_idx, "Unknown")

  key = (pre_group, post_group)
  if key not in group_edges:
    group_edges[key] = {"total_weight": 0.0, "count": 0}

  group_edges[key]["total_weight"] += float(weight)
  group_edges[key]["count"] += 1

# 4. Build a directed graph for group-level flow
G = nx.DiGraph()
groups = ["PN", "KC", "MBON", "DAN"]
for g in groups:
  count = len(neurons_df[neurons_df["group"] == g])
  G.add_node(g, size=count)

for (pre_g, post_g), stats in group_edges.items():
  if pre_g in groups and post_g in groups:
    mean_weight = stats["total_weight"] / stats["count"]
    G.add_edge(
        pre_g,
        post_g,
        weight=mean_weight,
        count=stats["count"],
    )

# 5. Plot the Group-Level Macro Graph (Fixed figsize)
plt.figure(figsize=(10, 7))

pos = {
    "PN": np.array([0.0, 0.5]),
    "KC": np.array([1.0, 0.8]),
    "DAN": np.array([1.0, 0.2]),
    "MBON": np.array([2.0, 0.5]),
}

node_sizes = [
    3000 + len(neurons_df[neurons_df["group"] == g]) * 5 for g in groups
]
nx.draw_networkx_nodes(
    G, pos, nodelist=groups, node_size=node_sizes, node_color="lightblue"
)
nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold")

edges = G.edges(data=True)
weights = [max(0.5, data["weight"] / 5.0) for _, _, data in edges]
nx.draw_networkx_edges(
    G,
    pos,
    edgelist=[(u, v) for u, v, _ in edges],
    width=weights,
    arrowstyle="->",
    arrowsize=20,
    edge_color="gray",
    connectionstyle="arc3,rad=0.1",
)

edge_labels = {(u, v): f"avg w:{d['weight']:.1f}" for u, v, d in edges}
nx.draw_networkx_edge_labels(
    G, pos, edge_labels=edge_labels, font_size=9, label_pos=0.5
)

plt.title(
    "MaleCNS Group-Level Connectome Flow (PN → KC → MBON ← DAN)",
    fontsize=12,
    fontweight="bold",
)
plt.axis("off")
plt.tight_layout()

os.makedirs("fly_project/logs", exist_ok=True)
plt.savefig(
    "fly_project/logs/group_connectome_graph.png", dpi=300, bbox_inches="tight"
)
print("Saved group-level graph to fly_project/logs/group_connectome_graph.png")
plt.show()