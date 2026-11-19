import numpy as np, pandas as pd, scipy.sparse as sp

D = "fly_project/data/"
neurons = pd.read_parquet(D + "neurons.parquet")
edges = pd.read_parquet(D + "subgraph_edges.parquet")   # aggregated per pair, weight >= 2

id2idx = dict(zip(neurons["id"], neurons["matrix_idx"]))
edges = edges[edges.bodyId_pre.isin(id2idx) & edges.bodyId_post.isin(id2idx)]
rows = edges.bodyId_pre.map(id2idx).values
cols = edges.bodyId_post.map(id2idx).values
A = sp.csr_matrix((edges.weight.values.astype(np.float32), (rows, cols)),
                  shape=(len(neurons), len(neurons)))
sp.save_npz(D + "connectome_adj_agg.npz", A)
print("pairs kept:", A.nnz)