import numpy as np, pandas as pd, scipy.sparse as sp

def _post_pre_normalized(M):
    """M is scipy [pre, post]. Returns COO [post, pre], each post row summing to 1."""
    W = M.T.tocsr().astype(np.float32)
    s = np.asarray(W.sum(axis=1)).ravel()
    s[s == 0] = 1.0
    return (sp.diags(1.0 / s) @ W).tocoo()

def load_feedforward(data_dir="fly_project/data", adj_file="connectome_adj_agg.npz", verbose=True):
    n = pd.read_parquet(f"{data_dir}/neurons.parquet")
    A = sp.load_npz(f"{data_dir}/{adj_file}").tocsr()            # A[pre, post]
    idx = {g: n.loc[n.group == g, "matrix_idx"].values for g in ["PN", "KC", "MBON"]}
    hemi = n.set_index("matrix_idx")["hemisphere"]

    pk_all = A[idx["PN"]][:, idx["KC"]]                           # [PN, KC]
    keep = np.where(np.asarray(pk_all.sum(axis=1)).ravel() > 0)[0]  # PNs with >= 1 KC edge
    pn_idx = idx["PN"][keep]
    pk = pk_all[keep]
    kc_ok = np.where(np.asarray(pk.sum(axis=0)).ravel() > 0)[0]   # KCs that get PN input
    kc_idx = idx["KC"][kc_ok]
    pk = pk[:, kc_ok]
    km = A[kc_idx][:, idx["MBON"]]                                # [KC, MBON]

    out = {
        "pn_matrix_idx": pn_idx, "kc_matrix_idx": kc_idx, "mbon_matrix_idx": idx["MBON"],
        "n_pn": len(pn_idx), "n_kc": len(kc_idx), "n_mbon": len(idx["MBON"]),
        "pk": _post_pre_normalized(pk),                            # scipy COO [KC, PN]
        "km": _post_pre_normalized(km),                            # scipy COO [MBON, KC]
    }
    if verbose:
        for name, M, pre, post in [("PN->KC", pk, pn_idx, kc_idx), ("KC->MBON", km, kc_idx, idx["MBON"])]:
            c = M.tocoo()
            cross = (hemi[pre[c.row]].values != hemi[post[c.col]].values).mean()
            print(f"{name}: {c.nnz} edges, {100*cross:.1f}% cross-hemisphere")
        print(f"PNs used: {out['n_pn']} of {len(idx['PN'])}  KCs used: {out['n_kc']} of {len(idx['KC'])}  MBON: {out['n_mbon']}")
    return out

def to_torch(coo):
    import torch
    i = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long)
    v = torch.tensor(coo.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(i, v, coo.shape).coalesce()