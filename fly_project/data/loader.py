import os
import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp

class ConnectomeLoader:
    def __init__(self, data_dir="fly_project/data", target_kc_sparsity=0.05):
        self.data_dir = data_dir
        self.target_kc_sparsity = target_kc_sparsity
        
        # 1. Load neuron table and sparse adjacency matrix from existing data folder
        self.neurons_df = pd.read_parquet(os.path.join(data_dir, "neurons.parquet"))
        self.adj_csr = sp.load_npz(os.path.join(data_dir, "connectome_adj.npz"))
        
        # Verify counts match
        assert len(self.neurons_df) == self.adj_csr.shape[0], "Neuron table and matrix shape mismatch!"
        
        # 2. Identify index ranges per group
        self.group_indices = {}
        for group in ['PN', 'KC', 'MBON', 'DAN']:
            indices = self.neurons_df[self.neurons_df['group'] == group]['matrix_idx'].values
            self.group_indices[group] = indices
            print(f"Group {group}: {len(indices)} neurons")
            
        # 3. Check hemisphere crossing
        self._analyze_hemispheres()
        
        # 4. Normalize weights per postsynaptic neuron (in-degree sum normalization)
        self.normalized_adj_csr = self._normalize_weights(self.adj_csr.copy())
        
        # 5. Convert to PyTorch sparse COO tensor for fast execution
        self.torch_adj = self._to_torch_sparse(self.normalized_adj_csr)
        
    def _analyze_hemispheres(self):
        idx_to_hemi = dict(zip(self.neurons_df['matrix_idx'], self.neurons_df['hemisphere']))
        coo = self.adj_csr.tocoo()
        
        cross_count = 0
        total_edges = len(coo.data)
        for pre, post in zip(coo.row, coo.col):
            h_pre = idx_to_hemi.get(pre, 'Unknown')
            h_post = idx_to_hemi.get(post, 'Unknown')
            if h_pre != 'Unknown' and h_post != 'Unknown' and h_pre != h_post:
                cross_count += 1
                
        print(f"Hemisphere analysis: {cross_count} / {total_edges} edges cross hemispheres.")

    def _normalize_weights(self, csr_mat):
        coo = csr_mat.tocoo()
        post_sums = np.array(csr_mat.sum(axis=0)).flatten()
        post_sums[post_sums == 0.0] = 1.0
        coo.data = coo.data / post_sums[coo.col]
        return coo.tocsr()

    def _to_torch_sparse(self, csr_mat):
        coo = csr_mat.tocoo()
        indices = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long)
        values = torch.tensor(coo.data, dtype=torch.float32)
        shape = torch.Size(csr_mat.shape)
        return torch.sparse_coo_tensor(indices, values, shape).coalesce()

    def apply_k_wta(self, kc_activations):
        """
        Applies k-Winners-Take-All (k-WTA) sparsity to KC activations 
        so approximately target_kc_sparsity (e.g., 5%) fire.
        """
        kc_indices = self.group_indices['KC']
        kcs = kc_activations[kc_indices]
        
        k = int(np.ceil(len(kc_indices) * self.target_kc_sparsity))
        if k < len(kcs):
            threshold_idx = np.argsort(kcs)[-k]
            threshold = kcs[threshold_idx]
            kcs_sparse = np.where(kcs >= threshold, kcs, 0.0)
        else:
            kcs_sparse = kcs
            
        kc_activations[kc_indices] = kcs_sparse
        return kc_activations