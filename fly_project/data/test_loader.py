import numpy as np
from loader import ConnectomeLoader

def test_connectome_loader():
    try:
        loader = ConnectomeLoader()
    except FileNotFoundError:
        print("[SKIP] Cached connectome files not found.")
        return
        
    assert loader.torch_adj.is_sparse, "Adjacency matrix must be a PyTorch sparse tensor!"
    
    dummy_activations = np.random.uniform(0.1, 1.0, size=(len(loader.neurons_df),))
    sparse_activations = loader.apply_k_wta(dummy_activations)
    
    kc_indices = loader.group_indices['KC']
    active_kcs = np.sum(sparse_activations[kc_indices] > 0.0)
    actual_sparsity = active_kcs / len(kc_indices)
    
    print(f"Target KC sparsity: {loader.target_kc_sparsity * 100}%")
    print(f"Actual KC sparsity after k-WTA: {actual_sparsity * 100:.2f}%")
    assert abs(actual_sparsity - loader.target_kc_sparsity) < 0.02, "k-WTA sparsity deviates too much!"
    print("[PASS] Connectome Loader tests passed successfully!")

if __name__ == "__main__":
    test_connectome_loader()