import torch

def kwta(x, k):
    """x: [B, n], non-negative. Keeps exactly the top-k per row."""
    idx = torch.topk(x, k, dim=-1).indices
    return x * torch.zeros_like(x).scatter_(-1, idx, 1.0)

def propagate(W, x):            # W sparse [n_out, n_in], x dense [B, n_in]
    return torch.sparse.mm(W, x.T).T