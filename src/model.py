import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from typing import List, Dict

def create_random_projection_matrix(n_nodes: int, dim: int, alpha: float, degrees: np.ndarray, s: int = 1, seed: int = 42) -> sp.csr_matrix:
    """
    Creates a sparse random projection matrix, with degree weighting (alpha).
    R' = D^alpha @ R
    """
    rng = np.random.default_rng(seed)
    
    # Create the base random matrix R
    rows, cols, data = [], [], []
    for j in range(dim):
        indices = rng.choice(n_nodes, size=s, replace=False)
        values = rng.choice([1.0, -1.0], size=s) # Simplified values, normalization happens later
        rows.extend(indices)
        cols.extend([j] * s)
        data.extend(values)
    R = sp.csr_matrix((data, (rows, cols)), shape=(n_nodes, dim), dtype=np.float32)

    # Apply alpha weighting: D^alpha
    if alpha != 0:
        with np.errstate(divide='ignore'):
            # Calculate D^alpha
            D_alpha = sp.diags(np.power(degrees, alpha))
        # R' = D^alpha @ R
        return D_alpha @ R
    
    return R

def scipy_to_torch_sparse(matrix: sp.csr_matrix, device: torch.device) -> torch.Tensor:
    """Converts a Scipy sparse matrix to a PyTorch sparse COO tensor."""
    coo = matrix.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape
    return torch.sparse_coo_tensor(i, v, torch.Size(shape), device=device)

class FastRPModel(nn.Module):
    def __init__(self, n_authors: int, dim: int, meta_paths: List[str], relations: Dict[str, Dict[str, sp.csr_matrix]], num_powers: int, alpha: float, beta: float, s: int, device: str = 'cpu'):
        super().__init__()
        self.device = torch.device(device)
        self.dim = dim
        self.feature_weights = nn.Parameter(torch.ones(len(meta_paths), num_powers))
        self.intercept = nn.Parameter(torch.tensor(0.0))
        self.slope = nn.Parameter(torch.tensor(1.0))
        
        # --- Feature Generation (Efficient Pre-computation) ---
        # Create the initial random projection matrix.
        R_prime = self._create_random_projection_matrix(
            n_authors, dim, alpha, relations['A']['A'].sum(axis=1).A.flatten(), s=s
        )

        features_list = []
        for path_str in meta_paths:
            print(f"  Computing features for meta-path: {path_str}")
            if '@' in path_str:
                raise NotImplementedError("Element-wise product of meta-paths ('@') is not supported in this version.")
            
            # Deconstruct the path into a sequence of normalized hop matrices (M_1, M_2, ... M_k).
            path_parts = list(path_str)
            hop_matrices = []
            for i in range(len(path_parts) - 1):
                matrix = self._hop(relations[path_parts[i]][path_parts[i+1]].copy(), beta)
                hop_matrices.append(matrix)

            # Efficiently compute features for each power of the meta-path matrix M.
            # We compute U_i = M^i @ R' by iteratively applying M.
            # U_i = M @ U_{i-1}, where U_0 = R'.
            # The multiplication M @ U is done efficiently as (M_1 @ ... @ M_k) @ U,
            # applying matrices from right-to-left to avoid forming the large (N,N) M matrix.
            
            # U_prev_sparse stores the sparse result of the previous power, starting with R'.
            U_prev_sparse = R_prime
            
            for _ in range(num_powers):
                # Calculate U_curr = M @ U_prev by applying hop matrices in reverse order.
                U_curr_sparse = U_prev_sparse
                for hop_matrix in reversed(hop_matrices):
                    U_curr_sparse = hop_matrix @ U_curr_sparse
                
                # Convert the sparse result to a dense, normalized tensor for the model.
                # The .toarray() call is memory-intensive but necessary here. The key efficiency
                # gain is avoiding the (N, N) matrix construction and multiplication.
                U_tensor = torch.from_numpy(U_curr_sparse.toarray()).float()
                U_tensor = F.normalize(U_tensor, p=2, dim=1)
                features_list.append(U_tensor)
                
                # The next iteration will build upon the sparse result of this one.
                U_prev_sparse = U_curr_sparse
        
        # Shape: (F, N, D) where F = num_paths * num_powers
        self.features = torch.stack(features_list, dim=0).to(self.device)

    def _create_random_projection_matrix(self, n_nodes, dim, alpha, degrees, s):
        rng = np.random.default_rng(42)
        rows, cols, data = [], [], []
        for j in range(dim):
            indices = rng.choice(n_nodes, size=s, replace=False)
            values = rng.choice([1.0, -1.0], size=s) / np.sqrt(s)
            rows.extend(indices)
            cols.extend([j] * s)
            data.extend(values)
        R = sp.csr_matrix((data, (rows, cols)), shape=(n_nodes, dim), dtype=np.float32)

        if alpha != 0:
            with np.errstate(divide='ignore'):
                D_alpha = sp.diags(np.power(degrees, alpha))
            return D_alpha @ R
        return R

    def _hop(self, mat: sp.csr_matrix, beta: float) -> sp.csr_matrix:
        """Applies degree normalization to a sparse matrix for one hop."""
        if beta == 0:
            return mat
        
        degrees = np.array(mat.sum(axis=1)).flatten()
        # Use `where` to avoid division by zero or 0^negative
        inv_degree = np.power(degrees, beta, where=degrees!=0) 
        # Clean up any potential inf/nan values
        inv_degree = np.nan_to_num(inv_degree, copy=False, posinf=0.0, neginf=0.0)
        
        normalizer = sp.diags(inv_degree)
        return normalizer @ mat

    def _mixed_embedding(self):
        # weights: (n_paths, n_powers) -> flatten -> (F)
        w = F.softmax(self.feature_weights.flatten(), dim=0)
        # einsum: w_f * X_fnd -> nd
        return torch.einsum('f,fnd->nd', w, self.features)

    def forward(self, idx_i: torch.Tensor, idx_j: torch.Tensor) -> torch.Tensor:
        E = self._mixed_embedding() # (N, D) - differentiable!
        zi = E[idx_i]
        zj = E[idx_j]
        
        dist_sq = ((zi - zj) ** 2).sum(dim=1)
        
        logits = self.intercept - self.slope * dist_sq
        return logits 