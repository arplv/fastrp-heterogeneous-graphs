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
    def __init__(self, n_authors: int, dim: int, meta_paths: List[str], relations: Dict[str, Dict[str, sp.csr_matrix]], num_powers: int, alpha: float, beta: float, device: str = 'cpu'):
        super().__init__()
        self.device = torch.device(device)

        # --- Feature Generation (Pre-computation) ---
        # This is done once at initialization. All heavy computation happens here.
        
        # 1. Create the random projection matrix R' (on CPU)
        aa_degrees = relations['A']['A'].sum(axis=1).A.flatten()
        R_prime_scipy = self._create_random_projection_matrix(n_authors, dim, alpha, aa_degrees)

        # 2. For each meta-path, generate a sequence of feature matrices
        all_path_features = []
        for path_str in meta_paths:
            # 2a. Compute the full (N x N) sparse meta-path matrix using scipy
            M = self._compute_meta_path_matrix(path_str, relations)
            
            # 2b. Apply degree normalization D^(-beta) * M
            with np.errstate(divide='ignore'):
                inv_degree = np.power(M.sum(axis=1).A.flatten(), beta)
            inv_degree[np.isinf(inv_degree)] = 0
            D_inv_beta = sp.diags(inv_degree)
            M_norm = D_inv_beta @ M

            # 2c. Compute the feature matrix U = M_norm @ R'
            # This results in a dense matrix of shape (n_authors, dim)
            current_U = M_norm @ R_prime_scipy
            
            # Convert to dense tensor and normalize
            current_U_tensor = torch.from_numpy(current_U.toarray()).float()
            current_U_tensor = F.normalize(current_U_tensor, p=2, dim=1)
            
            path_features = [current_U_tensor]
            
            # 2c. Compute powers of the feature matrix: U_2 = M_norm @ U, U_3 = M_norm @ U_2 ...
            # Note: current_U is a numpy array for the loop
            current_U_numpy = current_U.toarray()
            for _ in range(num_powers - 1):
                # This mat-mat product is the expensive step for powers
                current_U_numpy = M_norm @ current_U_numpy
                
                # Convert to tensor and normalize
                current_U_tensor = torch.from_numpy(current_U_numpy).float()
                current_U_tensor = F.normalize(current_U_tensor, p=2, dim=1)
                
                path_features.append(current_U_tensor)
            
            all_path_features.append(torch.stack(path_features, dim=0))

        # 3. Store the pre-computed features (on CPU for now)
        #    Shape: (num_meta_paths, num_powers, n_authors, dim)
        self.precomputed_features = torch.stack(all_path_features, dim=0)

        # --- Learnable Parameters (on target device) ---
        self.feature_weights = nn.Parameter(torch.ones(self.precomputed_features.shape[0], self.precomputed_features.shape[1]))
        self.intercept = nn.Parameter(torch.tensor(0.0))

    def _create_random_projection_matrix(self, n_nodes, dim, alpha, degrees):
        s = int(np.sqrt(n_nodes))
        if s == 0: s = 1
        rng = np.random.default_rng(42)
        rows, cols, data = [], [], []
        for j in range(dim):
            indices = rng.choice(n_nodes, size=s, replace=False)
            values = rng.choice([1.0, -1.0], size=s)
            rows.extend(indices)
            cols.extend([j] * s)
            data.extend(values)
        R = sp.csr_matrix((data, (rows, cols)), shape=(n_nodes, dim), dtype=np.float32)

        if alpha != 0:
            with np.errstate(divide='ignore'):
                D_alpha = sp.diags(np.power(degrees, alpha))
            return D_alpha @ R
        return R

    def _compute_meta_path_matrix(self, path_str: str, relations: Dict[str, Dict[str, sp.csr_matrix]]):
        """Computes a single sparse meta-path matrix using scipy."""
        print(f"  Computing features for meta-path: {path_str}")
        if '@' in path_str:
            left_str, right_str = path_str.split('@')
            def get_matrix(s):
                if s.endswith('.T'):
                    mat_key = s[:-2]
                    return relations[mat_key[0]][mat_key[1]].T
                else:
                    return relations[s[0]][s[1]]
            left_mat = get_matrix(left_str)
            right_mat = get_matrix(right_str)
            return left_mat @ right_mat
        else:
            parts = list(path_str)
            final_mat = relations[parts[0]][parts[1]]
            for i in range(1, len(parts) - 1):
                final_mat = final_mat @ relations[parts[i]][parts[i+1]]
            return final_mat

    def get_embedding(self) -> torch.Tensor:
        """Computes the final embedding using the pre-computed features."""
        weights = F.softmax(self.feature_weights, dim=0)
        embedding = torch.einsum('f,fad->ad', weights, self.precomputed_features)
        return embedding

    def forward(self, idx_i: torch.Tensor, idx_j: torch.Tensor) -> torch.Tensor:
        emb = self.get_embedding()
        # Move indices to the model's device
        zi = emb[idx_i.to(self.device)]
        zj = emb[idx_j.to(self.device)]
        
        dist_sq = ((zi - zj) ** 2).sum(dim=1)
        logits = self.intercept - dist_sq
        return torch.sigmoid(logits) 