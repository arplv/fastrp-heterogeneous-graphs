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

        # --- Feature Generation (Pre-computation) ---
        # 1. Create the random projection matrix R' (on CPU)
        aa_degrees = relations['A']['A'].sum(axis=1).A.flatten()
        R_prime_scipy = self._create_random_projection_matrix(n_authors, dim, alpha, aa_degrees, s=s)

        # 2. For each meta-path, generate a sequence of feature matrices
        all_path_features = []
        for path_str in meta_paths:
            # 2a. Compute the full (N x N) sparse meta-path matrix with per-hop normalization
            M_norm = self._compute_meta_path_matrix(path_str, relations, beta)
            
            # 2b. Compute the feature matrix U = M_norm @ R'
            current_U = M_norm @ R_prime_scipy
            
            # Convert to dense tensor and normalize
            current_U_tensor = torch.from_numpy(current_U.toarray()).float()
            current_U_tensor = F.normalize(current_U_tensor, p=2, dim=1)
            
            path_features = [current_U_tensor]
            
            # 2c. Compute powers of the feature matrix: U_2 = M_norm @ U, U_3 = M_norm @ U_2 ...
            current_U_numpy = current_U.toarray()
            for _ in range(num_powers - 1):
                current_U_numpy = M_norm @ current_U_numpy
                
                current_U_tensor = torch.from_numpy(current_U_numpy).float()
                current_U_tensor = F.normalize(current_U_tensor, p=2, dim=1)
                
                path_features.append(current_U_tensor)
            
            all_path_features.append(torch.stack(path_features, dim=0))

        # 3. Store the pre-computed features and move to the target device ONCE.
        #    Shape: (num_meta_paths, num_powers, n_authors, dim)
        self.precomputed_features = torch.stack(all_path_features, dim=0).to(self.device)

        # --- Learnable Parameters (on target device) ---
        # Use a 2D weight matrix to distinguish between paths and powers
        self.feature_weights = nn.Parameter(torch.ones(self.precomputed_features.shape[0], self.precomputed_features.shape[1]))
        self.intercept = nn.Parameter(torch.tensor(0.0))

    def _create_random_projection_matrix(self, n_nodes, dim, alpha, degrees, s):
        """
        Creates a sparse random projection matrix R' = D^alpha @ R.
        R has `s` non-zero entries per column, scaled by 1/sqrt(s).
        """
        rng = np.random.default_rng(42)
        rows, cols, data = [], [], []
        for j in range(dim):
            indices = rng.choice(n_nodes, size=s, replace=False)
            # Scale values by 1/sqrt(s) for variance control
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

    def _compute_meta_path_matrix(self, path_str: str, relations: Dict[str, Dict[str, sp.csr_matrix]], beta: float):
        """
        Computes a single sparse meta-path matrix, applying degree normalization at each hop.
        Formula for a path v1->v2->v3: (D_v1^-beta @ M_v1v2) @ (D_v2^-beta @ M_v2v3)
        """
        print(f"  Computing features for meta-path: {path_str}")

        if '@' in path_str:
            raise NotImplementedError("Element-wise product of meta-paths ('@') is not supported in this version.")

        parts = list(path_str)
        
        # Initialize with the first normalized matrix in the path
        start_node, end_node = parts[0], parts[1]
        final_mat = relations[start_node][end_node].copy()

        if beta != 0:
            degrees = np.array(final_mat.sum(axis=1)).flatten()
            inv_degree = np.power(degrees, beta, where=degrees!=0) # Avoid 0^negative
            inv_degree = np.nan_to_num(inv_degree, copy=False, posinf=0.0, neginf=0.0)
            D_inv_beta = sp.diags(inv_degree)
            final_mat = D_inv_beta @ final_mat
        
        # Chain matrix multiplications for the rest of the path
        for i in range(1, len(parts) - 1):
            start_node, end_node = parts[i], parts[i+1]
            next_mat = relations[start_node][end_node].copy()

            if beta != 0:
                degrees = np.array(next_mat.sum(axis=1)).flatten()
                inv_degree = np.power(degrees, beta, where=degrees!=0)
                inv_degree = np.nan_to_num(inv_degree, copy=False, posinf=0.0, neginf=0.0)
                D_inv_beta = sp.diags(inv_degree)
                next_mat = D_inv_beta @ next_mat

            final_mat = final_mat @ next_mat
        
        return final_mat

    def get_embedding(self) -> torch.Tensor:
        """
        Computes the final embedding by taking a weighted average of all pre-computed features.
        The weights are learned during training.
        """
        # Softmax over the 'powers' dimension for each 'path'
        weights = F.softmax(self.feature_weights, dim=1)

        # Reshape weights for broadcasting: (num_paths, num_powers, 1, 1)
        weights_reshaped = weights.view(weights.shape[0], weights.shape[1], 1, 1)

        # Weighted sum. 'precomputed_features' is already on the correct device.
        # Sum across both the paths and powers dimensions to get the final embedding.
        embedding = (self.precomputed_features * weights_reshaped).sum(dim=(0, 1))
        
        return embedding

    def forward(self, idx_i: torch.Tensor, idx_j: torch.Tensor) -> torch.Tensor:
        emb = self.get_embedding()
        # Indices must be on the same device as the embedding matrix
        zi = emb[idx_i.to(self.device)]
        zj = emb[idx_j.to(self.device)]
        
        dist_sq = ((zi - zj) ** 2).sum(dim=1)
        
        # Scale distance by dimension to make it independent of embedding size
        scaled_dist = dist_sq / self.dim
        
        logits = self.intercept - scaled_dist
        return torch.sigmoid(logits) 