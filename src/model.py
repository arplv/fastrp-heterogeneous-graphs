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
        self.n_authors = n_authors
        self.dim = dim
        self.meta_paths = meta_paths
        self.num_powers = num_powers
        self.beta = beta
        self.device = torch.device(device)

        # Store relations as sparse torch tensors on the specified device
        # We keep them on CPU because sparse-sparse matmul is not supported on CUDA
        self.relations = {
            k1: {k2: scipy_to_torch_sparse(m, 'cpu') for k2, m in v.items()}
            for k1, v in relations.items()
        }

        # Create the random projection matrix R' = D^alpha @ R
        aa_degrees = relations['A']['A'].sum(axis=1).A.flatten()
        s = int(np.sqrt(n_authors))
        if s == 0: s = 1

        rng = np.random.default_rng(42)
        rows, cols, data = [], [], [], []
        for j in range(dim):
            indices = rng.choice(n_authors, size=s, replace=False)
            values = rng.choice([1.0, -1.0], size=s)
            rows.extend(indices)
            cols.extend([j] * s)
            data.extend(values)
        R = sp.csr_matrix((data, (rows, cols)), shape=(n_authors, dim), dtype=np.float32)

        if alpha != 0:
            with np.errstate(divide='ignore'):
                D_alpha = sp.diags(np.power(aa_degrees, alpha))
            R_prime_scipy = D_alpha @ R
        else:
            R_prime_scipy = R
            
        self.R_prime = torch.from_numpy(R_prime_scipy.toarray()).float().to('cpu')

        num_features = len(self.meta_paths) * self.num_powers
        self.feature_weights = nn.Parameter(torch.ones(num_features))
        self.intercept = nn.Parameter(torch.tensor(0.0))

    def _calculate_feature_vector(self, path_str: str) -> torch.Tensor:
        """Dynamically computes the base feature vector M @ R for a given path."""
        M_norm = self._get_normalized_path_matrix(path_str)
        return torch.sparse.mm(M_norm, self.R_prime)

    def _get_normalized_path_matrix(self, path_str: str) -> sp.csr_matrix:
        """Parses a path string and computes the normalized adjacency matrix."""
        # This part remains on the CPU with sparse torch tensors
        if '@' in path_str:
            left_str, right_str = path_str.split('@')
            def get_matrix(s):
                if s.endswith('.T'):
                    mat_key = s[:-2]
                    return self.relations[mat_key[0]][mat_key[1]].t()
                else:
                    return self.relations[s[0]][s[1]]
            left_mat = get_matrix(left_str)
            right_mat = get_matrix(right_str)
            M = torch.sparse.mm(left_mat, right_mat.to_dense()).to_sparse()
        else:
            parts = list(path_str)
            M = self.relations[parts[0]][parts[1]]
            for i in range(1, len(parts) - 1):
                M = torch.sparse.mm(M, self.relations[parts[i]][parts[i+1]].to_dense()).to_sparse()

        # Degree normalization D^(-beta) * M
        with np.errstate(divide='ignore'):
            sum_dim = 1 if M.shape[0] == self.n_authors else 0
            degrees = torch.sparse.sum(M, dim=sum_dim).to_dense()
            inv_degree = np.power(degrees.cpu().numpy(), self.beta)
        inv_degree[np.isinf(inv_degree)] = 0
        D_inv_beta = torch.diag(torch.from_numpy(inv_degree).float()).to_sparse()
        
        M_norm = torch.sparse.mm(D_inv_beta, M.to_dense()).to_sparse()
        return M_norm.cpu()

    def get_embedding(self) -> torch.Tensor:
        all_features = []
        for path_str in self.meta_paths:
            # Iteratively compute projections: M@R, M@(M@R), ...
            # All computations are done on the fly on the CPU
            M_norm = self._get_normalized_path_matrix(path_str)
            current_U = torch.sparse.mm(M_norm, self.R_prime)
            current_U = F.normalize(current_U, p=2, dim=1)
            all_features.append(current_U)
            
            for _ in range(self.num_powers - 1):
                current_U = torch.sparse.mm(M_norm, current_U)
                current_U = F.normalize(current_U, p=2, dim=1)
                all_features.append(current_U)

        stacked_features = torch.stack(all_features, dim=0).to(self.device)
        weights = F.softmax(self.feature_weights, dim=0).to(self.device)
        
        embedding = torch.einsum('f,fad->ad', weights, stacked_features)
        return embedding

    def forward(self, idx_i: torch.Tensor, idx_j: torch.Tensor) -> torch.Tensor:
        emb = self.get_embedding()
        zi = emb[idx_i]
        zj = emb[idx_j]
        
        dist_sq = ((zi - zj) ** 2).sum(dim=1)
        logits = self.intercept.to(self.device) - dist_sq
        return torch.sigmoid(logits) 