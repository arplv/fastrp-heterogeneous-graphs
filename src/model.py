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

class FastRPModel(nn.Module):
    def __init__(self, n_authors: int, dim: int, meta_paths: Dict[str, sp.csr_matrix], num_powers: int, alpha: float = 0.0, beta: float = -0.5, device: str = 'cpu'):
        super().__init__()
        self.n_authors = n_authors
        self.dim = dim
        self.meta_path_names = list(meta_paths.keys())
        self.num_powers = num_powers
        self.device = torch.device(device)

        # 1. Pre-process and store meta-path matrices
        self.processed_matrices = {}
        for name, M in meta_paths.items():
            # Degree normalization: D^(-beta) * M
            with np.errstate(divide='ignore'):
                inv_degree = np.power(M.sum(axis=1).A.flatten(), beta) # beta is now the exponent directly
            inv_degree[np.isinf(inv_degree)] = 0
            D_inv_beta = sp.diags(inv_degree)
            M_norm = D_inv_beta @ M
            self.processed_matrices[name] = torch.from_numpy(M_norm.toarray()).float().to(self.device)

        # 2. Create the random projection matrix R' = D^alpha @ R
        # Use the degrees from the base Author-Author matrix for the alpha weighting
        aa_degrees = meta_paths['AAA'].sum(axis=1).A.flatten()
        s = int(np.sqrt(n_authors))
        if s == 0: s = 1
        R_prime_scipy = create_random_projection_matrix(n_authors, dim, alpha=alpha, degrees=aa_degrees, s=s)
        self.R_prime = torch.from_numpy(R_prime_scipy.toarray()).float().to(self.device)

        # 3. Learnable weights for combining all generated features
        # (num_meta_paths * num_powers)
        num_features = len(self.meta_path_names) * self.num_powers
        self.feature_weights = nn.Parameter(torch.ones(num_features, device=self.device))

        # 4. Barbera-style latent space model parameters
        self.intercept = nn.Parameter(torch.tensor(0.0, device=self.device))

    def get_embedding(self) -> torch.Tensor:
        """
        Computes the final embedding by generating features on the fly and combining them.
        """
        all_features = []
        
        for name in self.meta_path_names:
            M_norm = self.processed_matrices[name]
            
            # Iteratively compute projections: M@R, M@(M@R), ...
            # This is the key insight from the reference code.
            current_U = F.normalize(M_norm @ self.R_prime, p=2, dim=1)
            all_features.append(current_U)
            
            for _ in range(self.num_powers - 1):
                current_U = F.normalize(M_norm @ current_U, p=2, dim=1)
                all_features.append(current_U)

        # Stack all features: (num_features, n_authors, dim)
        stacked_features = torch.stack(all_features, dim=0)
        
        # Get learned weights, normalized with softmax
        weights = F.softmax(self.feature_weights, dim=0)
        
        # Weighted sum of all features
        embedding = torch.einsum('f,fad->ad', weights, stacked_features)
        return embedding

    def forward(self, idx_i: torch.Tensor, idx_j: torch.Tensor) -> torch.Tensor:
        """
        Predicts the probability of a link between nodes i and j.
        """
        emb = self.get_embedding()
        zi = emb[idx_i.to(self.device)]
        zj = emb[idx_j.to(self.device)]
        
        dist_sq = ((zi - zj) ** 2).sum(dim=1)
        logits = self.intercept - dist_sq
        return torch.sigmoid(logits) 