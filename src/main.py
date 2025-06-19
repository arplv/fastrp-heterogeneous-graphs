import argparse
import numpy as np
import scipy.sparse as sp
import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from pathlib import Path
from tqdm import tqdm
from numba import njit

from data_loader import load_data
from model import FastRPModel

@njit
def numba_spgemm(a_indptr, a_indices, a_data, b_indptr, b_indices, b_data, n):
    """
    Performs a memory-efficient sparse-sparse matrix multiplication (SpGEMM) using Numba.
    This version runs in serial to avoid parallelism-related memory issues.
    """
    c_indptr = np.zeros(n + 1, dtype=np.int64)
    nnz_per_row = np.zeros(n, dtype=np.int64)

    # First pass: determine the number of non-zero elements in each row of C
    for i in range(n):
        # Use a boolean array as a hash set to save memory
        seen_mask = np.zeros(n, dtype=np.bool_)
        nnz = 0
        for j_ptr in range(a_indptr[i], a_indptr[i+1]):
            j = a_indices[j_ptr]
            for k_ptr in range(b_indptr[j], b_indptr[j+1]):
                k = b_indices[k_ptr]
                if not seen_mask[k]:
                    seen_mask[k] = True
                    nnz += 1
        nnz_per_row[i] = nnz
    
    # Prefix sum to get the final indptr
    c_indptr[1:] = np.cumsum(nnz_per_row)
        
    nnz_c = c_indptr[n]
    if nnz_c < 0: # Overflow check
        print("Error: The number of non-zero elements has overflowed the integer capacity.")
        # Return an empty matrix to prevent crashing
        return np.zeros(n + 1, dtype=np.int64), np.zeros(0, dtype=np.int32)

    c_indices = np.zeros(nnz_c, dtype=np.int32)
    
    # Second pass: fill the indices
    for i in range(n):
        row_start = c_indptr[i]
        row_end = c_indptr[i+1]
        
        if row_start == row_end:
            continue
            
        seen_mask = np.zeros(n, dtype=np.bool_)
        row_nnz = 0
        temp_indices = np.zeros(nnz_per_row[i], dtype=np.int32)

        for j_ptr in range(a_indptr[i], a_indptr[i+1]):
            j = a_indices[j_ptr]
            for k_ptr in range(b_indptr[j], b_indptr[j+1]):
                k = b_indices[k_ptr]
                if not seen_mask[k]:
                    seen_mask[k] = True
                    temp_indices[row_nnz] = k
                    row_nnz += 1
        
        temp_indices.sort()
        c_indices[row_start:row_end] = temp_indices

    return c_indptr, c_indices


def multiply_matrices(mat1: sp.csr_matrix, mat2: sp.csr_matrix) -> sp.csr_matrix:
    """Helper to multiply two CSR matrices using the Numba kernel."""
    n = mat1.shape[0]
    c_indptr, c_indices = numba_spgemm(
        mat1.indptr.astype(np.int64), mat1.indices.astype(np.int32), mat1.data,
        mat2.indptr.astype(np.int64), mat2.indices.astype(np.int32), mat2.data,
        n
    )
    # The result is binary, so data is all ones
    c_data = np.ones(len(c_indices), dtype=np.float32)
    return sp.csr_matrix((c_data, c_indices, c_indptr), shape=(n, n))


def scipy_to_torch_sparse(matrix: sp.csr_matrix, device: torch.device) -> torch.Tensor:
    """Converts a Scipy sparse matrix to a PyTorch sparse COO tensor."""
    coo = matrix.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape
    
    return torch.sparse_coo_tensor(i, v, torch.Size(shape), device=device)


def parse_meta_path(path_str: str, relations: dict) -> sp.csr_matrix:
    """
    Computes the adjacency matrix for a given meta-path string.
    Supports two types of definitions:
    1. Standard path: 'ACA' (Author-Conference-Author)
    2. Product path: 'AT@AT.T' (Author-Term x Term-Author)
    """
    print(f"  Computing base matrix for meta-path: {path_str}")
    
    if '@' in path_str:
        # Handle product paths like 'AT@AT.T'
        left_str, right_str = path_str.split('@')
        
        def get_matrix(s):
            if s.endswith('.T'):
                mat_key = s[:-2]
                return relations[mat_key[0]][mat_key[1]].T
            else:
                return relations[s[0]][s[1]]

        left_mat = get_matrix(left_str)
        right_mat = get_matrix(right_str)
        
        # Note: This is a standard, potentially dense multiplication
        # It's not using the binary numba kernel
        final_mat = left_mat @ right_mat
    else:
        # Handle standard paths like 'ACA'
        parts = list(path_str)
        final_mat = relations[parts[0]][parts[1]]
        for i in range(1, len(parts) - 1):
            final_mat = final_mat @ relations[parts[i]][parts[i+1]]
    
    return final_mat.tocsr()


def get_training_pairs(adj_matrix: sp.csr_matrix, num_neg_samples: int, seed: int):
    """
    Get positive and negative training pairs.
    This implementation is simple but can be slow for dense graphs.
    """
    rng = np.random.default_rng(seed)
    pos_pairs = np.array(adj_matrix.nonzero()).T
    
    n_nodes = adj_matrix.shape[0]
    n_pos = len(pos_pairs)
    n_neg = n_pos * num_neg_samples
    
    neg_pairs = np.zeros((n_neg, 2), dtype=pos_pairs.dtype)
    
    # Pre-generate random pairs and filter out existing edges
    # This is more efficient than one-by-one generation.
    candidate_neg = rng.integers(0, n_nodes, size=(int(n_neg * 1.2), 2)) # Generate a bit more
    
    # Filter out self-loops and positive edges
    valid_mask = (candidate_neg[:, 0] != candidate_neg[:, 1])
    candidate_neg = candidate_neg[valid_mask]
    
    adj_coo = adj_matrix.tocoo()
    pos_set = set(zip(adj_coo.row, adj_coo.col))
    
    count = 0
    for i in range(len(candidate_neg)):
        if count >= n_neg:
            break
        p = tuple(candidate_neg[i])
        if p not in pos_set:
            neg_pairs[count] = p
            count += 1
            
    if count < n_neg:
        # Fallback for the case we didn't generate enough unique negative samples
        # This part remains slow, but is less likely to be hit.
        extra_needed = n_neg - count
        more_negs = []
        while len(more_negs) < extra_needed:
            i, j = rng.integers(0, n_nodes, 2)
            if i != j and not adj_matrix[i, j]:
                more_negs.append([i,j])
        neg_pairs[count:] = np.array(more_negs)

    return pos_pairs, neg_pairs


def main(args):
    # Setup
    # Auto-select device for the *model*
    if args.device == 'auto':
        if torch.backends.mps.is_available():
            model_device = torch.device('mps')
        elif torch.cuda.is_available():
            model_device = torch.device('cuda')
        else:
            model_device = torch.device('cpu')
    else:
        model_device = torch.device(args.device)
    print(f"Using device: {model_device} for model training.")
    print("Note: Meta-path matrix computations are forced to CPU due to PyTorch limitations.")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create cache directory
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using cache directory: {cache_dir.resolve()}")

    # Load data
    relations, n_authors, _, _ = load_data(args.data_dir)
    
    # Compute base meta-path matrices
    print("Computing base meta-path matrices...")
    meta_path_matrices = {}
    for path_str in args.meta_paths:
        meta_path_matrices[path_str] = parse_meta_path(path_str, relations)
        print(f"    - Computed {path_str}: {meta_path_matrices[path_str].nnz} non-zero entries")

    # Create model
    model = FastRPModel(
        n_authors=n_authors,
        dim=args.dim,
        meta_paths=meta_path_matrices,
        num_powers=args.num_powers,
        alpha=args.alpha,
        beta=args.beta,
        device=model_device
    ).to(model_device)

    # Prepare training data (using author-author co-occurrence)
    train_adj = relations['A']['A']
    pos_pairs, neg_pairs = get_training_pairs(train_adj, args.neg_samples, args.seed)
    
    train_pairs = np.vstack([pos_pairs, neg_pairs])
    train_labels = torch.FloatTensor(
        [1] * len(pos_pairs) + [0] * len(neg_pairs)
    ).to(model_device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        
        # Shuffle data for batching
        perm = torch.randperm(len(train_pairs))
        train_pairs_shuffled = train_pairs[perm]
        train_labels_shuffled = train_labels[perm]

        epoch_loss = 0.0
        epoch_preds = []
        epoch_labels = []

        for i in tqdm(range(0, len(train_pairs), args.batch_size), desc=f"Epoch {epoch+1}"):
            batch_pairs = train_pairs_shuffled[i:i+args.batch_size]
            batch_labels = train_labels_shuffled[i:i+args.batch_size].to(model_device)
            
            idx_i = torch.from_numpy(batch_pairs[:, 0]).long()
            idx_j = torch.from_numpy(batch_pairs[:, 1]).long()

            optimizer.zero_grad()
            preds = model(idx_i, idx_j)
            
            # --- Loss Calculation ---
            # 1. Standard link prediction loss
            bce_loss = F.binary_cross_entropy(preds, batch_labels)
            
            # 2. Entropy regularization to encourage diverse weights
            weights_softmax = F.softmax(model.feature_weights, dim=0)
            entropy = -torch.sum(weights_softmax * torch.log(weights_softmax + 1e-7))
            
            # 3. Combined loss
            loss = bce_loss - args.entropy_beta * entropy
            
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(batch_pairs)
            epoch_preds.extend(preds.detach().cpu().numpy())
            epoch_labels.extend(batch_labels.cpu().numpy())

        avg_loss = epoch_loss / len(train_pairs)
        
        # Evaluation
        auc = roc_auc_score(epoch_labels, epoch_preds)
        
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f} | AUC: {auc:.4f}")
        print(f"  Feature weights (softmaxed): {F.softmax(model.feature_weights, dim=0).detach().cpu().numpy()}")
        print(f"  Intercept: {model.intercept.item():.4f}")

    # Save embeddings
    final_embeddings = model.get_embedding().detach().cpu().numpy()
    np.save(args.output, final_embeddings)
    print(f"Embeddings saved to {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FastRP for Heterogeneous Graphs")
    parser.add_argument('--data-dir', type=str, default='data', help='Directory containing the dataset')
    parser.add_argument('--meta-paths', type=str, nargs='+', default=['AAA', 'ATA', 'ACA', 'AT@AT.T'], help='Meta-paths to use. Supports standard paths (e.g., ACA) and product paths (e.g., AT@AT.T)')
    parser.add_argument('--dim', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--num-powers', type=int, default=3, help='Number of matrix powers to use (q)')
    parser.add_argument('--alpha', type=float, default=-0.5, help='Exponent for degree weighting of projection matrix')
    parser.add_argument('--beta', type=float, default=-0.5, help='Exponent for degree normalization of meta-path matrix')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--entropy-beta', type=float, default=0.05, help='Coefficient for entropy regularization term.')
    parser.add_argument('--neg-samples', type=int, default=3, help='Number of negative samples per positive sample.')
    parser.add_argument('--batch-size', type=int, default=4096, help='Training batch size')
    parser.add_argument('--device', type=str, default='auto', help='Device to use for training (e.g., "cpu", "cuda", "mps", "auto").')
    parser.add_argument('--output', type=str, default='author_embeddings.npy', help='Path to save final embeddings')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--cache-dir', type=str, default='./matrix_cache', help='Directory to cache computed meta-path matrices.')
    
    args = parser.parse_args()
    main(args) 