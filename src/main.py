import argparse
import numpy as np
import scipy.sparse as sp
import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from pathlib import Path
from tqdm import tqdm

from data_loader import load_data
from model import FastRPModel

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
    print("Loading data...")
    relations, n_authors, _, _ = load_data(args.data_dir)
    print("Data loading complete.")

    # Create model
    model = FastRPModel(
        n_authors=n_authors,
        dim=args.dim,
        meta_paths=args.meta_paths,
        relations=relations,
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
        epoch_labels_np = np.array(epoch_labels)
        epoch_preds_np = np.array(epoch_preds)
        auc = roc_auc_score(epoch_labels_np, epoch_preds_np)
        
        # Convert predictions to binary (0 or 1) for precision/recall
        binary_preds = (epoch_preds_np > 0.5).astype(int)
        precision = precision_score(epoch_labels_np, binary_preds, zero_division=0)
        recall = recall_score(epoch_labels_np, binary_preds, zero_division=0)
        
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f} | AUC: {auc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
        print(f"  Feature weights (softmaxed): {F.softmax(model.feature_weights, dim=0).detach().cpu().numpy()}")
        print(f"  Intercept: {model.intercept.item():.4f}")

    # Save embeddings
    final_embeddings = model.get_embedding().detach().cpu().numpy()
    np.save(args.output, final_embeddings)
    print(f"Embeddings saved to {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FastRP for Heterogeneous Graphs")
    parser.add_argument('--data-dir', type=str, default='data', help='Directory containing the dataset')
    parser.add_argument('--meta-paths', type=str, nargs='+', 
                        default=['AAA', 'ACA', 'ATA'], 
                        help='List of meta-paths to use for feature generation.')
    parser.add_argument('--dim', type=int, default=512, help='Embedding dimension.')
    parser.add_argument('--num-powers', type=int, default=4, help='Number of matrix powers to use for each meta-path feature.')
    parser.add_argument('--alpha', type=float, default=-0.5, help='Exponent for degree weighting of the random projection matrix.')
    parser.add_argument('--beta', type=float, default=-1.0, help='Exponent for degree normalization of the meta-path matrix.')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--entropy-beta', type=float, default=0.05, help='Coefficient for entropy regularization term.')
    parser.add_argument('--neg-samples', type=int, default=15, help='Number of negative samples per positive sample.')
    parser.add_argument('--batch-size', type=int, default=64, help='Training batch size')
    parser.add_argument('--device', type=str, default='auto', help='Device to use for training (e.g., "cpu", "cuda", "mps", "auto").')
    parser.add_argument('--output', type=str, default='author_embeddings.npy', help='Path to save final embeddings')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--cache-dir', type=str, default='./matrix_cache', help='Directory to cache computed meta-path matrices.')
    
    args = parser.parse_args()
    main(args) 