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
    Uses a fast, vectorized approach for negative sampling.
    """
    rng = np.random.default_rng(seed)
    pos_pairs = np.array(adj_matrix.nonzero()).T
    
    n_nodes = adj_matrix.shape[0]
    n_pos = len(pos_pairs)
    n_neg = n_pos * num_neg_samples
    
    # Use 64-bit integer hashes for fast checking of existing edges.
    pos_hashes = set(pos_pairs[:, 0].astype(np.uint64) << 32 | pos_pairs[:, 1].astype(np.uint64))
    
    neg_pairs = np.zeros((n_neg, 2), dtype=pos_pairs.dtype)
    
    # Generate more candidates than needed to account for collisions.
    # This is a vectorized approach, much faster than looping.
    num_candidates = int(n_neg * 1.1)
    
    # Generate random pairs
    candidates = rng.integers(0, n_nodes, size=(num_candidates, 2))
    
    # Filter out self-loops
    self_loop_mask = candidates[:, 0] != candidates[:, 1]
    candidates = candidates[self_loop_mask]

    # Filter out positive edges using the hash set
    candidate_hashes = candidates[:, 0].astype(np.uint64) << 32 | candidates[:, 1].astype(np.uint64)
    collision_mask = ~np.isin(candidate_hashes, list(pos_hashes))
    
    neg_pairs = candidates[collision_mask][:n_neg]

    if len(neg_pairs) < n_neg:
        print(f"Warning: Generated {len(neg_pairs)} negative samples, less than the requested {n_neg}. Consider increasing the candidate multiplier.")

    return pos_pairs, neg_pairs


def main(args):
    # Setup
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

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using cache directory: {cache_dir.resolve()}")

    print("Loading data...")
    relations, n_authors, _, _ = load_data(args.data_dir)
    print("Data loading complete.")

    model = FastRPModel(
        n_authors=n_authors,
        dim=args.dim,
        meta_paths=args.meta_paths,
        relations=relations,
        num_powers=args.num_powers,
        alpha=args.alpha,
        beta=args.beta,
        s=args.s,
        device=model_device
    ).to(model_device)

    # Prepare training data
    train_adj = relations['A']['A']
    pos_pairs, neg_pairs = get_training_pairs(train_adj, args.neg_samples, args.seed)
    
    train_pairs = np.vstack([pos_pairs, neg_pairs])
    train_labels = torch.FloatTensor([1] * len(pos_pairs) + [0] * len(neg_pairs))

    # Move all training data to the target device once to avoid copies in the loop
    train_pairs_tensor = torch.from_numpy(train_pairs).long().to(model_device)
    train_labels_tensor = train_labels.to(model_device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        
        perm = torch.randperm(len(train_pairs_tensor), device=model_device)

        epoch_loss = 0.0
        # Store predictions and labels on the device during the epoch
        all_preds_gpu = []
        all_labels_gpu = []

        for i in tqdm(range(0, len(train_pairs_tensor), args.batch_size), desc=f"Epoch {epoch+1}"):
            batch_indices = perm[i:i+args.batch_size]
            batch_pairs = train_pairs_tensor[batch_indices]
            batch_labels = train_labels_tensor[batch_indices]
            
            idx_i = batch_pairs[:, 0]
            idx_j = batch_pairs[:, 1]

            optimizer.zero_grad()
            preds = model(idx_i, idx_j)
            
            bce_loss = F.binary_cross_entropy(preds, batch_labels)
            
            # Add entropy regularization. Positive lambda -> lower entropy (sharper weights).
            # The goal is to encourage the model to focus on the most informative features.
            weights_softmax = F.softmax(model.feature_weights, dim=1) # (paths, powers)
            entropy = -torch.sum(weights_softmax * torch.log(weights_softmax + 1e-7))
            
            loss = bce_loss + args.lambda_entropy * entropy
            
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(batch_pairs)
            all_preds_gpu.append(preds.detach())
            all_labels_gpu.append(batch_labels)

        avg_loss = epoch_loss / len(train_pairs)
        
        # Evaluation: move data to CPU only once at the end of the epoch
        epoch_preds_np = torch.cat(all_preds_gpu).cpu().numpy()
        epoch_labels_np = torch.cat(all_labels_gpu).cpu().numpy()
        
        auc = roc_auc_score(epoch_labels_np, epoch_preds_np)
        
        binary_preds = (epoch_preds_np > 0.5).astype(int)
        precision = precision_score(epoch_labels_np, binary_preds, zero_division=0)
        recall = recall_score(epoch_labels_np, binary_preds, zero_division=0)
        
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f} | AUC: {auc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

    print("Training finished.")
    model.eval()

    if args.output:
        print(f"Computing and saving final embeddings to {args.output}...")
        final_embeddings = model.get_embedding().detach().cpu()
        # Save as a standard .pt file for PyTorch
        torch.save(final_embeddings, args.output)
        print("Embeddings saved.")

    if args.save_model_path:
        print(f"Saving model checkpoint to {args.save_model_path}...")
        # Don't save 'relations' as it's large and can be reconstructed
        args_dict = {k: v for k, v in vars(args).items() if k != 'relations'}
        checkpoint = {
            'args': args_dict,
            'model_state_dict': model.state_dict(),
        }
        torch.save(checkpoint, args.save_model_path)
        print("Model saved.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FastRP for Heterogeneous Graphs")
    parser.add_argument('--data-dir', type=str, default='data', help='Directory containing the dataset')
    parser.add_argument('--meta-paths', type=str, nargs='+', 
                        default=['AAA', 'ACA', 'ATA'], 
                        help='List of meta-paths to use. Element-wise products are not supported.')
    parser.add_argument('--dim', type=int, default=128, help='Embedding dimension.')
    parser.add_argument('--s', type=int, default=3, help='Sparsity for random projection matrix (s non-zero entries per column).')
    parser.add_argument('--num-powers', type=int, default=2, help='Number of matrix powers to use for each meta-path feature.')
    parser.add_argument('--alpha', type=float, default=-0.5, help='Exponent for degree weighting of the random projection matrix.')
    parser.add_argument('--beta', type=float, default=-1.0, help='Exponent for degree normalization of the meta-path matrix.')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--lambda-entropy', type=float, default=0.001, help='Coefficient for entropy regularization. Positive values encourage lower entropy (sharper weights).')
    parser.add_argument('--neg-samples', type=int, default=3, help='Number of negative samples per positive sample.')
    parser.add_argument('--batch-size', type=int, default=4096, help='Training batch size')
    parser.add_argument('--device', type=str, default='auto', help='Device to use for training (e.g., "cpu", "cuda", "mps").')
    parser.add_argument('--output', type=str, default='author_embeddings.pt', help='Path to save final embeddings')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--cache-dir', type=str, default='./matrix_cache', help='Directory to cache computed meta-path matrices.')
    parser.add_argument('--save-model-path', type=str, default='fastrp_model.pth', help='Path to save the trained model checkpoint.')
    
    args = parser.parse_args()
    main(args) 