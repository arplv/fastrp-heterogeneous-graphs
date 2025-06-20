import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from torchmetrics import AUROC, Precision, Recall
from pathlib import Path
from tqdm import tqdm

from data_loader import load_data
from model import FastRPModel

def main(args):
    # Setup
    if args.device == 'auto':
        if torch.backends.mps.is_available(): model_device = torch.device('mps')
        elif torch.cuda.is_available(): model_device = torch.device('cuda')
        else: model_device = torch.device('cpu')
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

    # Prepare training data: positive edges
    train_adj = relations['A']['A']
    train_adj.setdiag(0)
    train_adj.eliminate_zeros()
    pos_edge_index = torch.from_numpy(np.array(train_adj.nonzero())).long().to(model_device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Initialize metrics on the target device
    auroc = AUROC(task="binary").to(model_device)
    precision_metric = Precision(task="binary").to(model_device)
    recall_metric = Recall(task="binary").to(model_device)

    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        
        perm = torch.randperm(pos_edge_index.size(1), device=model_device)

        total_loss = 0.0
        
        # Reset metrics at the start of each epoch
        auroc.reset()
        precision_metric.reset()
        recall_metric.reset()

        for i in tqdm(range(0, pos_edge_index.size(1), args.batch_size), desc=f"Epoch {epoch+1}"):
            batch_indices = perm[i:i+args.batch_size]
            pos_batch = pos_edge_index[:, batch_indices]
            
            # Generate negative samples on the fly on the GPU
            neg_batch = negative_sampling(
                edge_index=pos_edge_index,
                num_nodes=n_authors,
                num_neg_samples=pos_batch.size(1) * args.neg_samples
            )

            # Combine positive and negative samples for the batch
            idx_i = torch.cat([pos_batch[0], neg_batch[0]])
            idx_j = torch.cat([pos_batch[1], neg_batch[1]])
            labels = torch.cat([
                torch.ones(pos_batch.size(1)), 
                torch.zeros(neg_batch.size(1))
            ]).to(model_device)

            optimizer.zero_grad()
            preds = model(idx_i, idx_j)
            
            bce_loss = F.binary_cross_entropy(preds, labels)
            
            # Add entropy to encourage diverse weights
            weights_softmax = F.softmax(model.feature_weights, dim=1)
            entropy = -torch.sum(weights_softmax * torch.log(weights_softmax + 1e-7))
            
            loss = bce_loss + args.lambda_entropy * entropy
            
            loss.backward()
            optimizer.step()

            # Refresh embedding cache after weights have been updated
            model.refresh_embedding_cache()

            total_loss += loss.item() * (pos_batch.size(1) + neg_batch.size(1))
            
            # Update metrics with the current batch's results
            auroc.update(preds, labels)
            precision_metric.update(preds, labels)
            recall_metric.update(preds, labels)

        avg_loss = total_loss / (pos_edge_index.size(1) * (1 + args.neg_samples))
        
        # Compute final metrics for the epoch
        epoch_auc = auroc.compute()
        epoch_precision = precision_metric.compute()
        epoch_recall = recall_metric.compute()
        
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f} | AUC: {epoch_auc:.4f} | Precision: {epoch_precision:.4f} | Recall: {epoch_recall:.4f}")

        # --- Debug: Print learned coefficients ---
        with torch.no_grad():
            weights_softmax = F.softmax(model.feature_weights, dim=1)
            # Format the meta-path weights for readability
            weights_str = "\n".join([f"    {path}: {weights.cpu().numpy()}" for path, weights in zip(args.meta_paths, weights_softmax)])
            print(f"  Feature Weights (softmaxed over powers):")
            print(weights_str)
            print(f"  Intercept: {model.intercept.item():.4f} | Slope: {model.slope.item():.4f}")
        # --- End Debug ---

    print("Training finished.")
    model.eval()

    if args.output:
        print(f"Computing and saving final embeddings to {args.output}...")
        final_embeddings = model.get_embedding().detach().cpu()
        torch.save(final_embeddings, args.output)
        print("Embeddings saved.")

    if args.save_model_path:
        print(f"Saving model checkpoint to {args.save_model_path}...")
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
    parser.add_argument('--meta-paths', type=str, nargs='+', default=['AAA', 'ACA', 'ATA'], help='List of meta-paths to use. Element-wise products are not supported.')
    parser.add_argument('--dim', type=int, default=256, help='Embedding dimension.')
    parser.add_argument('--s', type=int, default=3, help='Sparsity for random projection matrix (s non-zero entries per column).')
    parser.add_argument('--num-powers', type=int, default=2, help='Number of matrix powers to use for each meta-path feature.')
    parser.add_argument('--alpha', type=float, default=-0.5, help='Exponent for degree weighting of the random projection matrix.')
    parser.add_argument('--beta', type=float, default=-1.0, help='Exponent for degree normalization of the meta-path matrix.')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--lambda-entropy', type=float, default=0.0, help='Coefficient for entropy regularization. Set to 0 to disable, or a small value like 1e-4 to encourage diversity.')
    parser.add_argument('--neg-samples', type=int, default=3, help='Number of negative samples per positive sample.')
    parser.add_argument('--batch-size', type=int, default=4096, help='Training batch size')
    parser.add_argument('--device', type=str, default='auto', help='Device to use for training (e.g., "cpu", "cuda", "mps").')
    parser.add_argument('--output', type=str, default='author_embeddings.pt', help='Path to save final embeddings')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--cache-dir', type=str, default='./matrix_cache', help='Directory to cache computed meta-path matrices.')
    parser.add_argument('--save-model-path', type=str, default='fastrp_model.pth', help='Path to save the trained model checkpoint.')
    
    args = parser.parse_args()
    main(args) 