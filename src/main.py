import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from torchmetrics import AUROC, Precision, Recall, F1Score, Accuracy
from pathlib import Path
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy
import scipy.sparse as sp
import matplotlib.pyplot as plt

from data_loader import load_data
from model import FastRPModel

def plot_metrics(metrics_history, output_path):
    """Plots training and validation metrics."""
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    fig.suptitle('Training and Validation Metrics Over Epochs')

    epochs = range(1, len(metrics_history['train_loss']) + 1)

    # Loss
    axs[0, 0].plot(epochs, metrics_history['train_loss'], 'b-', label='Training Loss')
    axs[0, 0].set_title('Training Loss')
    axs[0, 0].set_xlabel('Epochs')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()

    # AUROC
    axs[0, 1].plot(epochs, metrics_history['train_auc'], 'b-', label='Training AUC')
    axs[0, 1].plot(epochs, metrics_history['val_auc'], 'r-', label='Validation AUC')
    axs[0, 1].set_title('Area Under ROC Curve (AUC)')
    axs[0, 1].set_xlabel('Epochs')
    axs[0, 1].set_ylabel('AUC')
    axs[0, 1].legend()
    
    # Accuracy
    axs[1, 0].plot(epochs, metrics_history['train_acc'], 'b-', label='Training Accuracy')
    axs[1, 0].plot(epochs, metrics_history['val_acc'], 'r-', label='Validation Accuracy')
    axs[1, 0].set_title('Accuracy')
    axs[1, 0].set_xlabel('Epochs')
    axs[1, 0].set_ylabel('Accuracy')
    axs[1, 0].legend()

    # Precision
    axs[1, 1].plot(epochs, metrics_history['train_precision'], 'b-', label='Training Precision')
    axs[1, 1].plot(epochs, metrics_history['val_precision'], 'r-', label='Validation Precision')
    axs[1, 1].set_title('Precision')
    axs[1, 1].set_xlabel('Epochs')
    axs[1, 1].set_ylabel('Precision')
    axs[1, 1].legend()

    # Recall
    axs[2, 0].plot(epochs, metrics_history['train_recall'], 'b-', label='Training Recall')
    axs[2, 0].plot(epochs, metrics_history['val_recall'], 'r-', label='Validation Recall')
    axs[2, 0].set_title('Recall')
    axs[2, 0].set_xlabel('Epochs')
    axs[2, 0].set_ylabel('Recall')
    axs[2, 0].legend()

    # F1 Score
    axs[2, 1].plot(epochs, metrics_history['train_f1'], 'b-', label='Training F1-Score')
    axs[2, 1].plot(epochs, metrics_history['val_f1'], 'r-', label='Validation F1-Score')
    axs[2, 1].set_title('F1 Score')
    axs[2, 1].set_xlabel('Epochs')
    axs[2, 1].set_ylabel('F1 Score')
    axs[2, 1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plot_dir = Path(output_path).parent
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path)
    print(f"Metrics plot saved to {output_path}")

def _get_edges_from_targets(targets, relations, node_offsets, device):
    """Gets positive edge indices for given training targets, mapped to the global ID space."""
    pos_edge_indices = []
    for target in targets:
        src_type, dst_type = target.split('-')
        if src_type not in relations or dst_type not in relations[src_type]:
            raise ValueError(f"Target '{target}' not found in relations.")
        
        matrix = relations[src_type][dst_type]
        src_offset = node_offsets[src_type]
        dst_offset = node_offsets[dst_type]

        # Get edges and map to global IDs
        rows, cols = sp.triu(matrix, k=1).nonzero()
        src_ids = torch.from_numpy(rows + src_offset).long()
        dst_ids = torch.from_numpy(cols + dst_offset).long()
        
        pos_edge_indices.append(torch.stack([src_ids, dst_ids], dim=0))

    return torch.cat(pos_edge_indices, dim=1).to(device)

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
    relations, stitched_relations, node_counts, node_offsets = load_data(args.data_dir)
    n_total = sum(node_counts.values())
    print("Data loading complete.")

    model = FastRPModel(
        n_total=n_total,
        dim=args.dim,
        meta_paths=args.meta_paths,
        relations=stitched_relations,
        num_powers=args.num_powers,
        alpha=args.alpha,
        beta=args.beta,
        s=args.s,
        device=model_device
    ).to(model_device)

    # --- Prepare training data with balancing ---
    if args.edge_split:
        # This part remains for users who provide their own splits, but we add a warning.
        print(f"Warning: Using a pre-computed edge split from {args.edge_split}.")
        print("         The balanced sampling strategy will be skipped.")
        if len(args.training_targets) > 1:
            print("Warning: Edge split is used, but multiple training targets are specified. Behavior may be unexpected.")
        
        split_data = torch.load(args.edge_split)
        # This assumes the split was created for a single relation type and needs manual offset adjustment.
        # It's a legacy path and less robust than the new balancing approach.
        src_type, _ = args.training_targets[0].split('-')
        train_pos_edge_index = split_data['train_pos_edge_index'] + node_offsets[src_type]
        val_pos_edge_index = split_data['val_pos_edge_index'] + node_offsets[src_type]
        all_pos_edges_count = train_pos_edge_index.size(1) + val_pos_edge_index.size(1)
    else:
        print(f"Using raw training targets: {args.training_targets}")
        
        # De-duplicate targets by creating a canonical representation (e.g., A-T for both A-T and T-A)
        canonical_targets = set()
        for target in args.training_targets:
            parts = sorted(target.split('-'))
            canonical_targets.add(f"{parts[0]}-{parts[1]}")
        
        print(f"Balancing canonical targets: {list(canonical_targets)}")
        edges_by_target = {}
        for target in canonical_targets:
            src_type, dst_type = target.split('-')
            is_symmetric = src_type == dst_type
            
            # Check for both A-C and C-A etc.
            if src_type in relations and dst_type in relations[src_type]:
                matrix = relations[src_type][dst_type]
            elif dst_type in relations and src_type in relations[dst_type]:
                matrix = relations[dst_type][src_type]
            else:
                print(f"Warning: Target '{target}' not found in relations. Skipping.")
                continue
            
            src_offset = node_offsets[src_type]
            dst_offset = node_offsets[dst_type]

            if is_symmetric:
                # For symmetric relations (A-A), take upper triangle to avoid duplicates
                rows, cols = sp.triu(matrix, k=1).nonzero()
            else:
                # For bipartite relations, take all edges
                rows, cols = matrix.nonzero()
                
            src_ids = torch.from_numpy(rows + src_offset).long()
            dst_ids = torch.from_numpy(cols + dst_offset).long()
            
            edges_by_target[target] = torch.stack([src_ids, dst_ids], dim=0)

        if not edges_by_target:
            raise ValueError("No valid training targets found. Aborting.")

        # Balance by oversampling to the size of the largest relation
        if not edges_by_target:
            print("Warning: No edges found for any target. Cannot proceed.")
            return
            
        max_edges = max(e.size(1) for e in edges_by_target.values())
        print(f"Balancing all relations to {max_edges} edges each by oversampling.")

        balanced_edges = []
        for target, edges in edges_by_target.items():
            num_current_edges = edges.size(1)
            
            if num_current_edges == 0:
                print(f"  - Warning: Skipping '{target}' as it has no edges.")
                continue

            if num_current_edges < max_edges:
                # Oversample with replacement
                indices = torch.randint(0, num_current_edges, (max_edges,), device=edges.device)
                balanced_edges.append(edges[:, indices])
                print(f"  - Oversampled '{target}' to {max_edges} edges (from {num_current_edges})")
            else:
                # This relation is already at max size, no sampling needed
                balanced_edges.append(edges)
                print(f"  - Using all {num_current_edges} edges for '{target}'")
        
        if not balanced_edges:
            print("Error: No edges to train on after balancing. Aborting.")
            return

        all_pos_edges = torch.cat(balanced_edges, dim=1).to(model_device)
        all_pos_edges_count = all_pos_edges.size(1)

        # Simple random split of the now-balanced set of edges
        perm = torch.randperm(all_pos_edges.size(1))
        val_size = int(all_pos_edges.size(1) * 0.1) # 10% for validation
        train_pos_edge_index = all_pos_edges[:, perm[val_size:]]
        val_pos_edge_index = all_pos_edges[:, perm[:val_size]]

    print(f"  Total positive edges after balancing: {all_pos_edges_count}")
    print(f"  Train positive edges: {train_pos_edge_index.size(1)}")
    print(f"  Validation positive edges: {val_pos_edge_index.size(1)}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=10)
    
    # Initialize metrics on the target device
    metrics = {
        'train_auroc': AUROC(task="binary").to(model_device),
        'val_auroc': AUROC(task="binary").to(model_device),
        'train_acc': Accuracy(task="binary").to(model_device),
        'val_acc': Accuracy(task="binary").to(model_device),
        'train_precision': Precision(task="binary").to(model_device),
        'val_precision': Precision(task="binary").to(model_device),
        'train_recall': Recall(task="binary").to(model_device),
        'val_recall': Recall(task="binary").to(model_device),
        'train_f1': F1Score(task="binary").to(model_device),
        'val_f1': F1Score(task="binary").to(model_device),
    }

    metrics_history = {
        'train_loss': [], 'train_auc': [], 'val_auc': [],
        'train_acc': [], 'val_acc': [], 'train_precision': [], 'val_precision': [],
        'train_recall': [], 'val_recall': [], 'train_f1': [], 'val_f1': []
    }
    
    best_val_auc = 0
    epochs_no_improve = 0
    best_model_state = None

    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        
        # --- Training Phase ---
        perm = torch.randperm(train_pos_edge_index.size(1), device=model_device)
        total_loss = 0.0
        for metric in metrics.values(): metric.reset()

        for i in tqdm(range(0, train_pos_edge_index.size(1), args.batch_size), desc=f"Epoch {epoch+1} [Train]"):
            batch_indices = perm[i:i+args.batch_size]
            pos_batch = train_pos_edge_index[:, batch_indices]
            
            neg_batch = negative_sampling(
                edge_index=train_pos_edge_index, # Sample negatives from the whole graph
                num_nodes=n_total,
                num_neg_samples=pos_batch.size(1) * args.neg_samples
            )

            idx_i = torch.cat([pos_batch[0], neg_batch[0]])
            idx_j = torch.cat([pos_batch[1], neg_batch[1]])
            labels = torch.cat([
                torch.ones(pos_batch.size(1)), 
                torch.zeros(neg_batch.size(1))
            ]).to(model_device)

            optimizer.zero_grad()
            logits = model(idx_i, idx_j)
            
            # Use weighted BCE loss to handle class imbalance
            pos_weight = torch.tensor([args.neg_samples], device=model_device)
            bce_loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
            
            # Optional L2 regularization on raw feature weights (if lambda > 0)
            if args.lambda_entropy > 0:
                weight_reg = torch.sum(model.feature_weights ** 2)
                loss = bce_loss + args.lambda_entropy * weight_reg
            else:
                loss = bce_loss
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            # Update all train metrics
            metrics['train_auroc'].update(logits, labels)
            metrics['train_acc'].update(logits, labels)
            metrics['train_precision'].update(logits, labels)
            metrics['train_recall'].update(logits, labels)
            metrics['train_f1'].update(logits, labels)

        avg_loss = total_loss / (len(range(0, train_pos_edge_index.size(1), args.batch_size)))
        metrics_history['train_loss'].append(avg_loss)
        metrics_history['train_auc'].append(metrics['train_auroc'].compute().item())
        metrics_history['train_acc'].append(metrics['train_acc'].compute().item())
        metrics_history['train_precision'].append(metrics['train_precision'].compute().item())
        metrics_history['train_recall'].append(metrics['train_recall'].compute().item())
        metrics_history['train_f1'].append(metrics['train_f1'].compute().item())

        # --- Validation Phase ---
        model.eval()
        # Reset only validation metrics, train metrics were computed
        metrics['val_auroc'].reset()
        metrics['val_acc'].reset()
        metrics['val_precision'].reset()
        metrics['val_recall'].reset()
        metrics['val_f1'].reset()
        with torch.no_grad():
            for i in tqdm(range(0, val_pos_edge_index.size(1), args.batch_size), desc=f"Epoch {epoch+1} [Val]"):
                pos_batch = val_pos_edge_index[:, i:i+args.batch_size]
                neg_batch = negative_sampling(
                    edge_index=train_pos_edge_index, # IMPORTANT: still sample negatives from the whole graph space
                    num_nodes=n_total,
                    num_neg_samples=pos_batch.size(1) * args.neg_samples
                )
                idx_i = torch.cat([pos_batch[0], neg_batch[0]])
                idx_j = torch.cat([pos_batch[1], neg_batch[1]])
                labels = torch.cat([torch.ones(pos_batch.size(1)), torch.zeros(neg_batch.size(1))]).to(model_device)
                
                logits = model(idx_i, idx_j)
                # Update all val metrics
                metrics['val_auroc'].update(logits, labels)
                metrics['val_acc'].update(logits, labels)
                metrics['val_precision'].update(logits, labels)
                metrics['val_recall'].update(logits, labels)
                metrics['val_f1'].update(logits, labels)
        
        epoch_val_auc = metrics['val_auroc'].compute().item()
        metrics_history['val_auc'].append(epoch_val_auc)
        metrics_history['val_acc'].append(metrics['val_acc'].compute().item())
        metrics_history['val_precision'].append(metrics['val_precision'].compute().item())
        metrics_history['val_recall'].append(metrics['val_recall'].compute().item())
        metrics_history['val_f1'].append(metrics['val_f1'].compute().item())

        scheduler.step(epoch_val_auc)

        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {metrics_history['train_loss'][-1]:.4f} | Train AUC: {metrics_history['train_auc'][-1]:.4f} | Val AUC: {metrics_history['val_auc'][-1]:.4f}")
        print(f"  Val Accuracy: {metrics_history['val_acc'][-1]:.4f} | Val Precision: {metrics_history['val_precision'][-1]:.4f} | Val Recall: {metrics_history['val_recall'][-1]:.4f} | Val F1: {metrics_history['val_f1'][-1]:.4f}")

        with torch.no_grad():
            raw_weights = model.feature_weights.flatten().cpu().numpy()
            print(f"  Slope: {model.slope.item():.4f} | Intercept: {model.intercept.item():.4f}")
            print(f"  Feature Weights (raw): {np.array2string(raw_weights, precision=4, suppress_small=True)}")

        if epoch_val_auc > best_val_auc:
            best_val_auc = epoch_val_auc
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"  New best validation AUC: {best_val_auc:.4f}. Saving model state.")
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= args.patience:
            print(f"Validation AUC did not improve for {args.patience} epochs. Early stopping.")
            break
    
    print(f"Training finished. Best validation AUC: {best_val_auc:.4f}")

    # Plot metrics
    plot_metrics(metrics_history, args.plot_path)

    # Load the best model state for final embedding generation and saving
    if best_model_state:
        print("Loading best model state...")
        model.load_state_dict(best_model_state)
    
    model.eval()

    if args.output:
        print(f"Computing and saving final embeddings to {args.output}...")
        final_embeddings = model._mixed_embedding().detach().cpu()
        
        # Split embeddings by type
        embeddings_by_type = {}
        for node_type, offset in node_offsets.items():
            count = node_counts[node_type]
            embeddings_by_type[node_type] = final_embeddings[offset : offset + count]

        torch.save(embeddings_by_type, args.output)
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
    parser.add_argument('--training-targets', type=str, nargs='+', default=['A-A', 'A-C', 'A-T', 'C-T'],
                        help="List of link types to train on (e.g., 'A-A', 'C-T'). Data will be balanced across these types.")
    parser.add_argument('--meta-paths', type=str, nargs='+', 
                        default=['AAA', 'ACA', 'ATA', 'CCC', 'CAC', 'CTC', 'TTT', 'TAT', 'TCT'], 
                        help='List of meta-paths to use. Default is a comprehensive set for all node types.')
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
    parser.add_argument('--output', type=str, default='joint_embeddings.pt', help='Path to save final embeddings')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--cache-dir', type=str, default='./matrix_cache', help='Directory to cache computed meta-path matrices.')
    parser.add_argument('--save-model-path', type=str, default='fastrp_model.pth', help='Path to save the trained model checkpoint.')
    parser.add_argument('--edge-split', type=str, default=None, help='Path to the pre-computed edge split file.')
    parser.add_argument('--patience', type=int, default=20, help='Number of epochs to wait for validation AUC improvement before early stopping.')
    parser.add_argument('--plot-path', type=str, default='plots/training_metrics.png', help='Path to save the training metrics plot.')
    
    args = parser.parse_args()
    main(args) 