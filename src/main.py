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
    
    # Apply node filtering based on minimum degree
    if args.min_degree > 0:
        print(f"Filtering nodes with degree < {args.min_degree}...")
        
        # Calculate total degree for each author (sum across all relation types)
        total_degrees = np.zeros(n_authors)
        for dst_type, matrix in relations.get('A', {}).items():
            if dst_type == 'A':
                # For author-author, count both directions but avoid double-counting diagonal
                degrees = np.array(matrix.sum(axis=1)).flatten()
                # Subtract identity matrix contribution to avoid inflating degrees
                identity_contrib = np.array(sp.eye(n_authors).sum(axis=1)).flatten()
                total_degrees += degrees - identity_contrib
            else:
                # For author-conference, author-term relationships
                total_degrees += np.array(matrix.sum(axis=1)).flatten()
        
        # Create mask for nodes to keep
        valid_nodes = total_degrees >= args.min_degree
        valid_indices = np.where(valid_nodes)[0]
        n_filtered = np.sum(valid_nodes)
        
        print(f"  Original authors: {n_authors}")
        print(f"  Filtered authors: {n_filtered} (removed {n_authors - n_filtered} low-degree nodes)")
        
        # Filter relation matrices
        filtered_relations = {}
        for src_type, dst_relations in relations.items():
            if src_type not in filtered_relations:
                filtered_relations[src_type] = {}
            for dst_type, matrix in dst_relations.items():
                if src_type == 'A' and dst_type == 'A':
                    # Filter both rows and columns for author-author matrix
                    filtered_matrix = matrix[valid_indices, :][:, valid_indices]
                elif src_type == 'A':
                    # Filter only rows for author-X matrices
                    filtered_matrix = matrix[valid_indices, :]
                elif dst_type == 'A':
                    # Filter only columns for X-author matrices
                    filtered_matrix = matrix[:, valid_indices]
                else:
                    # Keep other matrices unchanged
                    filtered_matrix = matrix
                
                filtered_relations[src_type][dst_type] = filtered_matrix.tocsr()
        
        # Update relations and author count
        relations = filtered_relations
        n_authors = n_filtered
        
        # Store the mapping from original indices to filtered indices for later use
        original_to_filtered = {orig_idx: new_idx for new_idx, orig_idx in enumerate(valid_indices)}
        filtered_to_original = {new_idx: orig_idx for new_idx, orig_idx in enumerate(valid_indices)}
    else:
        print("No degree filtering applied (min-degree=0)")
        original_to_filtered = {i: i for i in range(n_authors)}
        filtered_to_original = {i: i for i in range(n_authors)}

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

    # EXPÉRIMENTATION : désactiver la pente (lambda) et l'intercept (gamma)
    model.slope.data.fill_(1.0)
    model.intercept.data.fill_(0.0)
    model.slope.requires_grad_(False)
    model.intercept.requires_grad_(False)
    print("[Expérience] Slope (lambda) fixé à 1.0 et intercept (gamma) fixé à 0.0 — non entraînables.")

    # Prepare training data: positive edges
    if args.edge_split:
        print(f"Loading edge split from {args.edge_split}")
        try:
            split_data = torch.load(args.edge_split, weights_only=False)
        except TypeError:
            # Fallback for older PyTorch versions
            split_data = torch.load(args.edge_split)
        train_pos_edge_index = split_data['train_pos_edge_index']
        val_pos_edge_index = split_data['val_pos_edge_index']
        
        # If we filtered nodes, we need to remap the edge indices
        if args.min_degree > 0:
            print("  Remapping edge indices for filtered nodes...")
            
            # Filter and remap training edges
            train_mask = torch.zeros(train_pos_edge_index.size(1), dtype=torch.bool)
            for i in range(train_pos_edge_index.size(1)):
                src, dst = train_pos_edge_index[0, i].item(), train_pos_edge_index[1, i].item()
                if src in original_to_filtered and dst in original_to_filtered:
                    train_pos_edge_index[0, i] = original_to_filtered[src]
                    train_pos_edge_index[1, i] = original_to_filtered[dst]
                    train_mask[i] = True
            
            train_pos_edge_index = train_pos_edge_index[:, train_mask]
            
            # Filter and remap validation edges
            val_mask = torch.zeros(val_pos_edge_index.size(1), dtype=torch.bool)
            for i in range(val_pos_edge_index.size(1)):
                src, dst = val_pos_edge_index[0, i].item(), val_pos_edge_index[1, i].item()
                if src in original_to_filtered and dst in original_to_filtered:
                    val_pos_edge_index[0, i] = original_to_filtered[src]
                    val_pos_edge_index[1, i] = original_to_filtered[dst]
                    val_mask[i] = True
            
            val_pos_edge_index = val_pos_edge_index[:, val_mask]
        
        train_pos_edge_index = train_pos_edge_index.to(model_device)
        val_pos_edge_index = val_pos_edge_index.to(model_device)
        print(f"  Train positive edges: {train_pos_edge_index.size(1)}")
        print(f"  Validation positive edges: {val_pos_edge_index.size(1)}")
    else:
        print("No edge split provided. Using all positive edges for training and validation.")
        train_adj = relations['A']['A']
        train_adj.setdiag(0)
        train_adj.eliminate_zeros()
        all_pos_edges = torch.from_numpy(np.array(sp.triu(train_adj, k=1).nonzero())).long()
        # Note: After filtering, the edge indices are already in the correct (filtered) space
        # Simple split for fallback
        perm = torch.randperm(all_pos_edges.size(1))
        val_size = int(all_pos_edges.size(1) * 0.1)
        train_pos_edge_index = all_pos_edges[:, perm[val_size:]].to(model_device)
        val_pos_edge_index = all_pos_edges[:, perm[:val_size]].to(model_device)

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
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
                num_nodes=(n_authors, n_authors),  # Specify as tuple for symmetric graph
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
            
            weights_softmax = F.softmax(model.feature_weights, dim=1)
            entropy = -torch.sum(weights_softmax * torch.log(weights_softmax + 1e-7))
            
            loss = bce_loss + args.lambda_entropy * entropy
            
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
                    num_nodes=(n_authors, n_authors),  # Specify as tuple for symmetric graph
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
            weights_softmax = F.softmax(model.feature_weights.flatten(), dim=0).cpu().numpy()
            print(f"  Slope: {model.slope.item():.4f} | Intercept: {model.intercept.item():.4f}")
            print(f"  Feature Weights (softmax): {np.array2string(weights_softmax, precision=4, suppress_small=True)}")

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

    # Save final model and embeddings if requested
    if args.save_final:
        print("Saving final epoch model and embeddings...")
        
        # Save final model state (current state)
        final_args_dict = {k: v for k, v in vars(args).items() if k != 'relations'}
        final_checkpoint = {
            'args': final_args_dict,
            'model_state_dict': model.state_dict(),
            'epoch': len(metrics_history['train_loss']),
            'final_train_loss': metrics_history['train_loss'][-1],
            'final_val_auc': metrics_history['val_auc'][-1] if metrics_history['val_auc'] else None
        }
        torch.save(final_checkpoint, args.final_model_path)
        print(f"  Final model saved to {args.final_model_path}")
        
        # Save final embeddings (current state)
        model.eval()
        final_embeddings = model._mixed_embedding().detach().cpu()
        
        final_embedding_data = {
            'embeddings': final_embeddings,
            'filtered_to_original': filtered_to_original if args.min_degree > 0 else None,
            'original_to_filtered': original_to_filtered if args.min_degree > 0 else None,
            'min_degree_threshold': args.min_degree,
            'n_original_authors': len(original_to_filtered) if args.min_degree > 0 else n_authors,
            'n_filtered_authors': n_authors,
            'epoch': len(metrics_history['train_loss']),
            'final_train_loss': metrics_history['train_loss'][-1],
            'final_val_auc': metrics_history['val_auc'][-1] if metrics_history['val_auc'] else None,
            'is_final_epoch': True
        }
        
        torch.save(final_embedding_data, args.final_embeddings_path)
        print(f"  Final embeddings saved to {args.final_embeddings_path}")

    # Load the best model state for best embedding generation and saving
    if best_model_state:
        print("Loading best model state for best embeddings...")
        model.load_state_dict(best_model_state)
    
    model.eval()

    if args.output:
        print(f"Computing and saving final embeddings to {args.output}...")
        final_embeddings = model._mixed_embedding().detach().cpu()
        
        # Create embedding dictionary with metadata
        embedding_data = {
            'embeddings': final_embeddings,
            'filtered_to_original': filtered_to_original if args.min_degree > 0 else None,
            'original_to_filtered': original_to_filtered if args.min_degree > 0 else None,
            'min_degree_threshold': args.min_degree,
            'n_original_authors': len(original_to_filtered) if args.min_degree > 0 else n_authors,
            'n_filtered_authors': n_authors,
            'best_val_auc': best_val_auc,
            'is_best_model': True
        }
        
        torch.save(embedding_data, args.output)
        print("Best embeddings saved.")

    if args.save_model_path:
        print(f"Saving model checkpoint to {args.save_model_path}...")
        args_dict = {k: v for k, v in vars(args).items() if k != 'relations'}
        checkpoint = {
            'args': args_dict,
            'model_state_dict': model.state_dict(),
            'best_val_auc': best_val_auc,
            'is_best_model': True
        }
        torch.save(checkpoint, args.save_model_path)
        print("Best model saved.")


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
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='Optimizer to use for training.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer (ignored for Adam).')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='Weight decay (L2 regularization) coefficient.')
    parser.add_argument('--lambda-entropy', type=float, default=0.0, help='Coefficient for entropy regularization. Set to 0 to disable, or a small value like 1e-4 to encourage diversity.')
    parser.add_argument('--neg-samples', type=int, default=3, help='Number of negative samples per positive sample.')
    parser.add_argument('--batch-size', type=int, default=4096, help='Training batch size')
    parser.add_argument('--device', type=str, default='auto', help='Device to use for training (e.g., "cpu", "cuda", "mps").')
    parser.add_argument('--output', type=str, default='author_embeddings.pt', help='Path to save final embeddings')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--cache-dir', type=str, default='./matrix_cache', help='Directory to cache computed meta-path matrices.')
    parser.add_argument('--save-model-path', type=str, default='fastrp_model.pth', help='Path to save the trained model checkpoint.')
    parser.add_argument('--edge-split', type=str, default=None, help='Path to the pre-computed edge split file.')
    parser.add_argument('--patience', type=int, default=20, help='Number of epochs to wait for validation AUC improvement before early stopping.')
    parser.add_argument('--plot-path', type=str, default='plots/training_metrics.png', help='Path to save the training metrics plot.')
    parser.add_argument('--min-degree', type=int, default=0, help='Minimum degree (total connections) required to include a node in training. Set to 0 to include all nodes.')
    parser.add_argument('--save-final', action='store_true', help='Save embeddings and model from the final epoch (in addition to best model).')
    parser.add_argument('--final-embeddings-path', type=str, default='final_embeddings.pt', help='Path to save final epoch embeddings.')
    parser.add_argument('--final-model-path', type=str, default='final_model.pth', help='Path to save final epoch model checkpoint.')
    
    args = parser.parse_args()
    main(args) 