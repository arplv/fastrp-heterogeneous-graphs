import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from pathlib import Path
from tqdm import tqdm
import random

from data_loader import load_and_preprocess_data
from model import JointEmbeddingModel

def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == 'auto' else "cpu")
    print(f"Using device: {device}")

    # Load and process data
    total_nodes, n_authors, node_offsets, adj_matrices, relations = load_and_preprocess_data(
        args.data_dir, 
        use_gpu=(device.type == 'cuda')
    )

    # Initialize model
    model = JointEmbeddingModel(
        num_total_nodes=total_nodes,
        embedding_dim=args.dim
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # --- Training Data Preparation ---
    # We will train on all direct heterogeneous and homogeneous relationships
    training_relations = {
        # Heterogeneous
        'AC': (relations['AC'], node_offsets['A'], node_offsets['C']),
        'AT': (relations['AT'], node_offsets['A'], node_offsets['T']),
        'CT': (relations['CT'], node_offsets['C'], node_offsets['T']),
        # Homogeneous (using the first pre-computed meta-path for each)
        'AA': (adj_matrices['A'][0], node_offsets['A'], node_offsets['A']),
        'CC': (adj_matrices['C'][0], node_offsets['C'], node_offsets['C']),
        'TT': (adj_matrices['T'][0], node_offsets['T'], node_offsets['T']),
    }
    
    # --- Evaluation Data Preparation ---
    # We evaluate on the author-author link prediction task
    eval_adj = relations['AA']
    eval_pos_pairs = np.array(eval_adj.nonzero()).T
    
    rng = np.random.default_rng(args.seed)
    num_eval_neg = len(eval_pos_pairs)
    eval_neg_pairs = []
    while len(eval_neg_pairs) < num_eval_neg:
        u, v = rng.integers(0, n_authors, size=2)
        if u != v and not eval_adj[u, v]:
            eval_neg_pairs.append([u, v])
    eval_neg_pairs = np.array(eval_neg_pairs)

    eval_pairs = np.vstack([eval_pos_pairs, eval_neg_pairs])
    eval_labels = np.array([1] * len(eval_pos_pairs) + [0] * len(eval_neg_pairs))

    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        
        # Shuffle the relation types to train on each epoch
        relation_keys = list(training_relations.keys())
        random.shuffle(relation_keys)

        for rel_key in tqdm(relation_keys, desc=f"Epoch {epoch+1}/{args.epochs}"):
            adj, offset1, offset2 = training_relations[rel_key]
            
            pos_pairs = np.array(adj.nonzero()).T
            if len(pos_pairs) == 0:
                continue

            # Simple negative sampling for this training step
            num_neg = len(pos_pairs) * args.neg_samples
            neg_pairs = np.zeros((num_neg, 2), dtype=int)
            n_nodes1, n_nodes2 = adj.shape
            
            for i in range(num_neg):
                while True:
                    u = rng.integers(0, n_nodes1)
                    v = rng.integers(0, n_nodes2)
                    if not adj[u,v]:
                        neg_pairs[i, 0] = u
                        neg_pairs[i, 1] = v
                        break

            # Combine, apply global offsets, and create labels
            train_pairs = np.vstack([pos_pairs, neg_pairs])
            train_labels = torch.tensor([1.0] * len(pos_pairs) + [0.0] * len(neg_pairs), device=device)
            
            idx_i = torch.tensor(train_pairs[:, 0] + offset1, device=device)
            idx_j = torch.tensor(train_pairs[:, 1] + offset2, device=device)

            optimizer.zero_grad()
            preds = model(idx_i, idx_j)
            loss = F.binary_cross_entropy(preds.squeeze(), train_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # --- Evaluation ---
        model.eval()
        print(f"\n--- Evaluating Epoch {epoch+1}/{args.epochs} ---")
        
        evaluation_tasks = {
            "A-A": (relations['AA'], node_offsets['A'], node_offsets['A']),
            "A-C": (relations['AC'], node_offsets['A'], node_offsets['C']),
            "A-T": (relations['AT'], node_offsets['A'], node_offsets['T']),
            "C-T": (relations['CT'], node_offsets['C'], node_offsets['T']),
        }

        for task_name, (adj, offset1, offset2) in evaluation_tasks.items():
            with torch.no_grad():
                pos_pairs = np.array(adj.nonzero()).T
                if len(pos_pairs) == 0:
                    continue

                num_neg = len(pos_pairs)
                neg_pairs = []
                n1, n2 = adj.shape
                while len(neg_pairs) < num_neg:
                    u, v = rng.integers(0, n1), rng.integers(0, n2)
                    if not adj[u, v]:
                        neg_pairs.append([u, v])
                
                eval_pairs = np.vstack([pos_pairs, np.array(neg_pairs)])
                eval_labels = np.array([1] * len(pos_pairs) + [0] * len(neg_pairs))

                eval_idx_i = torch.tensor(eval_pairs[:, 0] + offset1, device=device)
                eval_idx_j = torch.tensor(eval_pairs[:, 1] + offset2, device=device)
                
                preds = model(eval_idx_i, eval_idx_j).cpu().numpy()
                
                auc = roc_auc_score(eval_labels, preds)
                binary_preds = (preds > 0.5).astype(int)
                precision = precision_score(eval_labels, binary_preds, zero_division=0)
                recall = recall_score(eval_labels, binary_preds, zero_division=0)

                print(f"  {task_name} Eval | AUC: {auc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
        print("-" * (25 + len(str(epoch+1)) + len(str(args.epochs))))

    # Save final embeddings
    print("Saving final embeddings...")
    final_embeddings = model.embeddings.weight.detach().cpu().numpy()
    np.save("joint_embeddings.npy", final_embeddings)
    # Also save author-specific embeddings for compatibility
    author_embeds = final_embeddings[:n_authors]
    np.save("author_embeddings.npy", author_embeds)
    print("Embeddings saved to joint_embeddings.npy and author_embeddings.npy")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Joint Embedding Model for Heterogeneous Graphs")
    parser.add_argument('--data-dir', type=str, default='data', help='Directory containing the dataset')
    parser.add_argument('--dim', type=int, default=128, help='Embedding dimension.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--neg-samples', type=int, default=3, help='Number of negative samples per positive sample.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--device', type=str, default='auto', help="Device to use ('auto', 'cpu', 'cuda')")
    args = parser.parse_args()
    main(args) 