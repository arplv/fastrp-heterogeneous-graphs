import argparse
from pathlib import Path
import numpy as np
import torch
import scipy.sparse as sp

def load_edge_list(path: Path, n_rows: int, n_cols: int):
    """Loads an edge list from a file (u v w per line, 1-based)."""
    rows, cols = [], []
    with open(path, 'r', encoding='latin-1') as f:
        for line in f:
            try:
                u, v, *w = line.strip().split()
                # Files are 1-based, so subtract 1
                rows.append(int(u) - 1)
                cols.append(int(v) - 1)
            except (ValueError, IndexError):
                continue
    # Use coo_matrix for efficient creation and then convert to csr
    mat = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_rows, n_cols))
    return mat.tocsr()

def load_dictionary(path: Path):
    """Loads a dictionary from a file (one item per line)."""
    with open(path, 'r', encoding='latin-1') as f:
        return {line.strip(): i for i, line in enumerate(f)}

def main(args):
    """Splits positive edges from the main adjacency matrix into train and validation sets."""
    np.random.seed(args.seed)
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading author dictionary to determine graph size...")
    author_dict = load_dictionary(data_dir / 'author_dict.txt')
    n_authors = len(author_dict)
    print(f"Found {n_authors} authors.")

    print(f"Loading co-authorship matrix from '{data_dir / 'AA.txt'}'...")
    adj_matrix = load_edge_list(data_dir / 'AA.txt', n_authors, n_authors)
    
    # Use the upper triangle to get unique positive edges
    adj_triu = sp.triu(adj_matrix, k=1)
    pos_edges = np.array(adj_triu.nonzero()).T
    
    print(f"Found {pos_edges.shape[0]} unique positive edges.")

    print(f"Shuffling edges and splitting with validation fraction: {args.val_frac}")
    np.random.shuffle(pos_edges)
    
    val_size = int(pos_edges.shape[0] * args.val_frac)
    
    val_edges = torch.from_numpy(pos_edges[:val_size]).long().T
    train_edges = torch.from_numpy(pos_edges[val_size:]).long().T
    
    print(f"  Train set size: {train_edges.size(1)}")
    print(f"  Validation set size: {val_edges.size(1)}")

    split_data = {
        'train_pos_edge_index': train_edges,
        'val_pos_edge_index': val_edges
    }
    
    output_path = output_dir / args.output_file
    torch.save(split_data, output_path)
    print(f"Saved edge split to '{output_path.resolve()}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create a train/validation split for positive edges.")
    parser.add_argument('--data-dir', type=str, default='data', help='Directory containing AA.txt and author_dict.txt')
    parser.add_argument('--val-frac', type=float, default=0.1, help='Fraction of edges to use for the validation set.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--output-dir', type=str, default='splits', help='Directory to save the output split file.')
    parser.add_argument('--output-file', type=str, default='edge_split.pt', help='Name for the output file.')
    
    args = parser.parse_args()
    main(args) 