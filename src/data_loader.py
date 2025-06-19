from pathlib import Path
import numpy as np
import scipy.sparse as sp
from typing import Union, Dict, Tuple, List

def load_dictionary(path: Path):
    """Loads a dictionary from a file (one item per line)."""
    with open(path, 'r', encoding='latin-1') as f:
        return {line.strip(): i for i, line in enumerate(f)}

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
                # Handle empty or malformed lines
                continue
    # Use coo_matrix for efficient creation and then convert to csr
    mat = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_rows, n_cols))
    return mat.tocsr()

def load_data(data_dir: Union[str, Path] = 'data'):
    """Loads the entire bibliographic dataset."""
    data_dir = Path(data_dir)
    print("Loading data...")

    # Load dictionaries to get entity counts
    author_dict = load_dictionary(data_dir / 'author_dict.txt')
    conf_dict = load_dictionary(data_dir / 'conf_dict.txt')
    term_dict = load_dictionary(data_dir / 'term_dict.txt')

    n_authors = len(author_dict)
    n_confs = len(conf_dict)
    n_terms = len(term_dict)

    print(f"  {n_authors} authors, {n_confs} conferences, {n_terms} terms.")

    # Define relations and their shapes based on available files
    relation_files = {
        ("A", "A"): ("AA.txt", (n_authors, n_authors)),
        ("A", "T"): ("AT.txt", (n_authors, n_terms)),
        ("C", "A"): ("CA.txt", (n_confs, n_authors)),
        ("C", "T"): ("CT.txt", (n_confs, n_terms)),
    }

    relations = {}
    for (src_type, dst_type), (filename, shape) in relation_files.items():
        path = data_dir / filename
        if path.exists():
            if src_type not in relations:
                relations[src_type] = {}
            relations[src_type][dst_type] = load_edge_list(path, shape[0], shape[1])

    # Create transpose relations
    if "A" in relations and "T" in relations["A"]:
        if "T" not in relations: relations["T"] = {}
        relations["T"]["A"] = relations["A"]["T"].T.tocsr()
    
    if "C" in relations and "A" in relations["C"]:
        if "A" not in relations: relations["A"] = {}
        relations["A"]["C"] = relations["C"]["A"].T.tocsr()
    
    if "C" in relations and "T" in relations["C"]:
        if "T" not in relations: relations["T"] = {}
        relations["T"]["C"] = relations["C"]["T"].T.tocsr()

    # Add identity to author-author matrix
    if "A" in relations and "A" in relations["A"]:
        relations["A"]["A"] = (relations["A"]["A"] + sp.eye(n_authors)).tocsr()
    else:
        if "A" not in relations: relations["A"] = {}
        relations["A"]["A"] = sp.eye(n_authors).tocsr()

    print("Data loading complete.")
    return relations, n_authors, n_confs, n_terms 

def load_sparse_matrix(filepath: Path, shape: Tuple[int, int]) -> sp.csr_matrix:
    """Loads a sparse matrix from a text file."""
    rows, cols, data = [], [], []
    with open(filepath, 'r', encoding='latin-1') as f:
        for line in f:
            r, c, w = map(int, line.strip().split())
            rows.append(r - 1)  # 1-based to 0-based
            cols.append(c - 1)
            data.append(w)
    return sp.csr_matrix((data, (rows, cols)), shape=shape, dtype=np.float32)

def compute_meta_path_matrix(path_str: str, relations: Dict[str, sp.csr_matrix]):
    """
    Computes a single sparse meta-path matrix using scipy.
    Handles both standard paths (e.g., 'ACA') and composite paths ('ACA@ATA').
    """
    print(f"  Computing adjacency matrix for: {path_str}")
    
    if '@' in path_str:
        left_str, right_str = path_str.split('@', 1)
        left_mat = compute_meta_path_matrix(left_str, relations)
        right_mat = compute_meta_path_matrix(right_str, relations)
        if left_mat.shape != right_mat.shape:
            raise ValueError(f"Shape mismatch for element-wise product in '{path_str}': {left_mat.shape} vs {right_mat.shape}")
        return left_mat.multiply(right_mat)
    else:
        parts = list(path_str)
        final_mat = relations[parts[0] + parts[1]].copy()
        for i in range(1, len(parts) - 1):
            next_mat = relations[parts[i] + parts[i+1]]
            final_mat = final_mat @ next_mat
        return final_mat

def load_and_preprocess_data(data_dir: str):
    """
    Loads all data, creates a global mapping, and computes all necessary adjacency matrices.
    """
    data_path = Path(data_dir)
    
    # 1. Load node counts
    n_authors = int(open(data_path / 'author_num.txt').read())
    n_conferences = int(open(data_path / 'conf_num.txt').read())
    n_terms = int(open(data_path / 'term_num.txt').read())
    
    total_nodes = n_authors + n_conferences + n_terms
    print(f"Total nodes: {total_nodes} (Authors: {n_authors}, Conf: {n_conferences}, Terms: {n_terms})")

    # 2. Create global ID mappings
    node_offsets = {
        'A': 0,
        'C': n_authors,
        'T': n_authors + n_conferences
    }

    # 3. Load base heterogeneous relation matrices
    relations = {
        'AC': load_sparse_matrix(data_path / 'AC.txt', (n_authors, n_conferences)),
        'AT': load_sparse_matrix(data_path / 'AT.txt', (n_authors, n_terms)),
        'CA': load_sparse_matrix(data_path / 'CA.txt', (n_conferences, n_authors)),
        'CT': load_sparse_matrix(data_path / 'CT.txt', (n_conferences, n_terms)),
    }
    relations['TC'] = relations['CT'].T
    relations['TA'] = relations['AT'].T

    # 4. Compute all homogeneous adjacency matrices from specified meta-paths
    author_meta_paths = ['AAA', 'ACA', 'ATA']
    conf_meta_paths = ['CAC', 'CTC', 'CATAC']
    term_meta_paths = ['TAT', 'TCT']

    # We need A-A for evaluation, let's derive it from C-A co-occurrence
    relations['AA'] = relations['AC'] @ relations['CA']
    
    adj_matrices = {
        'A': [compute_meta_path_matrix(p, relations) for p in author_meta_paths],
        'C': [compute_meta_path_matrix(p, relations) for p in conf_meta_paths],
        'T': [compute_meta_path_matrix(p, relations) for p in term_meta_paths]
    }

    return total_nodes, n_authors, node_offsets, adj_matrices, relations

def get_training_pairs(adj: sp.csr_matrix, offset: int, num_neg_samples: int):
    """Generates positive and negative training pairs with global offsets."""
    rng = np.random.default_rng(42)
    pos_pairs = np.array(adj.nonzero()).T
    
    # Apply global offset
    pos_pairs += offset
    
    n_nodes = adj.shape[0]
    n_pos = len(pos_pairs)
    n_neg = n_pos * num_neg_samples
    
    neg_pairs = np.zeros((n_neg, 2), dtype=int)
    for i in range(n_neg):
        while True:
            u = rng.integers(0, n_nodes)
            v = rng.integers(0, n_nodes)
            if u != v and not adj[u, v]:
                neg_pairs[i, 0] = u + offset
                neg_pairs[i, 1] = v + offset
                break
                
    return pos_pairs, neg_pairs 