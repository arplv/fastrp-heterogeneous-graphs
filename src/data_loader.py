from pathlib import Path
import numpy as np
import scipy.sparse as sp
from typing import Union

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

def load_author_mappings(data_dir: Union[str, Path]):
    """Loads author names and creates id-to-name and name-to-id mappings."""
    data_dir = Path(data_dir)
    id_to_name = {}
    name_to_id = {}
    with open(data_dir / 'author.txt', 'r', encoding='latin-1') as f:
        for line in f:
            try:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    author_id, author_name = parts
                    # Convert from 1-based ID in file to 0-based internal ID
                    zero_based_id = int(author_id) - 1
                    id_to_name[zero_based_id] = author_name
                    name_to_id[author_name.lower()] = zero_based_id
            except (ValueError, IndexError):
                continue
    return id_to_name, name_to_id

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