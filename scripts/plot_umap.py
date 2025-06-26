"""
This script generates a 2D UMAP visualization of node embeddings, typically
for authors in a co-authorship network.

It takes pre-trained embeddings and author labels as input, and can optionally
annotate specific authors by name or ID.
"""
import argparse
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import torch
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from adjustText import adjust_text

from src.data_loader import load_author_mappings, load_dictionary, load_edge_list

def load_area_names(path: Path) -> dict[str, str]:
    """Loads descriptive names for research areas."""
    area_map = {}
    if not path.exists():
        print(f"Warning: Area names file not found at '{path}'. Legend will show numeric IDs.")
        return area_map
    with open(path, 'r', encoding='latin-1') as f:
        for line in f:
            if '->' in line:
                try:
                    name, num = line.strip().split('->')
                    area_map[num.strip()] = name.strip()
                except ValueError:
                    continue
    return area_map

def load_coauthorship(path: Path, n_authors: int) -> sp.csr_matrix:
    """Loads the symmetric co-authorship graph from a 1-based edge list."""
    rows, cols = [], []
    if not path.exists():
        print(f"Warning: Co-authorship file not found at '{path}'. Cannot apply --min-coauthors filter.")
        return None
    with open(path, 'r', encoding='latin-1') as f:
        for line in f:
            try:
                u, v, *w = line.strip().split()
                # Files are 1-based, so subtract 1 for 0-based indexing
                rows.append(int(u) - 1)
                cols.append(int(v) - 1)
            except (ValueError, IndexError):
                continue
    # Symmetrize the matrix by adding both (u, v) and (v, u)
    all_rows = rows + cols
    all_cols = cols + rows
    # Use coo_matrix for efficient creation, then convert to csr
    mat = sp.coo_matrix((np.ones(len(all_rows)), (all_rows, all_cols)), shape=(n_authors, n_authors))
    return mat.tocsr()

def load_labels(label_path: Path) -> tuple[dict[int, str], dict[str, int]]:
    """Loads author labels from a tab-separated file (author_id<tab>label_string)."""
    author_id_to_label_str = {}
    label_str_to_int = {}
    next_label_id = 0
    try:
        with label_path.open('r', encoding='latin-1') as f:
            for line in f:
                try:
                    author_id, label_str = line.strip().split('\t')
                    # Convert from 1-based ID in file to 0-based index
                    author_id = int(author_id) - 1
                    if label_str not in label_str_to_int:
                        label_str_to_int[label_str] = next_label_id
                        next_label_id += 1
                    author_id_to_label_str[author_id] = label_str
                except ValueError:
                    print(f"Warning: Skipping malformed line in label file: {line.strip()}")
                    continue
    except FileNotFoundError:
        print(f"Error: Label file not found at '{label_path}'")
        exit(1)
    return author_id_to_label_str, label_str_to_int

def load_names(names_path: Path) -> dict[int, str]:
    """Loads author names from a tab-separated file (author_id<tab>name)."""
    id_to_name = {}
    try:
        with names_path.open('r', encoding='latin-1') as f:
            for line in f:
                try:
                    author_id, name = line.strip().split('\t')
                    # Convert from 1-based ID in file to 0-based index
                    id_to_name[int(author_id) - 1] = name
                except ValueError:
                    print(f"Warning: Skipping malformed line in names file: {line.strip()}")
                    continue
    except FileNotFoundError:
        print(f"Error: Author names file not found at '{names_path}'")
        exit(1)
    return id_to_name

def main(args):
    """Main function to orchestrate loading, projecting, and plotting."""
    # --- 1. Load Data ---
    print("Loading data...")
    if not args.embeddings.exists():
        print(f"Error: Embeddings file not found at '{args.embeddings}'")
        return
        
    # Load the dictionary of embeddings by type
    embeddings_by_type = torch.load(args.embeddings, map_location='cpu')
    print(f"Loaded embeddings for types: {list(embeddings_by_type.keys())}")

    # Filter to only include types specified in --plot-types
    if args.plot_types:
        embeddings_by_type = {k: v for k, v in embeddings_by_type.items() if k in args.plot_types}
        if not embeddings_by_type:
            print(f"Error: None of the specified plot types {args.plot_types} were found in the embeddings file.")
            return
        print(f"Filtering visualization for types: {list(embeddings_by_type.keys())}")

    # Load all dictionaries for names
    author_id_to_name, _ = load_author_mappings(args.data_dir)
    conf_name_to_id = load_dictionary(args.data_dir / 'conf_dict.txt')
    term_name_to_id = load_dictionary(args.data_dir / 'term_dict.txt')
    conf_id_to_name = {v: k for k, v in conf_name_to_id.items()}
    term_id_to_name = {v: k for k, v in term_name_to_id.items()}
    id_to_name_map = {'A': author_id_to_name, 'C': conf_id_to_name, 'T': term_id_to_name}

    # Combine embeddings and create type-based color mapping
    all_embeddings = []
    point_type_colors = []
    type_color_map = {'A': 'blue', 'C': 'red', 'T': 'green'}
    type_labels = []
    
    current_offset = 0
    node_offsets = {}
    for node_type, embeds in sorted(embeddings_by_type.items()):
        all_embeddings.append(embeds)
        point_type_colors.extend([type_color_map[node_type]] * len(embeds))
        type_labels.append(node_type)
        node_offsets[node_type] = current_offset
        current_offset += len(embeds)
        
    embeddings = np.vstack(all_embeddings)

    # --- 2. Pre-process and Filter ---
    print("Pre-processing data...")
    # L2 normalize embeddings for cosine distance
    embeddings_norm = normalize(embeddings, norm='l2', axis=1)

    # Create a mask to filter authors based on the minimum number of co-authorships
    plot_mask = np.ones(embeddings.shape[0], dtype=bool)
    if args.min_coauthors > 0 and 'A' in node_offsets:
        print(f"Applying filter for authors with at least {args.min_coauthors} co-authorships.")
        author_offset = node_offsets['A']
        n_authors = len(embeddings_by_type['A'])
        
        coauthorship_path = args.data_dir / 'AA.txt'
        adj_matrix = load_edge_list(coauthorship_path, n_authors, n_authors)
        if adj_matrix is not None:
            degrees = np.array(adj_matrix.sum(axis=1)).flatten()
            author_mask = degrees >= args.min_coauthors
            plot_mask[author_offset : author_offset + n_authors] = author_mask
            print(f"  {author_mask.sum()} of {n_authors} authors will be shown.")

    # --- 3. UMAP & PCA Projections ---
    print("Performing UMAP projection...")
    umap_params = {'n_neighbors': 15, 'min_dist': 0.5, 'metric': "euclidean"}
    reducer = umap.UMAP(**umap_params)
    
    start_time = time.time()
    embeddings_umap = reducer.fit_transform(embeddings_norm)
    end_time = time.time()
    print(f"UMAP projection took {end_time - start_time:.2f} seconds.")

    print("Performing PCA projection...")
    pca = PCA(n_components=3)
    start_time = time.time()
    embeddings_pca = pca.fit_transform(embeddings_norm)
    end_time = time.time()
    explained_variance = pca.explained_variance_ratio_
    print(f"PCA projection took {end_time - start_time:.2f} seconds.")
    print(f"PCA Explained Variance: PC1={explained_variance[0]:.2%}, PC2={explained_variance[1]:.2%}, PC3={explained_variance[2]:.2%}, Total={explained_variance.sum():.2%}")

    # --- 4. Annotations & Visualization ---
    # Create a global mapping from ID to name for annotations
    global_id_to_name = {}
    for node_type, id_map in id_to_name_map.items():
        if node_type in node_offsets:
            offset = node_offsets[node_type]
            for local_id, name in id_map.items():
                global_id_to_name[local_id + offset] = name

    # Randomly select nodes to annotate from each type
    annotation_ids = []
    for node_type in embeddings_by_type.keys():
        offset = node_offsets[node_type]
        count = len(embeddings_by_type[node_type])
        num_to_sample = min(10, count)
        if num_to_sample > 0:
            # Get indices of nodes that are not filtered out by the co-authorship mask
            possible_indices = np.where(plot_mask[offset:offset+count])[0]
            if len(possible_indices) > 0:
                num_to_sample = min(num_to_sample, len(possible_indices))
                random_local_ids = np.random.choice(possible_indices, size=num_to_sample, replace=False)
                annotation_ids.extend(list(random_local_ids + offset))
    print(f"Randomly selected {len(annotation_ids)} nodes to annotate.")

    print("Generating plot...")
    fig, axes = plt.subplots(2, 2, figsize=(22, 18))
    fig.suptitle(f'Joint Embedding Visualization ({", ".join(type_labels)})', fontsize=16)

    # --- Plot UMAP (top-left) ---
    ax1 = axes[0, 0]
    umap_title = f"UMAP Projection"
    ax1.set_title(umap_title, fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    handles = []
    for node_type, color in type_color_map.items():
        if node_type in embeddings_by_type:
            offset = node_offsets[node_type]
            count = len(embeddings_by_type[node_type])
            type_mask = np.zeros(embeddings.shape[0], dtype=bool)
            type_mask[offset:offset+count] = True
            
            # Combine with the plot_mask to respect filtering
            final_mask = type_mask & plot_mask

            ax1.scatter(embeddings_umap[final_mask, 0], embeddings_umap[final_mask, 1], s=10, color=color, label=f"Type: {node_type}", alpha=0.3)
            # Create a handle for the legend
            handles.append(plt.Line2D([0], [0], marker='o', color='w', label=f"Type: {node_type}",
                                      markerfacecolor=color, markersize=10))

    # Add annotations to UMAP plot
    if annotation_ids:
        texts = []
        for node_id in annotation_ids:
            # Check if the node to be annotated wasn't filtered out
            if plot_mask[node_id]:
                label = global_id_to_name.get(node_id, f"ID: {node_id}")
                texts.append(ax1.text(embeddings_umap[node_id, 0], embeddings_umap[node_id, 1], label,
                                        fontsize=9, ha='center',
                                        bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.1')))
        if texts:
            adjust_text(texts, ax=ax1, arrowprops=dict(arrowstyle='->', color='black', lw=0.5))

    ax1.set_xlabel("UMAP Dimension 1")
    ax1.set_ylabel("UMAP Dimension 2")
    ax1.legend(handles=handles, title="Node Types")

    # This part needs the colors as an array for c argument
    num_embeddings = embeddings.shape[0]
    point_colors = np.array(point_type_colors)

    # --- Plot PCA PC1 vs PC2 (top-right) ---
    ax2 = axes[0, 1]
    ax2.set_title("PCA Projection", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.scatter(embeddings_pca[plot_mask, 0], embeddings_pca[plot_mask, 1], c=point_colors[plot_mask], s=10, alpha=0.3)
    ax2.set_xlabel(f"PC 1 ({explained_variance[0]:.2%})")
    ax2.set_ylabel(f"PC 2 ({explained_variance[1]:.2%})")

    # --- Plot PCA PC2 vs PC3 (bottom-left) ---
    ax3 = axes[1, 0]
    ax3.set_title("PCA Projection", fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.scatter(embeddings_pca[plot_mask, 1], embeddings_pca[plot_mask, 2], c=point_colors[plot_mask], s=10, alpha=0.3)
    ax3.set_xlabel(f"PC 2 ({explained_variance[1]:.2%})")
    ax3.set_ylabel(f"PC 3 ({explained_variance[2]:.2%})")

    # --- Plot PCA PC1 vs PC3 (bottom-right) ---
    ax4 = axes[1, 1]
    ax4.set_title("PCA Projection", fontsize=12)
    ax4.grid(True, linestyle='--', alpha=0.6)
    ax4.scatter(embeddings_pca[plot_mask, 0], embeddings_pca[plot_mask, 2], c=point_colors[plot_mask], s=10, alpha=0.3)
    ax4.set_xlabel(f"PC 1 ({explained_variance[0]:.2%})")
    ax4.set_ylabel(f"PC 3 ({explained_variance[2]:.2%})")

    # --- 5. Final Touches & Save/Show ---
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the plot
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=300)
    print(f"Plot saved to '{args.output}'")
    
    # Optionally display the plot
    if not args.no_show:
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize joint embeddings with UMAP and PCA.")
    parser.add_argument('--data-dir', type=Path, default='data',
                        help="Path to the data directory, needed for name lookups.")
    parser.add_argument('--embeddings', type=Path, required=True,
                        help="Path to the saved joint embeddings file (*.pt).")
    parser.add_argument('--min-coauthors', type=int, default=0,
                        help="Minimum number of co-authorships required for an author to be visualized. Set to 0 to disable.")
    parser.add_argument('--output', type=Path, default='plots/umap_visualization.png',
                        help="Path to save the output plot.")
    parser.add_argument('--plot-types', type=str, nargs='+', default=['A', 'C', 'T'],
                        help="List of node types to include in the plot (e.g., 'A' 'C').")
    parser.add_argument('--no-show', action='store_true',
                        help="If set, the script will save the plot but not display it.")
    
    main(parser.parse_args()) 