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

def plot_2d_chart(ax, title, embeddings_2d, point_colors, plot_mask, label_str_to_int, area_names, id_to_name, annotate_ids, type_colors=None):
    """Helper to draw a 2D scatter plot on a given axes."""
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.grid(True, linestyle='--', alpha=0.6)

    # Use specified type_colors if provided (for UMAP), else a default cmap (for PCA)
    cmap = type_colors if type_colors else plt.get_cmap('tab20', len(label_str_to_int))

    # Plot unlabeled points
    unlabeled_mask = (point_colors == -1) & plot_mask
    ax.scatter(embeddings_2d[unlabeled_mask, 0], embeddings_2d[unlabeled_mask, 1], s=5, color='lightgray', alpha=0.6)

    # Plot labeled points
    handles, labels = [], []
    sorted_labels = sorted(label_str_to_int.items(), key=lambda item: area_names.get(item[0], item[0]))

    for label_str, label_int in sorted_labels:
        mask = (point_colors == label_int) & plot_mask
        if np.any(mask):
            legend_label = area_names.get(label_str, label_str)
            scatter = ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], s=10, color=cmap(label_int), label=legend_label, alpha=0.8)
            handles.append(scatter)
            labels.append(legend_label)

    # Annotations
    if annotate_ids:
        texts = []
        num_embeddings = embeddings_2d.shape[0]
        for raw_author_id in annotate_ids:
            author_id = raw_author_id - 1
            if 0 <= author_id < num_embeddings and plot_mask[author_id]:
                label = id_to_name.get(author_id, f"ID: {raw_author_id}")
                texts.append(ax.text(embeddings_2d[author_id, 0], embeddings_2d[author_id, 1], label,
                                     fontsize=9, ha='center',
                                     bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.1')))
            else:
                print(f"Warning: Annotation ID {raw_author_id} is out of bounds or filtered. Skipping.")
        
        if texts:
            adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='->', color='black', lw=0.5))
    
    return handles, labels

def plot_3d_chart(ax, title, embeddings_3d, point_colors, plot_mask, label_str_to_int, area_names):
    """Helper to draw a 3D scatter plot on a given axes."""
    ax.set_title(title, fontsize=12)

    cmap = plt.get_cmap('tab20', len(label_str_to_int))

    # Plot unlabeled points
    unlabeled_mask = (point_colors == -1) & plot_mask
    ax.scatter(embeddings_3d[unlabeled_mask, 0], embeddings_3d[unlabeled_mask, 1], embeddings_3d[unlabeled_mask, 2],
               s=5, color='lightgray', alpha=0.4)

    # Plot labeled points
    sorted_labels = sorted(label_str_to_int.items(), key=lambda item: area_names.get(item[0], item[0]))
    for label_str, label_int in sorted_labels:
        mask = (point_colors == label_int) & plot_mask
        if np.any(mask):
            legend_label = area_names.get(label_str, label_str)
            ax.scatter(embeddings_3d[mask, 0], embeddings_3d[mask, 1], embeddings_3d[mask, 2],
                       s=10, color=cmap(label_int), label=legend_label, alpha=0.8)
    
    # Annotations are omitted in 3D for clarity

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

    # --- 4. Visualization ---
    print("Generating plot...")
    fig, axes = plt.subplots(2, 2, figsize=(22, 18))

    # --- Set colors based on node type for all plots ---
    # This overrides the author-label coloring for simplicity in the joint view.
    # The original author labels are not used, we color by A, C, T.
    num_embeddings = embeddings.shape[0]
    point_colors = np.array(point_type_colors) # Use the type colors directly

    # Plot UMAP (top-left)
    ax1 = axes[0, 0]
    umap_title = f"UMAP Projection of {', '.join(type_labels)}"
    ax1.set_title(umap_title, fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    handles = []
    for node_type, color in type_color_map.items():
        if node_type in embeddings_by_type:
            offset = node_offsets[node_type]
            count = len(embeddings_by_type[node_type])
            mask = np.zeros(num_embeddings, dtype=bool)
            mask[offset:offset+count] = True
            
            ax1.scatter(embeddings_umap[mask, 0], embeddings_umap[mask, 1], s=10, color=color, label=f"Type: {node_type}", alpha=0.7)
            handles.append(plt.Line2D([0], [0], marker='o', color='w', label=f"Type: {node_type}",
                                      markerfacecolor=color, markersize=10))

    ax1.set_xlabel("UMAP Dimension 1")
    ax1.set_ylabel("UMAP Dimension 2")

    # Plot PCA PC1 vs PC2 (top-right)
    ax2 = axes[0, 1]
    ax2.set_title("PCA Projection", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], c=point_colors, s=10, alpha=0.7)
    ax2.set_xlabel(f"PC 1 ({explained_variance[0]:.2%})")
    ax2.set_ylabel(f"PC 2 ({explained_variance[1]:.2%})")

    # Plot PCA PC2 vs PC3 (bottom-left)
    ax3 = axes[1, 0]
    ax3.set_title("PCA Projection", fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.scatter(embeddings_pca[:, 1], embeddings_pca[:, 2], c=point_colors, s=10, alpha=0.7)
    ax3.set_xlabel(f"PC 2 ({explained_variance[1]:.2%})")
    ax3.set_ylabel(f"PC 3 ({explained_variance[2]:.2%})")

    # Plot PCA PC1 vs PC3 (bottom-right)
    ax4 = axes[1, 1]
    ax4.set_title("PCA Projection", fontsize=12)
    ax4.grid(True, linestyle='--', alpha=0.6)
    ax4.scatter(embeddings_pca[:, 0], embeddings_pca[:, 2], c=point_colors, s=10, alpha=0.7)
    ax4.set_xlabel(f"PC 1 ({explained_variance[0]:.2%})")
    ax4.set_ylabel(f"PC 3 ({explained_variance[2]:.2%})")

    # --- 5. Final Touches & Save/Show ---
    total_explained_variance_str = f"Total Explained Variance (3 components): {explained_variance.sum():.2%}"
    fig.suptitle(f"Embedding Visualization for {Path(args.embeddings).name} ({embeddings.shape[1]}-D)\n{total_explained_variance_str}", fontsize=16)
    
    # Add shared legend for node types
    fig.legend(handles=handles, title="Node Types", bbox_to_anchor=(1.0, 0.9), loc='upper left')

    # Adjust layout to make space for suptitle and legend
    fig.tight_layout(rect=[0, 0, 0.9, 0.95])

    # Ensure the output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving figure to '{args.output}'")
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print("Figure saved.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate a 2D UMAP visualization of author embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--embeddings', type=Path, required=True,
                        help="Path to the saved author embeddings file (*.pt).")
    parser.add_argument('--labels', type=Path,
                        help="Path to author labels file (e.g., author_label.txt). Used for filtering, not coloring.")
    parser.add_argument('--names', type=Path,
                        help="Optional path to author names file (author_id<tab>name).")
    parser.add_argument('--data-dir', type=Path, default='data',
                        help="Directory containing the co-authorship file (AA.txt).")
    parser.add_argument('--min-coauthors', type=int, default=0,
                        help="Minimum number of co-authorships required for an author to be visualized. Set to 0 to disable.")
    parser.add_argument('--annotate', type=lambda s: [int(item) for item in s.split(',')],
                        help="Optional comma-separated list of 1-based author IDs to annotate in the plot.")
    parser.add_argument('--output', type=Path, default='plots/umap_visualization.png',
                        help="Path to save the output plot.")
    parser.add_argument('--plot-types', type=str, nargs='+', default=['A', 'C', 'T'],
                        help="List of node types to include in the plot (e.g., 'A' 'C').")
    
    main(parser.parse_args()) 