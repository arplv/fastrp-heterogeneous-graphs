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
    """Loads research area names from the readme file (e.g., 'Database->1')."""
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

def plot_2d_chart(ax, title, embeddings_2d, point_colors, plot_mask, label_str_to_int, area_names, id_to_name, annotate_ids):
    """Helper to draw a 2D scatter plot on a given axes."""
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.grid(True, linestyle='--', alpha=0.6)

    cmap = plt.get_cmap('tab20', len(label_str_to_int))

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
            scatter = ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], s=10, color=cmap(label_int), label=legend_label, alpha=0.6)
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
    try:
        embedding_data = torch.load(args.embeddings, map_location='cpu', weights_only=False)
        
        # Handle both old format (direct tensor) and new format (dictionary)
        if isinstance(embedding_data, dict):
            embeddings = embedding_data['embeddings'].numpy()
            filtered_to_original = embedding_data.get('filtered_to_original', None)
            min_degree_threshold = embedding_data.get('min_degree_threshold', 0)
            n_original = embedding_data.get('n_original_authors', embeddings.shape[0])
            n_filtered = embedding_data.get('n_filtered_authors', embeddings.shape[0])
            
            if filtered_to_original is not None:
                print(f"  Loaded filtered embeddings: {n_filtered} authors (from {n_original} original)")
                print(f"  Minimum degree threshold was: {min_degree_threshold}")
            else:
                print(f"  Loaded unfiltered embeddings: {embeddings.shape[0]} authors")
        else:
            # Old format - direct tensor
            embeddings = embedding_data.numpy()
            filtered_to_original = None
            print(f"  Loaded embeddings: {embeddings.shape[0]} authors (legacy format)")
            
    except FileNotFoundError:
        print(f"Error: Embeddings file not found at '{args.embeddings}'")
        return
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return

    author_id_to_label_str, label_str_to_int = load_labels(args.labels)
    id_to_name = load_names(args.names) if args.names else {}
    area_names = load_area_names(args.data_dir / 'readme.txt')
    
    # If embeddings are filtered, remap author IDs to match the filtered indices
    if filtered_to_original is not None:
        print("  Remapping author IDs for filtered embeddings...")
        
        # Create new mappings using filtered indices
        filtered_author_id_to_label_str = {}
        filtered_id_to_name = {}
        
        for filtered_idx, original_idx in filtered_to_original.items():
            if original_idx in author_id_to_label_str:
                filtered_author_id_to_label_str[filtered_idx] = author_id_to_label_str[original_idx]
            if original_idx in id_to_name:
                filtered_id_to_name[filtered_idx] = id_to_name[original_idx]
        
        # Replace the original mappings
        author_id_to_label_str = filtered_author_id_to_label_str
        id_to_name = filtered_id_to_name
        
        print(f"    Mapped {len(author_id_to_label_str)} labels to filtered indices")

    # --- 2. Pre-processing & Filtering ---
    print("Pre-processing data...")
    n_authors = embeddings.shape[0]

    # Create a mask to filter authors based on the minimum number of co-authorships
    if args.min_coauthors > 0:
        print(f"Will only visualize authors with at least {args.min_coauthors} co-authorships.")
        coauthorship_path = args.data_dir / 'AA.txt'
        adj_matrix = load_coauthorship(coauthorship_path, n_authors)
        if adj_matrix is not None:
            degrees = np.array(adj_matrix.sum(axis=1)).flatten()
            plot_mask = degrees >= args.min_coauthors
            print(f"Applying filter: {plot_mask.sum()} of {n_authors} authors will be shown.")
        else:
            # If AA.txt not found, don't filter
            plot_mask = np.ones(n_authors, dtype=bool)
    else:
        plot_mask = np.ones(n_authors, dtype=bool)

    max_author_id = max(author_id_to_label_str.keys())
    if embeddings.shape[0] <= max_author_id:
        print(f"Error: The number of embeddings ({embeddings.shape[0]}) is less than the "
              f"largest author ID in the labels file ({max_author_id}). "
              "Embeddings and labels may be mismatched.")
        return

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

    # Determine colors for each point
    num_embeddings = embeddings_umap.shape[0]
    point_colors = np.full(num_embeddings, -1, dtype=int)
    # Create a reverse mapping from integer to the string label for the legend
    int_to_label_str = {v: k for k, v in label_str_to_int.items()}

    for author_id, label_str in author_id_to_label_str.items():
        if author_id < num_embeddings:
            point_colors[author_id] = label_str_to_int[label_str]

    # Handle legend: show top 15 labels, group rest into "other"
    num_unique_labels = len(label_str_to_int)
    legend_label_strs = list(label_str_to_int.keys())
    if num_unique_labels > 15:
        print(f"Found {num_unique_labels} labels, will group minor ones into 'other' for legend.")
        label_counts = Counter(author_id_to_label_str.values())
        top_14_labels = {label for label, count in label_counts.most_common(14)}
        
        # Remap non-top labels to a new "other" category
        other_label_id = len(label_str_to_int)
        legend_label_strs = []
        for label_str, label_int in label_str_to_int.items():
            if label_str in top_14_labels:
                legend_label_strs.append(label_str)
            else:
                # Update point_colors for all authors with this minor label
                for author_id, author_label_str in author_id_to_label_str.items():
                    if author_label_str == label_str and author_id < len(point_colors):
                        point_colors[author_id] = other_label_id
        
        label_str_to_int['Other'] = other_label_id
        int_to_label_str[other_label_id] = 'Other' # Also update reverse mapping
        legend_label_strs.append('Other')

    # Plot UMAP (top-left)
    ax1 = axes[0, 0]
    umap_title = f"UMAP Projection\n(n_neighbors={umap_params['n_neighbors']}, min_dist={umap_params['min_dist']})"
    handles, labels = plot_2d_chart(ax1, umap_title, embeddings_umap, point_colors, plot_mask, label_str_to_int, area_names, id_to_name, args.annotate)
    ax1.set_xlabel("UMAP Dimension 1")
    ax1.set_ylabel("UMAP Dimension 2")

    # Plot PCA PC1 vs PC2 (top-right)
    ax2 = axes[0, 1]
    plot_2d_chart(ax2, "PCA Projection", embeddings_pca[:, [0, 1]], point_colors, plot_mask, label_str_to_int, area_names, id_to_name, args.annotate)
    ax2.set_xlabel(f"PC 1 ({explained_variance[0]:.2%})")
    ax2.set_ylabel(f"PC 2 ({explained_variance[1]:.2%})")

    # Plot PCA PC2 vs PC3 (bottom-left)
    ax3 = axes[1, 0]
    plot_2d_chart(ax3, "PCA Projection", embeddings_pca[:, [1, 2]], point_colors, plot_mask, label_str_to_int, area_names, id_to_name, args.annotate)
    ax3.set_xlabel(f"PC 2 ({explained_variance[1]:.2%})")
    ax3.set_ylabel(f"PC 3 ({explained_variance[2]:.2%})")

    # Plot PCA PC1 vs PC3 (bottom-right)
    ax4 = axes[1, 1]
    plot_2d_chart(ax4, "PCA Projection", embeddings_pca[:, [0, 2]], point_colors, plot_mask, label_str_to_int, area_names, id_to_name, args.annotate)
    ax4.set_xlabel(f"PC 1 ({explained_variance[0]:.2%})")
    ax4.set_ylabel(f"PC 3 ({explained_variance[2]:.2%})")

    # --- 5. Final Touches & Save/Show ---
    total_explained_variance_str = f"Total Explained Variance (3 components): {explained_variance.sum():.2%}"
    fig.suptitle(f"Embedding Visualization for {Path(args.embeddings).name} ({embeddings.shape[1]}-D)\n{total_explained_variance_str}", fontsize=16)
    
    # Add shared legend
    fig.legend(handles, labels, title="Research Fields", bbox_to_anchor=(1.0, 0.9), loc='upper left')

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
    parser.add_argument('--labels', type=Path, required=True,
                        help="Path to the tab-separated author labels file (author_id<tab>label).")
    parser.add_argument('--names', type=Path,
                        help="Optional path to author names file (author_id<tab>name).")
    parser.add_argument('--data-dir', type=Path, default='data',
                        help="Directory containing the co-authorship file (AA.txt).")
    parser.add_argument('--min-coauthors', type=int, default=10,
                        help="Minimum number of co-authorships required to be visualized.")
    parser.add_argument('--annotate', type=lambda s: [int(item) for item in s.split(',')],
                        help="Optional comma-separated list of 1-based author IDs to annotate in the plot.")
    parser.add_argument('--output', type=Path, default='plots/umap_visualization.png',
                        help="Path to save the output plot.")
    
    main(parser.parse_args()) 