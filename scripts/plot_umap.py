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
import torch
import umap
from sklearn.preprocessing import normalize

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
                    author_id = int(author_id)
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
                    id_to_name[int(author_id)] = name
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
    try:
        embeddings = torch.load(args.embeddings, map_location='cpu').numpy()
    except FileNotFoundError:
        print(f"Error: Embeddings file not found at '{args.embeddings}'")
        return
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return

    author_id_to_label_str, label_str_to_int = load_labels(args.labels)
    id_to_name = load_names(args.names) if args.names else {}

    # --- 2. Pre-processing ---
    print("Pre-processing data...")
    max_author_id = max(author_id_to_label_str.keys())
    if embeddings.shape[0] <= max_author_id:
        print(f"Error: The number of embeddings ({embeddings.shape[0]}) is less than the "
              f"largest author ID in the labels file ({max_author_id}). "
              "Embeddings and labels may be mismatched.")
        return

    # L2 normalize embeddings for cosine distance
    embeddings_norm = normalize(embeddings, norm='l2', axis=1)

    # --- 3. UMAP Projection ---
    print("Performing UMAP projection...")
    umap_params = {'n_neighbors': 15, 'min_dist': 0.1, 'metric': "cosine", 'random_state': 42}
    reducer = umap.UMAP(**umap_params)
    
    start_time = time.time()
    embeddings_2d = reducer.fit_transform(embeddings_norm)
    end_time = time.time()
    print(f"UMAP projection took {end_time - start_time:.2f} seconds.")

    # --- 4. Visualization ---
    print("Generating plot...")
    fig, ax = plt.subplots(figsize=(10, 8))

    # Determine colors for each point
    num_embeddings = embeddings_2d.shape[0]
    point_colors = np.full(num_embeddings, -1, dtype=int)
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
        other_label_id = next_label_id = len(label_str_to_int)
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
        legend_label_strs.append('Other')

    # Create colormap
    cmap = plt.get_cmap('tab20', len(label_str_to_int))

    # Plot unlabeled points
    ax.scatter(embeddings_2d[point_colors == -1, 0], embeddings_2d[point_colors == -1, 1], s=5, color='lightgray', alpha=0.6)

    # Plot labeled points
    for label_str, label_int in label_str_to_int.items():
        mask = point_colors == label_int
        if np.any(mask):
            ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], s=10, color=cmap(label_int), label=label_str, alpha=0.8)

    # Add legend
    ax.legend(title="Research Fields", bbox_to_anchor=(1.02, 1), loc='upper left')

    # Add annotations
    if args.annotate:
        print(f"Annotating authors: {args.annotate}")
        for author_id in args.annotate:
            if 0 <= author_id < num_embeddings:
                label = id_to_name.get(author_id, f"ID: {author_id}")
                ax.text(embeddings_2d[author_id, 0], embeddings_2d[author_id, 1], label,
                        fontsize=9, fontweight='bold', ha='center', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.7))
            else:
                print(f"Warning: Annotation ID {author_id} is out of bounds. Skipping.")

    # --- 5. Final Touches & Save/Show ---
    title = f"UMAP Projection of {embeddings.shape[1]}-D Embeddings\n(n_neighbors={umap_params['n_neighbors']}, min_dist={umap_params['min_dist']})"
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.grid(True, linestyle='--', alpha=0.6)
    fig.tight_layout()

    if args.output:
        print(f"Saving figure to '{args.output}'")
        plt.savefig(args.output, dpi=300, bbox_inches='tight')
    else:
        print("Displaying figure...")
        plt.show()


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
    parser.add_argument('--annotate', type=lambda s: [int(item) for item in s.split(',')],
                        help="Optional comma-separated list of author IDs to annotate in the plot.")
    parser.add_argument('--output', type=Path,
                        help="Optional path to save the output plot. If not provided, shows the plot.")
    
    main(parser.parse_args()) 