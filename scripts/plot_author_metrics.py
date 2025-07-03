#!/usr/bin/env python3
"""Plot author embeddings colored by various graph-based metrics.

Usage example:
    python scripts/plot_author_metrics.py \
        --embeddings author_embeddings.pt \
        --data-dir data \
        --output plots/author_metrics.png

The script loads pre-computed embeddings (produced by src/main.py), computes
several intuitive metrics from the raw relation matrices, performs a 2-D PCA
projection of the embeddings, then generates a multi-panel figure where each
subplot is coloured by a different metric (log-scaled when appropriate).
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import scipy.sparse as sp

# Make src importable when the script is executed from project root
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))
from data_loader import load_data  # noqa: E402


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def safe_row_boolean_count(mat: sp.csr_matrix) -> np.ndarray:
    """Return (# non-zero columns in each row) for a CSR matrix."""
    # ``mat.indptr`` gives cumulative counts of non-zeros per row
    row_nnz = mat.indptr[1:] - mat.indptr[:-1]
    return row_nnz


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualise embeddings coloured by author-level metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--embeddings", type=Path, required=True,
                        help="Path to the .pt embeddings file (output of training).")
    parser.add_argument("--data-dir", type=Path, default="data",
                        help="Directory containing raw relation files (AA.txt, AC.txt, AT.txt, etc.).")
    parser.add_argument("--output", type=Path, default="plots/author_metrics.png",
                        help="Destination path for the generated figure (PNG).")
    args = parser.parse_args()

    # ---------------------------- Load embeddings ---------------------------
    print("Loading embeddings …")
    try:
        emb_obj = torch.load(args.embeddings, map_location="cpu", weights_only=False)
    except TypeError:  # older torch
        emb_obj = torch.load(args.embeddings, map_location="cpu")

    if isinstance(emb_obj, dict):
        E = emb_obj["embeddings"].numpy()
        filtered_to_original = emb_obj.get("filtered_to_original")
        print(f"  Loaded embeddings: shape={E.shape}. Filtered={filtered_to_original is not None}")
    else:
        E = emb_obj.numpy()
        filtered_to_original = None
        print(f"  Loaded embeddings tensor directly: shape={E.shape}")

    n_emb_authors = E.shape[0]

    # ---------------------------- Load relations ---------------------------
    relations, n_authors, *_ = load_data(args.data_dir)
    if n_authors < n_emb_authors:
        raise ValueError("Number of embeddings exceeds number of authors in data directory — mismatch?")

    # If embeddings were filtered, restrict matrices to filtered index space
    if filtered_to_original is not None:
        keep_idx = np.array([orig for _, orig in sorted(filtered_to_original.items())])
    else:
        keep_idx = np.arange(n_emb_authors)

    print("Computing author-level metrics …")

    # --- Prepare relation shortcuts (with possible row/col selection) ----
    def rows(mat: sp.csr_matrix):
        return mat[keep_idx, :]

    def submatrix(mat: sp.csr_matrix):
        # For AA we need rows AND cols restricted
        return mat[keep_idx, :][:, keep_idx]

    AA = submatrix(relations["A"]["A"]) if "A" in relations and "A" in relations["A"] else sp.csr_matrix((len(keep_idx), len(keep_idx)))
    AC = rows(relations["A"].get("C", sp.csr_matrix((len(keep_idx), 0))))
    AT = rows(relations["A"].get("T", sp.csr_matrix((len(keep_idx), 0))))

    # Remove self-loops from AA before computing collaboration counts
    AA_no_eye = AA.copy()
    AA_no_eye.setdiag(0)
    AA_no_eye.eliminate_zeros()

    # ---- Metric calculations -------------------------------------------
    metrics = {
        "unique_conferences": safe_row_boolean_count(AC > 0),
        "unique_collaborators": safe_row_boolean_count(AA_no_eye > 0),
        "collaboration_count": np.array(AA_no_eye.sum(axis=1)).flatten(),
        "publication_count": np.array(AC.sum(axis=1)).flatten(),
        "unique_terms": safe_row_boolean_count(AT > 0),
    }

    # ------------------------------ PCA -----------------------------------
    print("Running PCA → 2D …")
    E_norm = normalize(E, axis=1)
    pca = PCA(n_components=2)
    E_2d = pca.fit_transform(E_norm)
    explained = pca.explained_variance_ratio_.sum()
    print(f"  Explained variance (2 components): {explained:.2%}")

    # ---------------------------- Plotting --------------------------------
    n_metrics = len(metrics)
    n_cols = 3
    n_rows = int(np.ceil(n_metrics / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), squeeze=False)
    axes = axes.flatten()

    for ax, (name, values) in zip(axes, metrics.items()):
        # Log-scale for highly skewed counts
        vals_log = np.log1p(values)
        sc = ax.scatter(E_2d[:, 0], E_2d[:, 1], c=vals_log, cmap="viridis", s=10, alpha=0.6)
        ax.set_title(name.replace("_", " ").title())
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(sc, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
        cbar.set_label(f"log(1+{name})")

    # Hide unused subplots
    for ax in axes[n_metrics:]:
        ax.axis("off")

    fig.suptitle(f"Author Metrics over PCA Projection\nEmbeddings: {args.embeddings.name}", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=300)
    print(f"Plot saved to {args.output.resolve()}")


if __name__ == "__main__":
    main() 