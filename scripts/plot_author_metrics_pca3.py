#!/usr/bin/env python3
"""Visualise author embeddings on the three PCA component pairs (PC1-PC2, PC2-PC3, PC1-PC3).

For each selected author-level metric, the script produces a grid of scatter
plots: each row corresponds to a (PCx, PCy) pair and each column to one metric.
Colour encodes the log-scaled metric value.

Usage:
    python scripts/plot_author_metrics_pca3.py \
        --embeddings author_embeddings.pt \
        --data-dir data \
        --output plots/author_metrics_pca3.png
"""
import argparse
from pathlib import Path
import sys
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import scipy.sparse as sp

# Allow `import data_loader` from src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))
from data_loader import load_data  # noqa: E402


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def safe_row_boolean_count(mat: sp.csr_matrix) -> np.ndarray:
    """Return (# non-zero columns in each row) for a CSR matrix."""
    return mat.indptr[1:] - mat.indptr[:-1]


def compute_metrics(relations: Dict[str, Dict[str, sp.csr_matrix]], keep_idx: np.ndarray):
    """Compute basic author-level metrics used for colouring."""
    def rows(mat: sp.csr_matrix):
        return mat[keep_idx, :]

    def submatrix(mat: sp.csr_matrix):
        return mat[keep_idx, :][:, keep_idx]

    # Fallback empty matrices when relation missing
    n_keep = len(keep_idx)
    AA = submatrix(relations.get("A", {}).get("A", sp.csr_matrix((n_keep, n_keep))))
    AC = rows(relations.get("A", {}).get("C", sp.csr_matrix((n_keep, 0))))
    AT = rows(relations.get("A", {}).get("T", sp.csr_matrix((n_keep, 0))))

    AA_no_eye = AA.copy()
    AA_no_eye.setdiag(0)
    AA_no_eye.eliminate_zeros()

    metrics = {
        "unique_conferences": safe_row_boolean_count(AC > 0),
        "unique_collaborators": safe_row_boolean_count(AA_no_eye > 0),
        "collaboration_count": np.asarray(AA_no_eye.sum(axis=1)).flatten(),
        "publication_count": np.asarray(AC.sum(axis=1)).flatten(),
        "unique_terms": safe_row_boolean_count(AT > 0),
    }
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot author metrics on three PCA component pairs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--embeddings", required=True, type=Path,
                        help="Path to embeddings .pt file (dict with 'embeddings').")
    parser.add_argument("--data-dir", default="data", type=Path,
                        help="Directory containing relation edge lists.")
    parser.add_argument("--output", default="plots/author_metrics_pca3.png", type=Path,
                        help="Destination PNG path.")
    args = parser.parse_args()

    # ---------------- Embeddings ----------------
    emb_obj = torch.load(args.embeddings, map_location="cpu")
    if isinstance(emb_obj, dict):
        E = emb_obj["embeddings"].numpy()
        filtered_to_original = emb_obj.get("filtered_to_original", None)
    else:  # legacy tensor only
        E = emb_obj.numpy()
        filtered_to_original = None
    n_emb = E.shape[0]

    # ---------------- Relations & metrics ----------------
    relations, n_authors, *_ = load_data(args.data_dir)
    if filtered_to_original is not None:
        keep_idx = np.array([orig for _, orig in sorted(filtered_to_original.items())])
    else:
        keep_idx = np.arange(n_emb)

    metrics = compute_metrics(relations, keep_idx)

    # ---------------- PCA -------------------
    E_norm = normalize(E, axis=1)
    pca = PCA(n_components=3)
    E_3d = pca.fit_transform(E_norm)
    expl = pca.explained_variance_ratio_

    pairs = [(0, 1), (1, 2), (0, 2)]
    pair_names = ["PC1 vs PC2", "PC2 vs PC3", "PC1 vs PC3"]

    n_pairs = len(pairs)
    n_metrics = len(metrics)

    fig, axes = plt.subplots(n_pairs, n_metrics, figsize=(4*n_metrics, 4*n_pairs), squeeze=False)

    for row, (idx_x, idx_y) in enumerate(pairs):
        for col, (metric_name, values) in enumerate(metrics.items()):
            ax = axes[row, col]
            ax.set_xticks([])
            ax.set_yticks([])
            # Use log1p scale to mitigate skew
            vals_log = np.log1p(values)
            sc = ax.scatter(E_3d[:, idx_x], E_3d[:, idx_y], c=vals_log,
                             cmap="viridis", s=8, alpha=0.6)
            if row == 0:
                ax.set_title(metric_name.replace("_", " ").title())
            if col == 0:
                ax.set_ylabel(pair_names[row])
            # Add small colorbar on first column only
            cbar = fig.colorbar(sc, ax=ax, orientation="vertical", fraction=0.04, pad=0.02)
            cbar.set_ticks([])

    supt = "Author metrics on PCA component pairs\n" \
        f"Explained variance: PC1 {expl[0]:.2%}, PC2 {expl[1]:.2%}, PC3 {expl[2]:.2%}"
    fig.suptitle(supt, fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=300)
    print(f"Saved figure to {args.output.resolve()}")


if __name__ == "__main__":
    main() 