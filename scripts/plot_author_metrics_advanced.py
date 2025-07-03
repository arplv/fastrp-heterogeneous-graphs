#!/usr/bin/env python3
"""Advanced PCA visualisation with standardized and graph-centrality metrics.

Generates two figures:
1. Grid of (PC1-PC2, PC2-PC3, PC1-PC3) coloured by individual z-scored metrics.
2. Single PC1-PC2 plot coloured by a composite *Influence Score* (average z-scores of
   publication_count, collaboration_count, unique_conferences, PageRank, k-core).

Run:
    python scripts/plot_author_metrics_advanced.py \
        --embeddings author_embeddings.pt \
        --data-dir data \
        --output-dir plots
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
import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))
from data_loader import load_data  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_row_boolean_count(mat: sp.csr_matrix) -> np.ndarray:
    return mat.indptr[1:] - mat.indptr[:-1]


def zscore(arr: np.ndarray) -> np.ndarray:
    mu, sigma = arr.mean(), arr.std(ddof=0)
    sigma = sigma if sigma > 0 else 1.0
    return (arr - mu) / sigma


def compute_base_metrics(relations: Dict[str, Dict[str, sp.csr_matrix]], keep_idx: np.ndarray):
    n_keep = len(keep_idx)

    def rows(mat: sp.csr_matrix):
        return mat[keep_idx, :]

    def sub(mat: sp.csr_matrix):
        return mat[keep_idx, :][:, keep_idx]

    AA = sub(relations.get("A", {}).get("A", sp.csr_matrix((n_keep, n_keep))))
    AC = rows(relations.get("A", {}).get("C", sp.csr_matrix((n_keep, 0))))
    AT = rows(relations.get("A", {}).get("T", sp.csr_matrix((n_keep, 0))))

    AA_no_eye = AA.copy()
    AA_no_eye.setdiag(0)
    AA_no_eye.eliminate_zeros()

    base = {
        "publication_count": np.asarray(AC.sum(axis=1)).flatten(),
        "unique_conferences": safe_row_boolean_count(AC > 0),
        "collaboration_count": np.asarray(AA_no_eye.sum(axis=1)).flatten(),
        "unique_collaborators": safe_row_boolean_count(AA_no_eye > 0),
        "unique_terms": safe_row_boolean_count(AT > 0),
        "AA_matrix": AA_no_eye,  # pass for centrality
    }
    return base


def compute_centrality_metrics(AA_no_eye: sp.csr_matrix):
    print("  Building NetworkX graph for centralities (may take a moment)…")
    G = nx.from_scipy_sparse_array(AA_no_eye, parallel_edges=False)
    pagerank = np.array([pr for _, pr in sorted(nx.pagerank(G, alpha=0.85).items())])
    core = nx.core_number(G)
    kcore = np.array([core.get(i, 0) for i in range(AA_no_eye.shape[0])])
    return pagerank, kcore


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Advanced PCA visualisation of author metrics.")
    parser.add_argument("--embeddings", required=True, type=Path)
    parser.add_argument("--data-dir", default="data", type=Path)
    parser.add_argument("--output-dir", default="plots", type=Path)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- Load embeddings -----
    emb_obj = torch.load(args.embeddings, map_location="cpu")
    if isinstance(emb_obj, dict):
        E = emb_obj["embeddings"].numpy()
        filtered_to_original = emb_obj.get("filtered_to_original")
    else:
        E = emb_obj.numpy()
        filtered_to_original = None
    n_emb = E.shape[0]

    # ----- Load relations -----
    relations, *_ = load_data(args.data_dir)
    keep_idx = (np.array([orig for _, orig in sorted(filtered_to_original.items())])
                if filtered_to_original else np.arange(n_emb))

    # ----- Metrics -----
    print("Computing base metrics …")
    base = compute_base_metrics(relations, keep_idx)
    AA_no_eye = base.pop("AA_matrix")
    print("Computing graph centrality metrics …")
    pagerank, kcore = compute_centrality_metrics(AA_no_eye)

    metrics = {**base, "pagerank": pagerank, "kcore_index": kcore}

    # Composite influence (average z-scores of selected metrics)
    influence_components = [metrics[m] for m in ["publication_count", "collaboration_count", "unique_conferences", "pagerank", "kcore_index"]]
    influence_z = np.mean([zscore(comp) for comp in influence_components], axis=0)
    metrics["influence_score"] = influence_z  # already standardized

    # Standardize all metrics (z-score) for colour scaling
    metrics_std = {k: zscore(v) for k, v in metrics.items() if k != "influence_score"}

    # ----- PCA -----
    E_norm = normalize(E, axis=1)
    pca = PCA(n_components=3)
    E_3d = pca.fit_transform(E_norm)
    expl = pca.explained_variance_ratio_

    # -------- Grid figure --------
    pairs = [(0, 1), (1, 2), (0, 2)]
    titles_pairs = ["PC1 vs PC2", "PC2 vs PC3", "PC1 vs PC3"]

    grid_metrics = [m for m in metrics_std.keys() if m != "influence_score"]
    n_pairs, n_metrics_plot = len(pairs), len(grid_metrics)
    fig, axes = plt.subplots(n_pairs, n_metrics_plot, figsize=(4*n_metrics_plot, 4*n_pairs), squeeze=False)

    for r, (ix, iy) in enumerate(pairs):
        for c, m_name in enumerate(grid_metrics):
            ax = axes[r, c]
            vals = metrics_std[m_name]
            sc = ax.scatter(E_3d[:, ix], E_3d[:, iy], c=vals, cmap="coolwarm", s=8, alpha=0.65)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(m_name.replace("_", " ").title())
            if c == 0:
                ax.set_ylabel(titles_pairs[r])
            fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)

    fig.suptitle("Standardised author metrics on PCA planes", fontsize=15)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    grid_path = out_dir / "author_metrics_pca3_zscore.png"
    fig.savefig(grid_path, dpi=300)
    print(f"Saved grid figure → {grid_path}")

    # -------- Influence plot (PC1-PC2) --------
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sc = ax2.scatter(E_3d[:, 0], E_3d[:, 1], c=influence_z, cmap="plasma", s=10, alpha=0.7)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title("Composite Influence Score on PC1-PC2 plane\n" +
                  f"Explained variance PC1 {expl[0]:.2%}, PC2 {expl[1]:.2%}")
    cbar = fig2.colorbar(sc, ax=ax2)
    cbar.set_label("Influence z-score")
    infl_path = out_dir / "author_influence_pca12.png"
    fig2.tight_layout()
    fig2.savefig(infl_path, dpi=300)
    print(f"Saved influence figure → {infl_path}")


if __name__ == "__main__":
    main() 