# FastRP for Heterogeneous Graphs: A PyTorch Implementation

This repository contains a PyTorch-based implementation of a FastRP-inspired embedding method for heterogeneous graphs. The project is designed to generate high-quality node embeddings from complex, multi-relational data, as demonstrated on a bibliographic dataset.

The core of this implementation lies in its ability to learn expressive embeddings by combining signals from multiple, user-defined **meta-paths**. The model iteratively generates features from different powers of meta-path adjacency matrices (`M`, `M^2`, ..., `M^q`) and learns to combine them into a final embedding, which is then used for link prediction.

![AUC Chart](https://i.imgur.com/rS4gL6e.png)
*<p align="center">AUC score stabilizing around 0.94, indicating strong predictive performance.</p>*

## Key Features

- **Heterogeneous Graph Support:** Define and use any number of meta-paths (e.g., `Author-Conference-Author`) to capture rich semantic relationships.
- **Efficient High-Order Projections:** Implements the `M @ (M @ R)` iterative strategy, allowing the model to leverage higher-order relationships (`M^2`, `M^3`, etc.) without the massive memory overhead of pre-computing the powered matrices.
- **Learnable Feature Combination:** Instead of concatenating or averaging, the model learns the optimal weight for each feature (`ACA^1`, `ACA^2`, `ATA^1`, etc.) via a softmax layer.
- **Entropy Regularization:** The loss function includes an entropy regularization term that encourages the model to use a diverse combination of features, preventing "feature collapse" and leading to richer embeddings.
- **Configurable & Reproducible:** Easily configure all key hyperparameters, including embedding dimension, meta-paths, matrix powers, and regularization strength, via command-line arguments.

## How It Works

The training process follows these main stages:

1.  **Meta-Path Adjacency Calculation:** The script first computes the base adjacency matrix for each user-defined meta-path (e.g., `ACA = A@C@A`).
2.  **Model Initialization:** The `FastRPModel` is set up. This includes creating a degree-weighted random projection matrix (`R' = D^alpha @ R`) and pre-processing the meta-path matrices with degree normalization (`M_norm = D^-beta @ M`).
3.  **On-the-Fly Feature Generation:** During the forward pass, for each meta-path `M`, the model iteratively computes a series of feature matrices:
    - `U_1 = normalize(M_norm @ R')`
    - `U_2 = normalize(M_norm @ U_1)`
    - ... up to `num_powers`.
4.  **Weighted Combination:** All generated feature matrices are combined using a learned, softmax-normalized set of weights to produce the final embedding `Z`.
5.  **Link Prediction & Loss:** The model predicts link probabilities between pairs of nodes based on the squared distance between their embeddings. The loss function combines standard Binary Cross-Entropy with entropy regularization on the feature weights to ensure a diverse and robust solution.

## Usage

### 1. Installation

First, clone the repository and set up the Python environment.

```bash
# Clone the repo
git clone <your-repo-url>
cd <repo-name>

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Running the Model

You can run the model with its default parameters. The script will automatically use your Apple Silicon (MPS) GPU if available.

```bash
python src/main.py --epochs 30
```

The script will generate the final embeddings in a file named `author_embeddings.npy`.

### 3. Configuration

You can customize the training process with various command-line arguments.

```bash
python src/main.py \
    --meta-paths AAA ATA ACA \
    --dim 128 \
    --num-powers 3 \
    --alpha -0.5 \
    --beta -0.5 \
    --lambda-entropy 0.01 \
    --lr 0.01 \
    --epochs 50
```

#### Key Arguments:

-   `--meta-paths`: A list of the base meta-paths to use.
-   `--dim`: The dimensionality of the final embedding.
-   `--num-powers`: The number of matrix powers to use for feature generation (e.g., 3 means `M^1, M^2, M^3`).
-   `--alpha`: The degree-weighting exponent for the random projection matrix.
-   `--beta`: The degree-normalization exponent for the meta-path matrices.
-   `--lambda-entropy`: The strength of the entropy regularization on the feature weights.

---
*This project was developed with the assistance of an AI pair programmer.* 