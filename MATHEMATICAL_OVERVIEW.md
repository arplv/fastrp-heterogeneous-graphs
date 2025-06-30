# Mathematical Overview: FastRP for Heterogeneous Graphs

This document provides a comprehensive mathematical explanation of the FastRP (Fast Random Projection) method for learning embeddings on heterogeneous graphs, specifically implemented for bibliographic networks with authors, conferences, and terms.

## Table of Contents

1. [Problem Formulation](#problem-formulation)
2. [Graph Representation](#graph-representation)
3. [Meta-Path Construction](#meta-path-construction)
4. [Random Projection Matrix](#random-projection-matrix)
5. [Degree Normalization](#degree-normalization)
6. [Feature Generation via Matrix Powers](#feature-generation-via-matrix-powers)
7. [Learnable Feature Combination](#learnable-feature-combination)
8. [Link Prediction via Distance-Based Scoring](#link-prediction-via-distance-based-scoring)
9. [Loss Function](#loss-function)
10. [Optimization](#optimization)
11. [Training Algorithm](#training-algorithm)

---

## Problem Formulation

**Objective**: Learn $d$-dimensional embeddings for nodes in a heterogeneous graph to predict missing links.

**Input**: 
- Heterogeneous graph $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathcal{T})$ where:
  - $\mathcal{V}$ is the set of nodes
  - $\mathcal{E}$ is the set of edges  
  - $\mathcal{T}$ is the set of node types (Authors, Conferences, Terms)

**Output**: 
- Node embeddings $\mathbf{Z} \in \mathbb{R}^{|\mathcal{V}| \times d}$ that preserve semantic relationships

---

## Graph Representation

The heterogeneous graph is represented using multiple relation matrices:

- $\mathbf{A}_{AA} \in \{0,1\}^{n_A \times n_A}$: Author-Author co-authorship
- $\mathbf{A}_{AC} \in \{0,1\}^{n_A \times n_C}$: Author-Conference publication  
- $\mathbf{A}_{AT} \in \{0,1\}^{n_A \times n_T}$: Author-Term keyword usage
- $\mathbf{A}_{CC}, \mathbf{A}_{CT}, \mathbf{A}_{TT}$: Additional relation types

Where $n_A$, $n_C$, $n_T$ are the number of authors, conferences, and terms respectively.

---

## Meta-Path Construction

**Meta-paths** capture semantic relationships by chaining relation types. For a meta-path $\mathcal{P} = T_1 \xrightarrow{R_1} T_2 \xrightarrow{R_2} \cdots \xrightarrow{R_k} T_{k+1}$:

The meta-path adjacency matrix is computed as:
$$\mathbf{M}_{\mathcal{P}} = \mathbf{A}_{T_1T_2} \cdot \mathbf{A}_{T_2T_3} \cdots \mathbf{A}_{T_kT_{k+1}}$$

**Example meta-paths**:
- $ACA$: $\mathbf{M}_{ACA} = \mathbf{A}_{AC} \cdot \mathbf{A}_{CA}$ (authors connected via conferences)
- $ATA$: $\mathbf{M}_{ATA} = \mathbf{A}_{AT} \cdot \mathbf{A}_{TA}$ (authors connected via terms)  
- $AAA$: $\mathbf{M}_{AAA} = \mathbf{A}_{AA} \cdot \mathbf{A}_{AA}$ (authors connected via 2-hop co-authorship)

---

## Random Projection Matrix

A sparse random projection matrix $\mathbf{R} \in \mathbb{R}^{n \times d}$ is constructed where each column has exactly $s$ non-zero entries:

$$\mathbf{R}_{ij} = \begin{cases}
\frac{1}{\sqrt{s}} & \text{with probability } \frac{s}{n} \\
\frac{-1}{\sqrt{s}} & \text{with probability } \frac{s}{n} \\
0 & \text{otherwise}
\end{cases}$$

**Degree weighting** is applied to account for node importance:
$$\mathbf{R}' = \mathbf{D}^{\alpha} \mathbf{R}$$

Where:
- $\mathbf{D} = \text{diag}(d_1, d_2, \ldots, d_n)$ is the degree matrix
- $\alpha \in \mathbb{R}$ is the degree weighting exponent (default: $\alpha = -0.5$)
- $d_i = \sum_{j} \mathbf{A}_{ij}$ is the degree of node $i$

---

## Degree Normalization

Each meta-path matrix $\mathbf{M}_{\mathcal{P}}$ is normalized to handle degree heterogeneity:

$$\tilde{\mathbf{M}}_{\mathcal{P}} = \mathbf{D}_{\mathcal{P}}^{\beta} \mathbf{M}_{\mathcal{P}}$$

Where:
- $\mathbf{D}_{\mathcal{P}} = \text{diag}(\mathbf{M}_{\mathcal{P}} \mathbf{1})$ contains row sums of $\mathbf{M}_{\mathcal{P}}$
- $\beta \in \mathbb{R}$ is the normalization exponent (default: $\beta = -1.0$)

This normalization prevents high-degree nodes from dominating the embeddings.

---

## Feature Generation via Matrix Powers

For each meta-path $\mathcal{P}$ and power $q \in \{1, 2, \ldots, Q\}$, features are generated as:

$$\mathbf{U}_{\mathcal{P}}^{(q)} = \text{normalize}\left(\tilde{\mathbf{M}}_{\mathcal{P}}^q \mathbf{R}'\right)$$

**Efficient computation**: Instead of computing $\tilde{\mathbf{M}}_{\mathcal{P}}^q$ explicitly, we use:
$$\mathbf{U}_{\mathcal{P}}^{(q)} = \tilde{\mathbf{M}}_{\mathcal{P}} \mathbf{U}_{\mathcal{P}}^{(q-1)}$$

with $\mathbf{U}_{\mathcal{P}}^{(0)} = \mathbf{R}'$.

**L2 Normalization** is applied row-wise:
$$\mathbf{U}_{\mathcal{P}}^{(q)} \leftarrow \frac{\mathbf{U}_{\mathcal{P}}^{(q)}}{\|\mathbf{U}_{\mathcal{P}}^{(q)}\|_2}$$

---

## Learnable Feature Combination

All generated features are combined using learnable weights:

$$\mathbf{Z} = \sum_{p=1}^{P} \sum_{q=1}^{Q} w_{p,q} \mathbf{U}_{\mathcal{P}_p}^{(q)}$$

Where the weights are softmax-normalized:
$$w_{p,q} = \frac{\exp(\theta_{p,q})}{\sum_{p'=1}^{P} \sum_{q'=1}^{Q} \exp(\theta_{p',q'})}$$

The parameters $\boldsymbol{\theta} \in \mathbb{R}^{P \times Q}$ are learned during training.

**Compact notation**:
$$\mathbf{Z} = \sum_{f=1}^{F} w_f \mathbf{U}_f$$

where $F = P \times Q$ and $\mathbf{U}_f$ represents the $f$-th feature matrix.

---

## Link Prediction via Distance-Based Scoring

**Embedding lookup**: For a node pair $(i,j)$, extract embeddings $\mathbf{z}_i, \mathbf{z}_j \in \mathbb{R}^d$.

**Distance computation**: 
$$d_{ij}^2 = \|\mathbf{z}_i - \mathbf{z}_j\|_2^2 = \sum_{k=1}^{d} (z_{i,k} - z_{j,k})^2$$

**Link probability**: 
$$\hat{y}_{ij} = \sigma(\gamma - \lambda \cdot d_{ij}^2)$$

Where:
- $\sigma(x) = \frac{1}{1 + e^{-x}}$ is the sigmoid function
- $\gamma$ is a learnable intercept parameter  
- $\lambda$ is a learnable slope parameter (constrained to be non-negative)

**Intuition**: Closer embeddings (smaller $d_{ij}^2$) result in higher link probabilities.

---

## Loss Function

The training objective combines link prediction loss with regularization:

$$\mathcal{L} = \mathcal{L}_{\text{BCE}} + \lambda_{\text{entropy}} \mathcal{L}_{\text{entropy}}$$

### Binary Cross-Entropy Loss

$$\mathcal{L}_{\text{BCE}} = -\frac{1}{|\mathcal{B}|} \sum_{(i,j) \in \mathcal{B}} \left[ y_{ij} \log(\hat{y}_{ij}) + (1-y_{ij}) \log(1-\hat{y}_{ij}) \right]$$

**Class imbalance handling**: Positive samples are weighted by the negative sampling ratio $r$:

$$\mathcal{L}_{\text{BCE}} = -\frac{1}{|\mathcal{B}|} \sum_{(i,j) \in \mathcal{B}} \left[ r \cdot y_{ij} \log(\hat{y}_{ij}) + (1-y_{ij}) \log(1-\hat{y}_{ij}) \right]$$

### Entropy Regularization

To encourage diverse feature usage:

$$\mathcal{L}_{\text{entropy}} = -\sum_{f=1}^{F} w_f \log(w_f + \epsilon)$$

Where $\epsilon = 10^{-7}$ prevents numerical instability.

**Effect**: Prevents the model from relying on a single meta-path/power combination.

---

## Optimization

**Optimizer**: Adam with learning rate $\eta = 0.01$

**Learnable parameters**:
- Feature weights: $\boldsymbol{\theta} \in \mathbb{R}^{P \times Q}$
- Intercept: $\gamma \in \mathbb{R}$  
- Slope: $\lambda \in \mathbb{R}$ (with ReLU constraint: $\lambda \geq 0$)

**Learning rate scheduling**: ReduceLROnPlateau with:
- Factor: $0.5$
- Patience: $10$ epochs
- Metric: Validation AUC (maximize)

**Early stopping**: Training stops if validation AUC doesn't improve for $20$ epochs.

---

## Training Algorithm

```
Algorithm: FastRP Heterogeneous Graph Embedding

Input: Graph G, meta-paths P, embedding dimension d, 
       num_powers Q, hyperparameters α, β, s

1. Construct relation matrices {A_XY} from G
2. Create degree-weighted random projection R'
3. For each meta-path P_p:
   a. Compute normalized meta-path matrix M̃_p
   b. For q = 1 to Q:
      - Compute U_p^(q) = M̃_p @ U_p^(q-1) (with U_p^(0) = R')
      - Apply L2 normalization
4. Initialize parameters θ, γ, λ
5. For each epoch:
   a. Sample positive edges and negative edges (ratio r:1)
   b. For each batch:
      - Compute embeddings Z = Σ w_f U_f
      - Extract pair embeddings z_i, z_j
      - Compute distances d_ij^2 = ||z_i - z_j||^2
      - Predict probabilities ŷ_ij = σ(γ - λ d_ij^2)  
      - Compute loss L = L_BCE + λ_entropy L_entropy
      - Update parameters via backpropagation
6. Return best model based on validation AUC
```

---

## Key Mathematical Properties

1. **Scalability**: $O(|\mathcal{E}| \cdot d \cdot Q)$ complexity per epoch
2. **Expressiveness**: Captures relationships up to $Q$-hop via matrix powers
3. **Regularization**: Entropy term prevents overfitting to single meta-paths
4. **Degree robustness**: Normalization handles heterogeneous node degrees
5. **Distance-based**: Euclidean distance naturally captures similarity

---

## Node Filtering for Noise Reduction

**Optional preprocessing step** to improve embedding quality by removing low-degree nodes:

**Degree calculation**: For each author node $i$:
$$d_i^{\text{total}} = \sum_{j \neq i} \mathbf{A}_{AA}[i,j] + \sum_{c} \mathbf{A}_{AC}[i,c] + \sum_{t} \mathbf{A}_{AT}[i,t]$$

**Filtering criterion**: 
$$\mathcal{V}_{\text{filtered}} = \{v \in \mathcal{V} : d_v^{\text{total}} \geq \tau\}$$

Where $\tau$ is the minimum degree threshold.

**Matrix re-indexing**: After filtering, all relation matrices are re-indexed to the filtered node space, ensuring computational consistency.

**Benefits**:
- Removes noisy, sparsely connected nodes
- Improves clustering quality in visualizations  
- Reduces computational overhead
- Enhances model robustness

---

## Hyperparameters Summary

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Embedding dimension | $d$ | 256 | Size of node embeddings |
| Meta-paths | $\mathcal{P}$ | [AAA, ACA, ATA] | Semantic relationship types |
| Matrix powers | $Q$ | 2 | Maximum meta-path power |
| Random projection sparsity | $s$ | 3 | Non-zeros per column in R |
| Degree weighting | $\alpha$ | -0.5 | Exponent for R' = D^α R |
| Degree normalization | $\beta$ | -1.0 | Exponent for M̃ = D^β M |
| Negative sampling ratio | $r$ | 3 | Negatives per positive |
| Entropy regularization | $\lambda_{\text{entropy}}$ | 0.0 | Weight for diversity loss |
| Learning rate | $\eta$ | 0.01 | Adam optimizer step size |
| Batch size | - | 4096 | Mini-batch size |
| **Minimum degree threshold** | $\tau$ | 0 | **Filter nodes with degree < τ** |

This mathematical framework enables the model to learn rich, multi-scale representations of heterogeneous graphs while maintaining computational efficiency and interpretability. 