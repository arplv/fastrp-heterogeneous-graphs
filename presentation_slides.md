# FastRP for Heterogeneous Graphs: Efficient Embedding via Random Projections

## A 30-Minute Technical Presentation

---

## Slide 1: Title & Overview

### FastRP for Heterogeneous Bibliographic Networks
**Efficient Graph Embeddings via Random Projections**

**Today's Agenda (30 minutes):**
- Problem Definition & Motivation (5 min)
- Theoretical Foundation: Johnson-Lindenstrauss Lemma (5 min)
- FastRP Algorithm & Implementation (10 min)
- Optimization & Learning (5 min)
- Benefits vs. Random Walk Methods (3 min)
- Results & Demo (2 min)

---

## Slide 2: Problem Definition

### The Challenge: Bibliographic Network Embedding

**Our Dataset:**
- 28,702 Authors
- 20 Conferences  
- 8,920 Terms
- Multiple relationship types: Author-Author, Author-Conference, Author-Term

**Goal:** Learn 256-dimensional embeddings that capture semantic relationships for link prediction

**Why is this hard?**
- Heterogeneous graph (multiple node types)
- Sparse connectivity
- Scale: Need to handle 28K+ nodes efficiently
- Semantic complexity: Different relationship types have different meanings

---

## Slide 3: Traditional Approach - Random Walks

### The Standard Method: Node2Vec/MetaPath2Vec

```
For each node:
  1. Generate random walks: A → C → A → T → A → ...
  2. Treat walks as "sentences"
  3. Apply Word2Vec (Skip-gram) to learn embeddings
```

**Problems:**
- **Computational Cost:** Need millions of walks
- **Memory:** Store all walk sequences  
- **Sampling Bias:** Random walks miss global structure
- **Hyperparameter Sensitivity:** Walk length, number of walks, etc.

**Example:** For 28K authors, typically need 10+ walks per node × 100 steps = 28M walk steps!

---

## Slide 4: Our Solution - FastRP Intuition

### Key Insight: Skip the Walks!

**Instead of random walks, use random projections:**

1. **Create sparse random matrix R** (cheap to compute)
2. **Apply graph operations directly:** M @ R, M² @ R, M³ @ R
3. **Combine features intelligently** with learned weights

**Why this works:** Johnson-Lindenstrauss Lemma guarantees that random projections preserve distances in high-dimensional spaces.

**Analogy:** Instead of exploring a city by taking random walks, we take aerial photos from different angles and combine them!

---

## Slide 5: Johnson-Lindenstrauss Lemma

### Theoretical Foundation

**Johnson-Lindenstrauss Lemma (1984):**
> For any set of n points in high-dimensional space, there exists a mapping to O(log n) dimensions that preserves all pairwise distances within (1±ε).

**Formally:** For points x₁, x₂, ..., xₙ ∈ ℝᵈ, there exists random matrix R ∈ ℝᵈˣᵏ where k = O(log n/ε²) such that:

```
(1-ε)||xᵢ - xⱼ||² ≤ ||R^T xᵢ - R^T xⱼ||² ≤ (1+ε)||xᵢ - xⱼ||²
```

**What this means for us:**
- Random projections preserve neighborhood structure
- We can work in lower dimensions without losing information
- Provides theoretical justification for our approach

---

## Slide 6: Random Projection Matrix Construction

### Step 1: Creating the Base Random Matrix R

**Matrix Dimensions:** R ∈ ℝⁿˣᵈ (28,702 × 256 in our case)

**Sparsity Pattern:** Each column has exactly s=3 non-zero entries:

```python
R[i,j] = {
    +1/√s  with probability s/n
    -1/√s  with probability s/n  
    0      otherwise
}
```

**Example for 6 authors, 4 dimensions, s=2:**
```
     dim1  dim2  dim3  dim4
A1  [ 0    +0.7   0    -0.7 ]
A2  [-0.7   0    +0.7   0   ]
A3  [ 0     0    -0.7  +0.7 ]
A4  [+0.7  -0.7   0     0   ]
A5  [ 0    +0.7  -0.7   0   ]
A6  [-0.7   0     0    +0.7 ]
```

**Key Properties:**
- Sparse: Only 2×4=8 non-zeros out of 24 entries
- Fast to multiply with
- Preserves distances (JL Lemma)

---

## Slide 7: Degree Weighting (Alpha Parameter)

### Step 2: Applying Degree Weighting

**Problem:** High-degree nodes (prolific authors) dominate embeddings

**Solution:** Degree weighting with parameter α = -0.5

```
R' = D^α @ R
```

**Example with α = -0.5:**
```
Author degrees: [100, 10, 5, 50, 20, 8]
D^(-0.5) = diag([0.10, 0.32, 0.45, 0.14, 0.22, 0.35])

Before weighting:
A1 (degree=100): [0, +0.7, 0, -0.7]
A2 (degree=10):  [-0.7, 0, +0.7, 0]

After weighting:
A1: [0, +0.07, 0, -0.07]  # Down-weighted
A2: [-0.22, 0, +0.22, 0]  # Up-weighted
```

**Effect:** Balances influence between prolific and emerging researchers

---

## Slide 8: Meta-Path Adjacency Matrices

### Step 3: Constructing Relationship Matrices

**Meta-Paths capture semantic relationships:**

**ACA (Author-Conference-Author):** Authors connected via conferences
```
M_ACA = A_AC @ A_CA
```

**ATA (Author-Term-Author):** Authors connected via keywords
```
M_ATA = A_AT @ A_TA  
```

**AAA (Author-Author-Author):** 2-hop collaborations
```
M_AAA = A_AA @ A_AA
```

**Matrix Dimensions Example:**
- A_AC: 28,702 × 20 (authors × conferences)
- A_CA: 20 × 28,702 (conferences × authors)  
- M_ACA: 28,702 × 28,702 (authors × authors)

**Interpretation:** M_ACA[i,j] = number of conferences authors i and j both published in

---

## Slide 9: Degree Normalization (Beta Parameter)

### Step 4: Normalizing Meta-Path Matrices

**Problem:** Some authors have many connections, others few

**Solution:** Row normalization with β = -1.0

```
M̃ = D^β @ M
```

**Example:**
```
Raw M_ACA row sums: [50, 5, 100, 20]  # conferences per author
D^(-1) = diag([1/50, 1/5, 1/100, 1/20])

Before normalization:
Author1: [10, 5, 30, 5]  # shared conferences
Author2: [2, 0, 1, 2]

After normalization:  
Author1: [0.2, 0.1, 0.6, 0.1]  # probabilities
Author2: [0.4, 0.0, 0.2, 0.4]
```

**Effect:** Converts counts to probability distributions

---

## Slide 10: Feature Generation - The Core Algorithm

### Step 5: Iterative Matrix Powers

**Key Innovation:** Instead of computing M², M³ explicitly, we iterate:

```python
U₀ = R'                    # Initial: degree-weighted random matrix
U₁ = normalize(M̃ @ U₀)     # 1-hop features  
U₂ = normalize(M̃ @ U₁)     # 2-hop features
U₃ = normalize(M̃ @ U₂)     # 3-hop features
```

**Concrete Example (ACA meta-path):**
```
U₀ = R' ∈ ℝ²⁸'⁷⁰² ˣ ²⁵⁶        # Random starting features
U₁ = M̃_ACA @ U₀              # Authors connected via 1 conference  
U₂ = M̃_ACA @ U₁              # Authors connected via 2 conferences
U₃ = M̃_ACA @ U₂              # Authors connected via 3 conferences
```

**Why this works:**
- U₁ captures direct conference relationships
- U₂ captures authors in same research community  
- U₃ captures broader field relationships

**Memory Efficiency:** Never store M̃² or M̃³ (would be 28K × 28K × 8 bytes = 6.4GB each!)

---

## Slide 11: Multi-Scale Feature Collection

### Step 6: Combining All Meta-Paths and Powers

**For each meta-path × power combination:**

```
Meta-paths: [AAA, ACA, ATA]
Powers: [1, 2, 3]
Total features: 3 × 3 = 9 feature matrices

Feature tensor F ∈ ℝ⁹ ˣ ²⁸'⁷⁰² ˣ ²⁵⁶
```

**Feature Interpretation:**
- F[0]: AAA¹ - Direct collaborations
- F[1]: AAA² - 2-hop collaborations  
- F[2]: AAA³ - 3-hop collaborations
- F[3]: ACA¹ - Shared conferences
- F[4]: ACA² - Conference communities
- F[5]: ACA³ - Broader research areas
- F[6]: ATA¹ - Shared keywords
- F[7]: ATA² - Topic neighborhoods  
- F[8]: ATA³ - Research domains

**Each feature matrix captures different aspects of author relationships!**

---

## Slide 12: Learnable Feature Combination

### Step 7: Intelligent Feature Weighting

**Problem:** Which features are most important? Let the model decide!

**Learnable weights with softmax normalization:**
```python
θ ∈ ℝ⁹  # Learnable parameters
w = softmax(θ)  # w[0] + w[1] + ... + w[8] = 1

# Final embedding:
Z = w[0]×F[0] + w[1]×F[1] + ... + w[8]×F[8]
```

**Example learned weights:**
```
AAA¹: 0.05  AAA²: 0.12  AAA³: 0.08   # Collaboration features
ACA¹: 0.25  ACA²: 0.30  ACA³: 0.15   # Conference features (dominant!)
ATA¹: 0.02  ATA²: 0.02  ATA³: 0.01   # Term features (minimal)
```

**Insight:** Model learns that conference relationships are most predictive for this dataset!

---

## Slide 13: Link Prediction via Distance

### Step 8: Distance-Based Scoring

**Embedding lookup:** For authors i and j, get embeddings z_i, z_j ∈ ℝ²⁵⁶

**Distance computation:**
```python
distance² = ||z_i - z_j||² = Σ(z_i[k] - z_j[k])²
```

**Link probability:**
```python
probability = sigmoid(γ - λ × distance²)
```

**Intuition:** 
- Small distance → High probability of collaboration
- Large distance → Low probability of collaboration

**Learnable parameters:**
- γ (intercept): Overall collaboration likelihood
- λ (slope): How much distance matters

---

## Slide 14: Optimization & Loss Function

### Training the Model

**Loss Function combines two terms:**

```python
Loss = BCE_Loss + λ_entropy × Entropy_Loss
```

**1. Binary Cross-Entropy (BCE):**
```python
BCE = -[y×log(p̂) + (1-y)×log(1-p̂)]
```
- Standard link prediction loss
- Weighted for class imbalance (3:1 negative:positive ratio)

**2. Entropy Regularization:**
```python
Entropy = -Σ w[i] × log(w[i])
```
- Prevents model from using only one feature type
- Encourages diverse feature usage

**Optimization:**
- Adam optimizer (lr=0.01)
- Early stopping on validation AUC
- ReduceLROnPlateau scheduling

---

## Slide 15: Benefits Over Random Walk Methods

### Computational Comparison

| Aspect | Random Walks (Node2Vec) | FastRP (Our Method) |
|--------|------------------------|-------------------|
| **Time Complexity** | O(rwl × n) | O(E × d × p) |
| **Memory** | O(rwl × n) | O(d × p) |
| **Preprocessing** | Generate walks | Sparse matrix ops |
| **Scalability** | Poor (quadratic) | Excellent (linear) |

**Where:**
- r = walks per node (typically 10-50)
- w = walk length (typically 80-100)  
- l = training epochs (typically 100)
- E = number of edges
- d = embedding dimension
- p = number of powers

**Real Numbers for our dataset:**
- Node2Vec: ~28K × 10 × 80 × 100 = 2.24B operations
- FastRP: ~500K edges × 256 × 3 = 384M operations
- **~6x speedup!**

---

## Slide 16: Quality Benefits

### Why FastRP Embeddings are Better

**1. Global Structure Preservation:**
- Random walks are local (limited by walk length)
- Matrix powers capture global connectivity patterns

**2. Multi-Scale Relationships:**
- Powers 1,2,3 capture immediate, community, and domain-level relationships
- Random walks only see fixed-length neighborhoods

**3. Deterministic Features:**
- No sampling variance
- Reproducible results
- Better for analysis and debugging

**4. Theoretical Guarantees:**
- Johnson-Lindenstrauss lemma provides distance preservation bounds
- Random walks have no such guarantees

**5. Heterogeneous Graph Support:**
- Natural handling of different relationship types
- Random walks struggle with heterogeneous graphs

---

## Slide 17: Concrete Example Walkthrough

### Example: Predicting if Alice and Bob will collaborate

**Step 1:** Alice (ID=1000), Bob (ID=1500)

**Step 2:** Extract their embeddings
```python
z_alice = Z[1000, :]  # 256-dimensional vector
z_bob = Z[1500, :]    # 256-dimensional vector
```

**Step 3:** Compute distance
```python
distance² = ||z_alice - z_bob||² = 2.34
```

**Step 4:** Predict probability
```python
γ = 0.8 (learned intercept)
λ = 0.5 (learned slope)
probability = sigmoid(0.8 - 0.5 × 2.34) = sigmoid(-0.37) = 0.41
```

**Result:** 41% chance Alice and Bob will collaborate

**How did we get z_alice?** From weighted combination of 9 different relationship features!

---

## Slide 18: Results & Performance

### Model Performance

**Link Prediction Results:**
- **Training AUC:** 0.94
- **Validation AUC:** 0.94
- **Training Time:** 15 minutes (vs. hours for random walks)

**Learned Feature Importance:**
```
Conference relationships (ACA): 70% weight
Collaboration patterns (AAA): 25% weight  
Topic similarity (ATA): 5% weight
```

**Node Filtering Impact:**
- Original: 28,702 authors
- Filtered (min-degree=5): 27,821 authors
- Removed 881 low-degree "noise" nodes
- Improved clustering quality

**Scalability:** Linear scaling with graph size

---

## Slide 19: Implementation Insights

### Key Engineering Decisions

**1. Sparse Matrix Operations:**
- Used SciPy sparse matrices throughout
- Never materialize full M² matrices (would exceed memory)
- Iterative computation: M @ (M @ R) instead of M² @ R

**2. Caching Strategy:**
- Disk cache for meta-path matrices
- Avoids recomputation across runs
- ~5x speedup for repeated experiments

**3. GPU Optimization:**
- Model training on GPU (MPS/CUDA)
- Matrix operations on CPU (sparse tensor limitations)
- Hybrid approach maximizes performance

**4. Memory Management:**
- Batch processing for large graphs
- Feature normalization prevents gradient explosion
- Efficient tensor operations with einsum

---

## Slide 20: Comparison Summary

### FastRP vs Random Walk Methods

**Advantages of FastRP:**
✅ **Speed:** 6x faster than Node2Vec
✅ **Memory:** Constant memory usage
✅ **Quality:** Better global structure capture
✅ **Deterministic:** No sampling variance
✅ **Scalable:** Linear time complexity
✅ **Interpretable:** Feature weights show what matters
✅ **Heterogeneous:** Natural multi-relation support

**Potential Disadvantages:**
⚠️ **Complexity:** More parameters to tune (α, β, powers)
⚠️ **Theory gap:** Less established than random walks
⚠️ **Implementation:** Requires careful sparse matrix handling

**Bottom Line:** FastRP provides superior speed-quality tradeoff for heterogeneous graphs

---

## Slide 21: Key Takeaways

### What You Should Remember

**1. Core Insight:**
Random projections + matrix powers = efficient graph embeddings

**2. Mathematical Foundation:**
Johnson-Lindenstrauss lemma justifies random projection approach

**3. Engineering Innovation:**
Iterative computation avoids memory explosion: M@(M@R) not M²@R

**4. Performance Gains:**
6x speedup with comparable/better quality vs. random walks

**5. Practical Impact:**
Real-world heterogeneous graphs benefit from multi-relation modeling

**The Big Picture:** FastRP makes graph embedding practical for large, complex, heterogeneous networks

---

## Slide 22: Questions & Discussion

### Thank You!

**Code & Implementation:**
- Available on GitHub: [Your Repository]
- Python implementation with PyTorch
- Supports CPU, CUDA, and Apple Silicon (MPS)

**Key Files:**
- `src/model.py`: FastRP implementation
- `src/main.py`: Training pipeline
- `scripts/plot_umap.py`: Visualization tools

**Try it yourself:**
```bash
python src/main.py --epochs 30 --meta-paths AAA ACA ATA
```

**Questions?**

---

## Appendix: Technical Details

### Matrix Dimension Reference

**Input Data:**
- Authors: 28,702
- Conferences: 20  
- Terms: 8,920

**Relation Matrices:**
- A_AA: 28,702 × 28,702 (sparse)
- A_AC: 28,702 × 20
- A_AT: 28,702 × 8,920
- A_CA: 20 × 28,702
- A_TA: 8,920 × 28,702

**Meta-Path Matrices:**
- M_AAA: 28,702 × 28,702
- M_ACA: 28,702 × 28,702  
- M_ATA: 28,702 × 28,702

**Feature Matrices:**
- Each U_i: 28,702 × 256
- Feature tensor F: 9 × 28,702 × 256
- Final embeddings Z: 28,702 × 256

**Memory Usage:**
- Sparse matrices: ~50MB
- Feature tensor: ~2.5GB
- Total: ~3GB (vs. 50GB+ for naive approach) 