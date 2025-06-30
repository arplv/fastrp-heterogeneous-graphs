# Node Filtering Usage Guide

The FastRP implementation now supports filtering out low-degree nodes to reduce noise and improve embedding quality.

## Basic Usage

### Default (No Filtering)
```bash
python src/main.py --epochs 30 --meta-paths AAA ACA ATA
```

### Filter Nodes with < 5 Connections
```bash
python src/main.py --epochs 30 --meta-paths AAA ACA ATA --min-degree 5
```

### Aggressive Filtering (< 10 Connections)
```bash
python src/main.py --epochs 30 --meta-paths AAA ACA ATA --min-degree 10
```

## What Gets Filtered

The filtering considers **total degree** across all relation types:
- **Author-Author** collaborations (co-authorship)
- **Author-Conference** publications  
- **Author-Term** keyword usage

**Example**: An author with 2 co-authors + 3 conferences + 1 term = 6 total connections
- `--min-degree 5`: ✅ Included  
- `--min-degree 10`: ❌ Filtered out

## Expected Results

From the server output, using `--min-degree 5`:
```
Original authors: 28,702
Filtered authors: 27,821 (removed 881 low-degree nodes)
```

This removes ~3% of the noisiest nodes while preserving the core research community structure.

## Visualization Impact

When using the plotting script, filtered embeddings automatically handle the index mapping:

```bash
# Train with filtering
python src/main.py --min-degree 5 --output filtered_embeddings.pt

# Visualize (automatically detects filtering)
python scripts/plot_umap.py \
    --embeddings filtered_embeddings.pt \
    --labels data/author_label.txt \
    --names data/author_dict.txt \
    --output plots/filtered_visualization.png
```

The visualization will show:
- Cleaner cluster separation
- Reduced noise in the periphery
- Better-defined research communities
- Higher explained variance in PCA

## Recommended Settings

| Use Case | Min Degree | Expected Effect |
|----------|------------|-----------------|
| **Exploratory** | 0 | No filtering, see all data |
| **Standard** | 3-5 | Remove peripheral noise |
| **Focus on Core** | 8-10 | Only established researchers |
| **Major Players** | 15+ | Highly connected authors only |

## Technical Details

- Filtering happens **before** model training
- All matrices are re-indexed to the filtered space
- Embedding files store mapping metadata for reconstruction
- Compatible with all existing meta-paths and parameters

This preprocessing step can significantly improve both training efficiency and result quality by focusing on the most informative parts of the heterogeneous graph.

## Saving Options

### Default Behavior (Best Model Only)
```bash
python src/main.py --epochs 30 --output best_embeddings.pt --save-model-path best_model.pth
```
Saves only the model state with highest validation AUC.

### Save Both Best and Final States
```bash
python src/main.py --epochs 30 \
    --output best_embeddings.pt --save-model-path best_model.pth \
    --save-final --final-embeddings-path final_embeddings.pt --final-model-path final_model.pth
```

This creates:
- `best_embeddings.pt` & `best_model.pth`: Highest validation AUC
- `final_embeddings.pt` & `final_model.pth`: Final training epoch

### Use Cases for Final Model
- **Continued training**: Resume from exact final state
- **Analysis**: Compare best vs. final performance
- **Debugging**: Examine overfitting behavior
- **Research**: Study training dynamics 