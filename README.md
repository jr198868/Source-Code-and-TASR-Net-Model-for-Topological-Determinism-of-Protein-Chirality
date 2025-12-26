# Topological Determinism of Protein Chirality: Source Code

## Overview

This repository contains the complete computational pipeline for investigating whether **protein chirality is topologically determined**—i.e., whether topological features of protein structure fundamentally encode handedness information.
---

## Installation

### Requirements
- **Python** 3.7+
- **pip** package manager

### Quick Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Core Dependencies

| Category | Libraries |
|----------|-----------|
| **TDA** | `ripser`, `gudhi` |
| **ML/DL** | `tensorflow`, `scikit-learn`, `xgboost` |
| **Bioinformatics** | `biopython` |
| **Scientific Computing** | `numpy`, `pandas`, `scipy` |
| **Visualization** | `matplotlib`, `seaborn`, `plotly`, `networkx`, `umap-learn` |
| **Utilities** | `tqdm`, `joblib`, `pillow` |

---

#### Note: This repository contains the analysis code and processed results (~8GB). The raw PDB structural data (~98GB) is NOT included due to its large volume.

#### To independently reproduce the results or run the full pipeline, YOU MUST FIRST RUN THE DATA DOWNLOAD SCRIPT:

```python
python step1.0_Batch_Download_human_protein_PDB.py
```

## Project Structure

```
src/
├── main.py                                          # Pipeline orchestrator
├── 
├── # STEP 1: Download PDB files
├── step1.0_Batch_Download_human_protein_PDB.py      # Batch download PDB files from RCSB
|
├── # STEP 2: TDA Pretreatment & Feature Extraction
├── step2.0_TDA_Pretreatment.py                      # Extract CA atoms, compute persistent homology
├── 
├── # STEP 3: High-Dimensional Embedding Analysis (HDEA)
├── step3.0_HDEA_batch_transfer_TDA.py               # Aggregate persistence diagrams → feature vectors
├── step3.1_HDEA_dimensionality_reduction_cluster_analysis.py    # PCA + KMeans
├── step3.2_HDEA_extract_all_barcode_points.py       # Extract all persistence diagram points
├── step3.3_HDEA_universal_barcode.py                # Identify universal barcode signatures
├── 
├── # STEP 4: Topological Signature Modeling (F3DR)
├── step4.1_F3DR_high_dim_projection_regression.py   # Train autoencoder for topological signature
├── step4.2_F3DR_combine_results.py                  # Combine batch results
├── step4.3_F3DR_analyze_error_distribution_and_fold_class.py    # Analyze reconstruction error
├── 
├── # STEP 5: Chirality Prediction (TCEPC)
├── step5.1_TCEPC_generate_mirrored_coords.py        # Generate D-amino acid mirror proteins
├── step5.2_TCEPC_build_chirality_dataset.py         # Construct L/D paired dataset
├── step5.3_TCEPC_tda_feature_recovery.py            # Extract TDA features for L/D pairs
├── step5.4_TCEPC_chirality_model_benchmark.py       # Benchmark classifiers (LR, NN, XGBoost)
├── step5.5_TCEPC_spiral_vs_protein_topo.py          # Spiral vs protein topology comparison
├── step5.6_TCEPC_chirality_visualization.py         # Chirality classification visualization
├── 
├── supplementary_information.py                     # Generate supplementary figures
└── README.md
```
---

## Key Concepts & Theoretical Framework

### Persistent Homology (TDA)

Persistent homology tracks how topological features (connected components, loops, cavities) appear and disappear as a distance parameter grows:

- **H₀** (dimension 0): Connected components – always non-trivial
- **H₁** (dimension 1): Loops/cycles – emerge when distance ≥ some birth value
- **H₂** (dimension 2): Cavities/voids – emerge from bubble-like structures

**Birth-Death Diagrams:**
- Point (b, d) means a feature appears at distance b and persists until d
- **Persistence** = d - b (how "real" the feature is; noise has low persistence)
- **Birth value** marks geometric scale where topology changes

### 3D Reconstruction via Autoencoder (F3DR)

The hypothesis: If TDA captures essential topology, we should be able to reconstruct 3D coordinates from topological features.

**Model:** Dense autoencoder
```
Input (24-D) → [256] → [512] → [256] → Latent (256-D) 
           → [256] → [512] → [256] → Output (3N-D coordinates)
```

- Dropout + BatchNorm for regularization
- Bottleneck = "topological signature" that preserves shape information
- Low reconstruction error → Topology suffices for structure

### Chirality Classification (TCEPC)

**Core question:** Can TDA features distinguish L from D proteins?

**Method:**
1. Create mirror-image (D) proteins: (x, y, -z) reflection
2. Compute TDA for both L and D
3. Train classifiers on feature pairs
4. Test if L/D discrimination is possible using only TDA

**Outcomes:**
- **AUC = 0.5** → TDA is chirality-blind (as expected if TDA is invariant under reflection)
- **AUC >> 0.5** → TDA captures chirality information (unexpected! suggests projection hypothesis)
- **AUC depends on 3D coords, not TDA** → TDA is indeed chirality-invariant, but coordinates contain handedness cues
---

## Configuration & Runtime Notes

### Hardcoded Paths Warning ⚠️

This codebase uses hardcoded paths. **Before running any script**, update the `DEFAULT_CONFIG` dictionary at the top of each file:

```python
DEFAULT_CONFIG = {
    "tda_csv_path": "/your/path/to/tda_features.csv",
    "coords_dir": "/your/path/to/aligned_coords",
    "output_dir": "/your/path/to/results",
    "data_folder": "/your/path/to/PDB_files",
}
```

### Batch Processing

Large datasets (73,315 proteins) are processed in batches:
- Default batch size: 2,000 proteins
- Adjust `chunk_size` parameter if memory-limited
- Use `--start_idx` and `--end_idx` flags to process subsets
---

## Complete Pipeline Example

Run the full analysis from data download to chirality predictions:

```bash
# Step 1: Acquire data
python src/step1.0_Batch_Download_human_protein_PDB.py

# Step 2: TDA pretreatment
python src/step2.0_TDA_Pretreatment.py \
    --data_folder Human_PDB_Data \
    --start_idx 0 \
    --end_idx 73315

# Step 3: HDEA - aggregate and analyze
python src/step3.0_HDEA_batch_transfer_TDA.py
python src/step3.1_HDEA_dimensionality_reduction_cluster_analysis.py
python src/step3.2_HDEA_extract_all_barcode_points.py
python src/step3.3_HDEA_universal_barcode.py

# Step 4: Topological signature modeling
python src/step4.1_F3DR_high_dim_projection_regression.py \
    --start_idx 0 \
    --end_idx 73315
python src/step4.2_F3DR_combine_results.py
python src/step4.3_F3DR_analyze_error_distribution_and_fold_class.py

# Step 5: Chirality analysis
python src/step5.1_TCEPC_generate_mirrored_coords.py
python src/step5.2_TCEPC_build_chirality_dataset.py
python src/step5.3_TCEPC_tda_feature_recovery.py
python src/step5.4_TCEPC_chirality_model_benchmark.py
python src/step5.5_TCEPC_spiral_vs_protein_topo.py
python src/step5.6_TCEPC_chirality_visualization.py

# Supplementary materials (Optional)
python src/supplementary_information.py
```
---

## System Requirements

| Resource | Recommendation | Minimum |
|----------|----------------|---------|
| **CPU** | 16+ cores | 4 cores |
| **Memory (RAM)** | 32 GB | 16 GB |
| **Disk Space** | 100 GB+ | 50 GB |
| **GPU** | CUDA 11.x + cuDNN 8.x (optional but recommended) | Not required |

---

## Troubleshooting

### "ModuleNotFoundError" for TDA libraries

```bash
pip install --upgrade ripser gudhi
# If gudhi installation fails, try conda:
conda install -c conda-forge gudhi
```

### Memory Errors During Step 2–3

The full dataset consumes ~100 MB in memory. If errors occur:
1. Process in smaller chunks:
   ```bash
   python src/step2.0_TDA_Pretreatment.py --start_idx 0 --end_idx 10000
   python src/step2.0_TDA_Pretreatment.py --start_idx 10000 --end_idx 20000
   # ... continue in 10,000-protein blocks
   ```
2. Reduce batch size in autoencoder training (Step 4):
   ```python
   BATCH_SIZE = 500  # Instead of 2000
   ```

### Hardcoded Path "File Not Found"

Every script has a `DEFAULT_CONFIG` dict at the top. Edit it:
```python
DEFAULT_CONFIG = {
    "tda_csv_path": "C:\\Users\\your_name\\Desktop\\Research_results\\step2.1_tda_features.csv",
    "coords_dir": "C:\\Users\\your_name\\Desktop\\Research_results\\step1.0_Pretreatment_results\\step1.0_aligned_coords",
    ...
}
```
Use absolute paths with proper escaping (`\\` on Windows).

---

## References & Further Reading

**Topological Data Analysis (TDA) Foundations:**
- Edelsbrunner, H., & Morozov, D. (2012). Persistent homology — theory and practice. *Current Trends in Computational Topology*.
- Chazal, F., & Michel, B. (2021). An introduction to topological data analysis. *arXiv preprint arXiv:1710.04019*.

**TDA Software:**
- Ripser: Fast computation of persistent homology ([ripser.org](https://ripser.org))
- GUDHI: Geometry Understanding in Higher Dimensions ([gudhi.inria.fr](https://gudhi.inria.fr))

**Protein Structure & Chirality:**
- Richardson, J. S. (1981). Anatomy and taxonomy of protein structure. *Advances in protein chemistry, 34*, 167-339.
- Brändén, C. I., & Jones, T. A. (1990). Between objectivity and subjectivity. *Nature, 343*(6256), 687-689.

**Machine Learning for Protein Analysis:**
- Senior, A. W., Evans, R., Jumper, J., et al. (2020). Improved protein structure prediction using potentials from deep learning. *Nature, 577*(7792), 706-710.
- Ingraham, J., Garg, V., Barzilay, R., & Jaakkola, T. (2019). Generative models for graph-based protein design. *Advances in Neural Information Processing Systems, 32*, 15820-15831.

**Dimensionality Reduction:**
- McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform manifold approximation and projection for dimension reduction. *arXiv preprint arXiv:1802.03426*.
- van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. *Journal of machine learning research, 9*(86), 2579-2605.

---

## Contact & Support

- **Issues & Questions:** Open an issue on GitHub or contact via email
- **Email:** rjingraymond88@gmail.com

