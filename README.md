# HIV-drug-discovery

An **open-source Graph Neural Network (GNN) framework** for **HIV activity classification**, designed to help researchers prioritize candidate molecules **before costly wet-lab testing**.

**Status:** Alpha (research prototype)

---

## Motivation

HIV drug discovery is expensive, slow, and experimentally intensive.  
This project aims to provide a **free, transparent, and reproducible baseline model** that can:

- Screen large molecular libraries
- Learn structure–activity relationships
- Reduce the number of compounds requiring laboratory assays

This is **not a production model** and should be used for **research and experimentation only**.

---

## Dataset

**Source:**  
https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv

**Description:**
- ~41,000 small molecules
- Binary label: `HIV_active`
- Heavy class imbalance (~2.5% actives)
- Mixed assay conditions
- Contains invalid and problematic SMILES

---

## Dataset Cleaning & Filtering

### 1. Molecule Validation
A molecule is removed if:
- SMILES parsing fails
- Graph construction fails
- Atom feature matrix is empty
- Scaffold extraction fails

Invalid edge cases (e.g., molecules with no bonds) are handled safely.

### 2. Scaffold Extraction
- Murcko scaffolds are generated using RDKit
- Used for **scaffold-based cross-validation**
- Prevents data leakage across folds

### 3. PAINS Filtering
- PAINS (Pan-Assay Interference Compounds) are filtered using RDKit `FilterCatalog`
- Helps reduce false positives common in high-throughput screening

> **Note:** The current model uses **2D molecular graphs only**.  
> 3D conformer generation and geometry-aware modeling are **not enabled**.

---

## Molecular Representation (2D)

### Nodes (Atoms)
Each atom is represented by a **26-dimensional feature vector**, including:
- Atom type
- Atom degree
- Formal charge
- Hybridization
- Aromaticity
- Hydrogen count

### Edges (Bonds)
- Undirected covalent bonds
- Purely topological (no geometry)

---

## Model Architecture

**Type:** 2D Graph Neural Network (GCN-based)

### Global Readout
- Global mean pooling
- Global max pooling

### Prediction Head
```python
self.mlp = nn.Sequential(
    nn.Linear(hidden_dim * 2, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 1)
)
```

### Training
- Loss: BCEWithLogitsLoss (class-weighted)
- Optimizer: Adam + weight decay
- Scheduler: ReduceLROnPlateau
- Early stopping on validation AUC

---

## Evaluation

- 5-fold scaffold-based cross-validation
- Metric: ROC-AUC

**Expected performance:** ~0.65–0.72 AUC (baseline)

---

## Limitations

- No bond features
- No 3D geometry
- No target conditioning
- Dataset noise from mixed assays

---

## Roadmap

- GIN / GAT architectures
- Bond feature integration
- Geometry-aware GNNs (EGNN)
- Multi-conformer pooling
- Model ensembling

---

## Citation Requirement

If you use this software in research, you **must cite**:

Arjun Chandra Agarwal,  
“HIV drug classification”,  
2026,  
https://github.com/v659/HIV-drug-discovery
