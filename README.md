# HIV-drug-discovery

An open-source ensemble of a **Graph Neural Network** and the **MolFormer-XL transformer** for **HIV activity classification**, built on a $0 compute budget (free Google Colab + local Apple Silicon).

The goal: provide a transparent, fully reproducible baseline that helps prioritize candidate molecules before wet-lab assays — without requiring institutional GPU access.

**Status:** Research prototype.

---

## Headline results

| Metric | Value | Notes |
|---|---|---|
| Per-fold test AUC (MolFormer, 5-fold scaffold split) | **0.8057 ± 0.0173** | Held-out test sets; the honest headline number |
| Per-fold test AUC (GNN v5b, 5-fold scaffold split) | 0.7740 ± 0.010 | From-scratch GATv2Conv GNN |
| Out-of-fold ensemble AUC (stacker, 4 folds) | 0.8581 | Logistic regression on OOF val predictions |

All numbers come from scaffold-based 5-fold cross-validation on the MoleculeNet HIV dataset (~41k molecules, ~3.5% actives). Scaffold splitting prevents structurally-related molecules from leaking across train/test, making this a much harder evaluation than random splits.

---

## What's actually in this repo

The project trains and combines two complementary models:

1. **GNN (`src/main.py`, `src/model.py`, `src/features.py`)**
   GATv2Conv-based graph neural network with edge features, descriptor concatenation, focal loss, scaffold k-fold CV.

2. **MolFormer-XL (`src/molformer_train.py`, `src/molformer_model.py`)**
   Fine-tunes IBM's MolFormer-XL (47M params, pretrained on 1.1B molecules) with focal loss on the same scaffold splits.

3. **Stacker (`src/fit_ensemble_stacker.py`)**
   Fits a 3-parameter logistic regression on out-of-fold validation predictions:
   `P_final = sigmoid(w_gnn · P_gnn + w_mf · P_mf + b)`
   Saves auto-tuned thresholds (F1-max, Youden's J, base-rate) into the same `.pt` file.

4. **Inference (`src/ensemble_inference.py`, `src/inference.py`, `src/eval_mixed.py`)**
   Combines GNN + MolFormer predictions either by weighted average (default) or by the trained stacker.

---

## Dataset

**Source:** MoleculeNet HIV — https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv

- ~41,000 small molecules
- Binary `HIV_active` label
- ~3.5% positive class (severe imbalance)

### Cleaning & splits
- SMILES parsed via RDKit; molecules with parse failures or empty graphs are dropped.
- Murcko scaffolds extracted with RDKit; identical scaffolds are kept together within a single fold to prevent leakage.
- 5-fold scaffold split deterministic from a fixed seed — both models train on bit-for-bit identical splits, so their predictions are honestly combinable.

---

## Molecular representation

### GNN side (2D graph + descriptors)

**Atom features** (23-dim per atom): atom type one-hot (10), degree one-hot (6), formal charge, hybridization one-hot (4), aromaticity, hydrogen count.

**Bond features** (8-dim per edge): bond type one-hot (4), conjugation, ring membership, stereo (2).

**Global descriptors** (RDKit-derived, per molecule, z-scored): MW, LogP, TPSA, H-bond donors/acceptors, rotatable bonds, ring counts, aromatic ring count, fraction Csp3, etc.

### MolFormer side
Tokenizes SMILES with MolFormer's custom tokenizer; the 47M-parameter pretrained backbone is fine-tuned end-to-end on the binary task.

---

## Model architecture (GNN, v5b)

- Atom projection: `Linear(23 → 256)`
- Edge projection: `Linear(8 → 256)`
- 4 × `GATv2Conv` layers (4 heads, residual + LayerNorm)
- JumpingKnowledge: concat per-layer outputs
- Readout: `GlobalAttention` + mean pooling
- MLP head: `Linear → ReLU → Dropout → Linear → 1` with descriptor features concatenated before the head

### Training (both models)
- Loss: **focal loss** (α=0.75, γ=2.0) — handles 3.5% imbalance better than reweighted BCE
- Optimizer: AdamW
- Scheduler: OneCycleLR (GNN) / linear warmup (MolFormer)
- Early stopping on validation AUC with `MIN_EPOCHS=30` and `val_pool=3` (rolling-mean validation to reduce noise)
- Mixed precision on CUDA only

---

## Combining the models

Two modes:

### 1. Weighted average (simple)
```
P_final = w · P_gnn + (1 - w) · P_mf
```
Both `P_gnn` and `P_mf` are themselves means over their 5 fold checkpoints. Tune `w` with `--gnn-weight` (default 0.5).

### 2. Logistic-regression stacker (recommended)
Fit on out-of-fold predictions:
```
P_final = sigmoid(w_gnn · P_gnn + w_mf · P_mf + b)
```
Strictly more flexible than weighted averaging — it learns the relative reliability of each model AND a calibration intercept that corrects focal-loss-induced overconfidence.

Empirically, the stacker assigns coefficients (3.36, 8.05, -6.89): MolFormer gets ~70% of the implied weight, and the negative intercept calibrates outputs to the 3.5% base rate.

---

## Threshold selection

Because the stacker calibrates probabilities to the base rate, the natural 0.5 cutoff is far too strict. The fit script computes three principled thresholds on the OOF predictions:

| Mode | Threshold | When to use |
|---|---|---|
| **Youden's J** (default) | ~0.05 | Virtual screening — high recall, accept noise |
| F1-max | ~0.10–0.23 | Balanced reporting (paper headline F1) |
| Base-rate match | ~0.17 | Top-k retrieval matching ~3.5% prior |

All three are saved into `ensemble_stacker.pt` and auto-loaded by `ensemble_inference.py`.

---

## Reproducing the results

### Train (Colab GPU recommended for MolFormer)
```bash
# GNN — full 5-fold CV
python src/main.py

# MolFormer — full 5-fold CV (use --start-fold N to resume)
python src/molformer_train.py
```

### Fit the stacker
```bash
python src/fit_ensemble_stacker.py
# Reads all available fold checkpoints, fits LR, saves src/ensemble_stacker.pt
```

### Inference
```bash
# Single SMILES
python src/ensemble_inference.py \
    --stacker src/ensemble_stacker.pt \
    --smiles 'CC1=CN(C(=O)NC1=O)[C@H]2C[C@H](N=[N+]=[N-])[C@@H](CO)O2'

# Sanity check on known actives
python src/ensemble_inference.py --stacker src/ensemble_stacker.pt --test-actives 100

# Honest precision/recall sweep on a labeled mixed set
python src/eval_mixed.py --file mixed_test.txt --n-active 100 \
    --stacker src/ensemble_stacker.pt
```

### Recover a missing test AUC after an interrupted run
```bash
python src/eval_fold.py --fold 2
```
Splits are deterministic, so this reproduces the exact test set the saved checkpoint never got to evaluate on.

---

## Compute footprint

- **Total training cost: $0.**
- MolFormer fine-tuning: free Colab T4 (~1 hr per fold, 5 folds total).
- GNN training: local Apple Silicon (M-series MPS) or Colab.
- Stacker fitting: ~10 minutes on local MPS.

For comparison, prior-art models in the same AUC range (e.g. Uni-Mol at ~0.83) typically used multi-GPU clusters for pretraining. This project shows that careful methodology + a strong pretrained encoder + ensembling can close most of that gap on a hobbyist budget.

---

## Limitations

- **3.5% positive class** — even at AUC 0.86, precision at high-recall thresholds is low. Useful for triage, not for replacing assays.
- **Scaffold split is harder than random** — some published HIV numbers use random splits and aren't directly comparable.
- **MolFormer fine-tuning is dominant** — the GNN's contribution is real but smaller than the transformer's. If you have larger compute, the GNN may not be worth the complexity vs. a longer MolFormer training run.
- **No 3D geometry, no target conditioning, no multi-task transfer.**

---

## File layout

```
src/
├── features.py               # Atom/bond/descriptor extraction + scaffold utilities
├── model.py                  # GATv2Conv GNN definition
├── main.py                   # GNN training pipeline (5-fold CV)
├── molformer_model.py        # MolFormer classifier wrapper
├── molformer_train.py        # MolFormer fine-tuning pipeline
├── fit_ensemble_stacker.py   # Logistic-regression stacker on OOF predictions
├── ensemble_inference.py     # GNN + MolFormer ensemble inference
├── eval_fold.py              # Recover test AUC from a saved checkpoint
├── eval_mixed.py             # Threshold sweep on a labeled mixed set
└── inference.py              # GNN-only inference (lighter, faster)
```

---

## Pretrained checkpoints

Trained MolFormer-XL and GNN fold checkpoints, per-fold normalization stats, the fitted ensemble stacker, and the preprocessed graph cache are archived on Zenodo:

> **DOI:** [10.5281/zenodo.19946459](https://doi.org/10.5281/zenodo.19946459)

Download the bundle and unzip into `src/` to run inference without retraining:

```bash
# After downloading hiv_classifier_artifacts_v1.zip from Zenodo
unzip hiv_classifier_artifacts_v1.zip -d src/
python src/ensemble_inference.py --stacker src/ensemble_stacker.pt --smiles 'CC(=O)OC1=CC=CC=C1C(=O)O'
```

---

## Citation

If you use this software or the pretrained checkpoints in research, please cite both:

> Agarwal, A. C. (2026). *Closing the gap on a $0 budget: ensembling public molecular foundation models for HIV bioactivity prediction*. Preprint, ChemRxiv.
> Source code: https://github.com/v659/HIV-drug-discovery
> Trained artifacts: https://doi.org/10.5281/zenodo.19946459
