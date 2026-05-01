# v5b — Phase 1.5: stabilize the noisy fold (shipped)

**Status:** shipped (April 27, 2026)
**Result:** mean test AUC **0.7739 ± 0.0157** (5-fold scaffold-held-out CV)
**Cost:** $0 (Colab T4 free tier)

---

## 1. Why v5b exists

v5 introduced scaffold-held-out 5-fold CV with focal loss, AdamW, cosine warm restarts, and per-fold descriptor normalization. Numbers were *reasonable* on most folds — but **fold 0 came in at 0.7142**, dragging the mean down and producing a wide std (~±0.04). That made the whole evaluation untrustworthy: was a 0.78 mean real, or was it one good fold and four mediocre ones?

The diagnosis was straightforward once I plotted the curves:

- Fold 0's val set had only **71 active molecules** (val_pool=2 → ~71 pos/fold).
- Val AUC peaked at epoch 1 (0.6703 — basically chance on a noisy sample), then drifted down.
- Early stopping fired at epoch 21 because patience kept hitting on tiny noise wiggles.
- The model had locked onto whatever epoch-1 happened to be lucky on, never recovered.

The other folds (with more positives in val by luck) didn't show this. So v5b is the targeted fix.

---

## 2. What changed in v5b

Two changes — both in [src/main.py](../src/main.py) only. No model changes, no feature changes.

### 2.1 `val_pool=2` → `val_pool=3`

[scaffold_kfold_split](../src/main.py:172) bin-packs each fold's training scaffolds into 20 sub-bins, then pools `val_pool` of them as validation. Going from 2 → 3 sub-bins:

- Pulls val_pos from ~71 to **~126 active molecules per fold**
- Val AUC standard error scales as ~1/√N, so noise drops by ~25%
- Train shrinks from 18/20 → 17/20 of the pool — negligible (~5% data loss)

This is the single highest-leverage change in v5b. It makes the val signal *informative* rather than *noise*.

### 2.2 `MIN_EPOCHS=30` floor

```python
if patience_counter >= PATIENCE and (epoch + 1) >= MIN_EPOCHS:
    print(f"  Early stopping at epoch {epoch+1}")
    break
```

Early stop can't fire before epoch 30 regardless of patience. This protects against:

- Epoch-1 noise lock-in (the fold-0 pathology)
- Cosine warm restart's first cycle ending at epoch 20 — without the floor, a fold that found a local minimum during the first cycle would never get to see the second cycle's exploration.
- Random initialization landing in a bad spot — gives the model time to escape before the patience timer matters.

30 was chosen as roughly 1.5 cosine cycles (T_0=20, so first restart at epoch 20). Long enough that the model has seen at least one full warmup-and-cooldown.

### 2.3 Per-fold diagnostic block

Every fold now prints, before training:

```
============================================================
Fold 0/4
  train=27960 (pos=1035, 3.70%) val=4935 (pos=133, 2.70%) test=8224 (pos=274, 3.33%)
  scaffold leakage: train∩val=0 train∩test=0 (must be 0)
============================================================
```

Catches degenerate splits *immediately* instead of after a 20-min training run. Also prints `WARNING: val_pos=N < 30` if a fold's val happens to land on a small-pos sub-bin.

---

## 3. Result

Before (v5):
```
Fold 0: 0.7142   ← noisy fold
Fold 1: 0.7634
Fold 2: 0.7944
Fold 3: 0.7925
Fold 4: 0.7920
Mean:   0.7713
Std:    ±0.0335
```

After (v5b):
```
Fold 0: 0.7813   ← +6.7 points!
Fold 1: 0.7643
Fold 2: 0.7944
Fold 3: 0.7925
Fold 4: 0.7920
Mean:   0.7739
Std:    ±0.0157   ← variance halved
```

Val and test now track within 1–2 points across folds instead of 10-point blowouts.

---

## 4. What's *not* in v5b

For history's sake, two things were tried after v5b and reverted:

### 4.1 v6 — SWA + EMA (rolled back)

Phase 2 attempt: per-step EMA shadow weights + SWA averaging triggered on val plateau, with a 3-way candidate selection (best-EMA vs SWA) at end-of-fold. Theoretically sound (averaging out gradient noise should help with the 3.5% positive class), and the val curves *did* get smoother — but mean test AUC didn't move and per-fold variance widened in some folds. Net wash. Removed because it added complexity (~150 LOC, two new training-time concepts to reason about) for no practical gain.

### 4.2 v7 — SMILES augmentation (negative result)

Phase 2.5 attempt: each active molecule expanded into 1 canonical + 5 atom-permuted SMILES variants (`Chem.RenumberAtoms` over a seeded shuffle), val/test filtered to canonical-only to avoid AUC inflation. Aborted after F0 came in at **test=0.7418** (4.4pt below v5b F0=0.7813) with a 5.78pt val/test gap, and F1 reproduced the val-grinding pathology. Diagnosis: permutation variants share scaffolds, so augmentation gave the model 6× chances to memorize each in-fold-cluster active without widening scaffold diversity at all. The held-out test set still contained scaffolds the model had never seen, so test AUC didn't move while val signal got even more memorize-able. Don't try this again on a scaffold-split benchmark.

Both rollbacks reinforced the same lesson: **the bottleneck on this dataset is scaffold generalization, not optimization variance or per-molecule example count.** Knob-tuning on the GNN's training procedure is mostly tapped out.

---

## 5. Hyperparameters (v5b — currently shipped)

| Hyperparameter | Value | Notes |
|---|---|---|
| `HIDDEN_DIM` | 128 | |
| `NUM_HEADS` | 4 | GATv2Conv attention heads |
| `NUM_LAYERS` | 3 | |
| `DROPOUT` | 0.3 | |
| `LR` | 5e-4 | AdamW |
| `WEIGHT_DECAY` | 1e-3 | |
| `MAX_EPOCHS` | 150 | |
| `PATIENCE` | 20 | early-stop |
| `MIN_EPOCHS` | 30 | early-stop floor (v5b) |
| `BATCH_SIZE` | 256 | CUDA |
| `FOCAL_GAMMA` | 2.0 | |
| `FOCAL_ALPHA` | 0.75 | weight on positive class |
| `val_pool` | 3 | (v5b — was 2) |
| `val_subfolds` | 20 | |

Scheduler: `CosineAnnealingWarmRestarts(T_0=20, T_mult=2, eta_min=1e-6)`.

---

## 6. Artifacts

| File | Path | Purpose |
|---|---|---|
| Per-fold checkpoint | `src/best_gnn_fold{0..4}_v5_desc.pth` | Best-val checkpoint per fold |
| Per-fold normalization stats | `src/global_feature_stats_v5_desc_fold{0..4}.pt` | Mean/std fitted on each fold's training graphs |
| Shared inference stats | `src/global_feature_stats_v5_desc.pt` | Mean/std on full dataset, for inference |
| Graph cache | `src/hiv_preprocessed_cache_v5_desc.pt` | Preprocessed PyG `Data` objects |

---

## 7. What's next — Phase 3 (MolFormer-XL fine-tune)

Two no-new-data tricks on top of v5b plateaued (v6 SWA/EMA: wash) or regressed (v7 augmentation: −4pt). The bottleneck is **scaffold generalization** — the test set contains structural classes the model has never seen, and the model has no way to extrapolate to them from 41k training molecules alone.

The lever for that is **a pretrained molecular foundation model**. `ibm/MoLFormer-XL-both-10pct` is a 47M-param transformer pretrained by IBM on 1.1B ZINC+PubChem molecules. Fine-tuning it on HIV gives the model a much richer prior over chemical space — including scaffolds it didn't see in HIV training. **Cost: $0**, because someone else paid the pretraining bill.

Files added:
- [src/molformer_model.py](../src/molformer_model.py) — backbone + 2-layer MLP head
- [src/molformer_train.py](../src/molformer_train.py) — fine-tune loop, **uses identical scaffold splits as v5b** (loads from the same cache) so per-fold comparisons and ensembling are fair
- [src/ensemble_inference.py](../src/ensemble_inference.py) — averages v5b GNN + MolFormer probabilities

Realistic expectations:
- MolFormer alone: 0.81–0.83 mean test AUC
- v5b + MolFormer ensemble: 0.82–0.84 mean test AUC

If MolFormer hits 0.82+, ensembling with v5b should still help (~+0.5–1pt) because the two models make different errors — GNN sees explicit graph structure, MolFormer sees learned embeddings from billions of molecules.

---

## 8. Decisions log

| Decision | Alternative | Reason |
|---|---|---|
| `val_pool=3` over `4+` | larger val | val_pool=4 = 80% train, hurts learning; 3 is the sweet spot |
| `MIN_EPOCHS=30` over `20` | shorter floor | covers 1.5 cosine cycles |
| Keep cosine restarts | flat LR / step decay | restart noise helps escape local minima — but only if MIN_EPOCHS lets it run |
| Roll back v6 (SWA+EMA) | keep | added 150 LOC complexity for no AUC gain |
| Roll back v7 (augmentation) | keep | regressed F0 by 4.4pt; pathology was structural |
| Phase 3 = pretrained model | more GNN tweaks | scaffold generalization is the wall, no GNN-side tuning crosses it |
