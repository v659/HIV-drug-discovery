"""GNN training pipeline for HIV bioactivity classification — v5b.

Trains the `HIVGNN` (defined in model.py) under scaffold-held-out 5-fold
cross-validation. This is the reference training script that produces the
`best_gnn_fold{0..4}_v5_desc.pth` checkpoints used by:
  - inference.py (single-model or GNN-only ensemble)
  - ensemble_inference.py (GNN + MolFormer combined ensemble)
  - molformer_train.py (reuses this script's exact scaffold splits via cache)

High-level flow:
  1. Load (or build) the preprocessed graph cache.
  2. Compute scaffold 5-fold splits with val_pool=3 sub-bins for validation.
  3. For each fold:
     a. Fit per-fold z-score stats on training molecules' descriptor features.
        Save stats so inference.py can apply the same normalization.
     b. Apply normalization non-destructively (each fold restarts from the
        raw features stored in `raw_global_features`).
     c. Train HIVGNN with focal loss, AdamW, cosine warm restarts.
     d. Track best val AUC; save checkpoint when it improves.
     e. Early stop after PATIENCE epochs of no improvement, but only after
        MIN_EPOCHS (protects noisy folds from epoch-1 noise lock-in).
     f. Reload best checkpoint and report test AUC.
  4. After all folds: also save a "shared" stats file fit on the full
     dataset, used at inference time when no fold-specific stats are available.
  5. Print per-fold and mean ± std test AUC.

v5b vs v5: two stability fixes after fold-0 came in noisy (0.7142) on v5.
  1. val_pool=3 (was 2): smoother val AUC signal. Pulls active count in val
     from ~71 to ~126/fold. AUC variance scales as 1/√n_pos so noise drops ~25%.
  2. MIN_EPOCHS=30: early-stop floor. Without it, fold 0's epoch-1 noise spike
     was getting locked in as best-val and the model never recovered.

Result: mean test AUC 0.7739 ± 0.0157 (v5 had ±0.04+ across folds).
"""
import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from features import (
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    USE_MORGAN,
    get_scaffold,
    mol_to_graph,
)
from model import HIVGNN

# ---------------------------------------------------------------------------
# Hyperparameters. Tuned for the HIV dataset (41k molecules, 3.5% positive,
# scaffold-held-out evaluation). Most knobs here are not free choices — they
# trade off in known ways against the dataset's specific properties.
# ---------------------------------------------------------------------------
HIDDEN_DIM = 128            # GNN channel width. Sweet spot — bigger overfits, smaller underfits.
NUM_HEADS = 4               # GATv2 attention heads. 128/4 = 32 dim per head.
NUM_LAYERS = 3              # GATv2 blocks. >3 oversmooths even with residuals.
DROPOUT = 0.3               # Used in conv attention AND MLP head.
LR = 5e-4                   # AdamW base LR. Higher → unstable on small batches.
WEIGHT_DECAY = 1e-3         # L2 regularization strength.
MAX_EPOCHS = 150            # Hard ceiling. Early stop usually fires earlier.
PATIENCE = 20               # Epochs without val-AUC improvement before considering stop.
MIN_EPOCHS = 30             # Don't early-stop before this. Covers 1.5 cosine cycles
                            # (T_0=20). Without this, fold 0's epoch-1 noise spike
                            # got locked in as "best" and the model never recovered.
NUM_FOLDS = 5
SEED = 42                   # For torch + numpy. Scaffold splits are deterministic given this.

# Focal loss hyperparameters. Tuned for the 96.5/3.5% class imbalance.
FOCAL_GAMMA = 2.0           # Higher = more focus on hard examples. 2.0 is the standard default.
FOCAL_ALPHA = 0.75          # Weight on positive class. Values above 0.5 upweight actives.
                            # Empirically 0.75 balanced precision/recall better than higher
                            # values that aggressively chased recall at precision's cost.

# Cache version bump invalidates all previously-built cache files. Bump this
# whenever feature extraction changes (e.g. new descriptor, atom feature dim
# change). Existing cache will be rebuilt automatically.
CACHE_VERSION = 5
VARIANT = "full" if USE_MORGAN else "desc"  # "full" = desc+morgan, "desc" = desc only
DATA_DIR = Path(__file__).parent
CACHE_PATH = DATA_DIR / f"hiv_preprocessed_cache_v5_{VARIANT}.pt"
CSV_PATH = DATA_DIR / "hiv.csv"

# Auto-detect best available compute. CUDA preferred (T4/L4/A100), then MPS
# (Apple Silicon), then CPU. Some optimizations (AMP, persistent workers,
# bigger batch) only kick in on CUDA where they're known to be safe.
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# Per-device batch sizing and worker count. CUDA can handle the big batch +
# multiple workers. MPS has known issues with multi-process DataLoader fork.
if DEVICE.type == "cuda":
    BATCH_SIZE = 256
    NUM_WORKERS = 2
    USE_AMP = True               # fp16 mixed precision — ~2x faster on Tensor Cores
else:
    BATCH_SIZE = 64
    NUM_WORKERS = 0 if DEVICE.type == "mps" else 2
    USE_AMP = False              # MPS AMP support is unreliable as of PyTorch 2.9

# CUDA performance tweaks — TF32 matmul (faster, slightly less precise) and
# cudnn autotuning (picks best conv algorithm for your input shape).
torch.set_float32_matmul_precision("high")
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
def csv_hash(path):
    """Stable SHA-256 of the CSV file. Used to invalidate the graph cache
    automatically if the underlying data changes — without this, you could
    silently train on a stale cache built from a different CSV version."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def load_or_build_cache():
    """Load preprocessed graphs from cache, or build them from the CSV.

    Cache invalidation rules (rebuild if any fail):
      - Cache file doesn't exist
      - cache_version mismatch (we bumped CACHE_VERSION since this was built)
      - csv_hash mismatch (the source CSV changed)

    The cache contains:
      graphs:    list of PyG Data objects (one per valid molecule)
      scaffolds: list of Murcko scaffold SMILES (parallel to graphs)
      smiles:    list of original SMILES strings (parallel to graphs)
      meta:      dict with cache_version + csv_hash

    Building from CSV takes ~2 minutes; loading from cache takes <1 second.
    Skipped molecules (RDKit-invalid SMILES) are dropped quietly here, so
    `len(graphs)` < `len(df)` is normal.
    """
    if CACHE_PATH.exists():
        cache = torch.load(CACHE_PATH, weights_only=False)
        meta = cache.get("meta", {})
        if (
            meta.get("cache_version") == CACHE_VERSION
            and meta.get("csv_hash") == csv_hash(CSV_PATH)
        ):
            print(f"Loaded cache: {len(cache['graphs'])} graphs")
            return cache
        print("Cache version mismatch, rebuilding...")

    print("Building graph cache from CSV...")
    # dropna ensures we don't crash on missing labels or SMILES.
    df = pd.read_csv(CSV_PATH).dropna(subset=["smiles", "HIV_active"])

    graphs = []
    scaffolds = []
    smiles_list = []
    skipped = 0

    # itertuples is faster than iterrows for this size of dataframe.
    for row in tqdm(
        df.itertuples(index=False), total=len(df), desc="Processing molecules"
    ):
        smi = row.smiles
        label = float(row.HIV_active)
        g = mol_to_graph(smi, label=label)
        if g is None:
            # RDKit couldn't parse this SMILES — skip silently. Counted for
            # the summary printout below. Very rare in HIV.csv (<10 mols).
            skipped += 1
            continue
        graphs.append(g)
        scaffolds.append(get_scaffold(smi))
        smiles_list.append(smi)

    print(f"Built {len(graphs)} graphs, skipped {skipped} invalid molecules")

    cache = {
        "graphs": graphs,
        "scaffolds": scaffolds,
        "smiles": smiles_list,
        "meta": {
            "cache_version": CACHE_VERSION,
            "csv_hash": csv_hash(CSV_PATH),
        },
    }
    torch.save(cache, CACHE_PATH)
    print(f"Saved cache to {CACHE_PATH}")
    return cache


def fit_global_feature_stats(graphs, eps=1e-6):
    """Fit per-feature z-score stats over a set of graphs' global_features.

    IMPORTANT: this should only be called on the *training* graphs of a fold —
    NEVER on val or test. Fitting normalization stats on val/test data is a
    classic form of data leakage that artificially inflates measured AUC.

    Only the descriptor block (first DESCRIPTOR_DIM dims) is normalized.
    The Morgan fingerprint block (if USE_MORGAN=1) is binary 0/1, so we
    leave it alone (mean=0, std=1 → identity transform).

    `eps` clamps the std floor to avoid division-by-zero on constant features
    (e.g. NumRadicalElectrons is 0 for >99% of molecules).

    Returns (mean, std) tensors of shape [GLOBAL_FEATURE_DIM].
    """
    # Local import avoids a circular import at module load time.
    from features import DESCRIPTOR_DIM

    # Stack all graphs' global features into one big [N, D] tensor for stats.
    all_gf = torch.cat([g.global_features for g in graphs], dim=0)  # [N, D]
    mean = all_gf.mean(dim=0)
    std = all_gf.std(dim=0)

    # Build a boolean mask: True for the first DESCRIPTOR_DIM positions
    # (continuous-valued descriptors), False for the rest (Morgan FP bits).
    norm_mask = torch.zeros_like(mean, dtype=torch.bool)
    norm_mask[:DESCRIPTOR_DIM] = True
    # Where the mask is False (Morgan bits), force mean=0/std=1 so the
    # (x - mean) / std transform becomes the identity.
    mean = torch.where(norm_mask, mean, torch.zeros_like(mean))
    std = torch.where(norm_mask, std, torch.ones_like(std))
    # Floor std to eps to handle constant-valued descriptors safely.
    std = torch.clamp(std, min=eps)
    return mean, std


def apply_global_feature_norm(graphs, raw_features, mean, std):
    """Apply z-score normalization in a non-destructive, restartable way.

    `raw_features` is the snapshot of original (unnormalized) global_features
    for every graph, captured ONCE at the top of main(). For each fold we
    replace `g.global_features` with a freshly-computed (raw - mean) / std,
    so applying a different fold's stats later just overwrites cleanly
    without compounding normalizations.

    Without this pattern, the second fold would normalize already-normalized
    values from the first fold — a sneaky bug that's invisible in unit tests
    but silently degrades model quality.
    """
    for g, raw in zip(graphs, raw_features):
        g.global_features = (raw - mean) / std


def _bin_pack_groups(groups, n_bins):
    """Greedy "Longest-Processing-Time-first" bin packing.

    `groups` is a list of lists, each containing the molecule indices that
    share one scaffold. ASSUMED to be pre-sorted largest-first by the caller.

    Algorithm: walk through groups in descending size order, assigning each
    to whichever bin currently has the smallest total. This produces near-
    equal bin sizes (~3-5% imbalance typically) despite scaffold groups
    varying from 1 to ~500 molecules each — it's the standard greedy
    approximation algorithm for the multiway partition problem.

    Returns:
        bin_indices: list of `n_bins` lists, each containing molecule indices.
        bin_groups:  list of `n_bins` lists, each containing scaffold groups
                     (used by the caller to do further sub-binning of train).
    """
    bin_sizes = [0] * n_bins
    bin_indices = [[] for _ in range(n_bins)]
    bin_groups = [[] for _ in range(n_bins)]
    for group in groups:
        # Assign this group to whichever bin currently has the smallest size.
        smallest = int(np.argmin(bin_sizes))
        bin_indices[smallest].extend(group)
        bin_groups[smallest].append(group)
        bin_sizes[smallest] += len(group)
    return bin_indices, bin_groups


def scaffold_kfold_split(scaffolds, n_folds=5, val_subfolds=20, val_pool=3, seed=42):
    """Build scaffold-held-out 5-fold splits with within-fold val partitioning.

    Each returned (train_idx, val_idx, test_idx) tuple has these guarantees:
      - Train, val, and test contain DISJOINT sets of scaffolds.
      - Test is fold i's scaffolds (~20% of data).
      - Val is `val_pool` sub-bins out of `val_subfolds` from the train pool
        (~val_pool/val_subfolds of the train pool, which is 3/20 = 15%).
      - Train is everything else.

    Why this matters: random splits would let the model cheat by memorizing
    scaffold→label associations — many molecules in the dataset share
    scaffolds (e.g. hundreds of compounds may share a benzene-pyrimidine
    core). Scaffold-held-out evaluation is the honest test of whether the
    model has learned generalizable structural patterns.

    Args:
        scaffolds:    list of Murcko scaffold SMILES, one per molecule
                      (parallel to the molecule list).
        n_folds:      Number of outer folds. 5 is standard for this dataset.
        val_subfolds: Granularity for partitioning the training pool. Higher
                      = finer-grained val candidates. 20 keeps val sub-bins
                      large enough to be statistically meaningful.
        val_pool:     How many sub-bins to pool as validation. Default 3.
                      Was 2 in v5 — bumping to 3 was the v5b stability fix.
        seed:         Currently unused (the algorithm is deterministic given
                      the input scaffold ordering), but kept for API symmetry.

    Returns:
        List of (train_idx, val_idx, test_idx) tuples, length == n_folds.
        Each idx is a list of integers indexing into `scaffolds`.
    """
    # Step 1: group all molecule indices by their scaffold SMILES.
    # scaffold_to_indices[s] = [indices of all molecules with scaffold s]
    scaffold_to_indices = defaultdict(list)
    for idx, scaffold in enumerate(scaffolds):
        scaffold_to_indices[scaffold].append(idx)

    # Sort groups largest-first so bin packing produces balanced bins.
    scaffold_groups = sorted(
        scaffold_to_indices.values(), key=len, reverse=True
    )

    # Step 2: outer split — pack scaffold groups into n_folds bins.
    # `fold_indices[i]` is the list of molecule indices in fold i's test set.
    # `fold_groups[i]` is the list of scaffold groups used in that fold (we
    # need both — the indices for slicing data, the groups for re-packing
    # the training pool below).
    fold_indices, fold_groups = _bin_pack_groups(scaffold_groups, n_folds)

    splits = []
    for i in range(n_folds):
        test_idx = fold_indices[i]

        # Step 3a: assemble the training pool — every scaffold group NOT in
        # fold i. Re-sort largest-first for the second bin-packing step.
        train_pool = []
        for j in range(n_folds):
            if j != i:
                train_pool.extend(fold_groups[j])
        train_pool = sorted(train_pool, key=len, reverse=True)

        # Step 3b: inner split — re-pack the training pool into 20 sub-bins
        # (much finer-grained than the outer 5-way split). This gives us
        # "scaffold sub-clusters" we can use as validation without sacrificing
        # too much training data.
        sub_indices, _ = _bin_pack_groups(train_pool, val_subfolds)

        # Step 3c: pick which sub-bins become val. We rotate the choice
        # across outer folds — fold 0 uses sub-bins {0,1,2}, fold 1 uses
        # {3,4,5}, etc. — so that no two outer folds share the same val set.
        # This rotation lets you reason about "what scaffolds the val set
        # saw" in a principled way across folds.
        val_subfold_set = {
            (val_pool * i + k) % val_subfolds for k in range(val_pool)
        }
        val_idx = []
        train_idx = []
        for k in range(val_subfolds):
            if k in val_subfold_set:
                val_idx.extend(sub_indices[k])
            else:
                train_idx.extend(sub_indices[k])

        splits.append((train_idx, val_idx, test_idx))

    return splits


def focal_bce_loss(logits, targets, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA):
    """Binary focal loss with class-weighting alpha.

    Standard BCE on imbalanced data lets the dominant class dominate gradients
    — the model learns to predict "negative" on everything and gets ~96.5%
    accuracy doing nothing useful. Focal loss fixes this in two ways:

    1. The (1 - p_t)^gamma term DOWNWEIGHTS easy/correctly-classified examples.
       When p_t (probability assigned to correct class) is high (~0.95),
       (1 - 0.95)^2 = 0.0025 → that example contributes ~0.25% of normal loss.
       When p_t is low (~0.3), (1 - 0.3)^2 = 0.49 → contributes ~half. So the
       gradient signal comes mostly from hard, uncertain, or wrong examples.

    2. The alpha factor UPWEIGHTS the positive class explicitly. alpha=0.75
       means positives count 0.75 in the loss, negatives count 0.25 — a 3:1
       ratio that partially compensates for the 96.5/3.5 imbalance.

    Setting alpha=0.5 + gamma=0 recovers vanilla BCE.
    """
    # Per-sample BCE (no reduction yet — we need to weight each element).
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    # p_t = predicted prob of the *correct* class for each sample.
    # If target=1: p_t = probs (the prob of being positive).
    # If target=0: p_t = 1 - probs (the prob of being negative).
    p_t = probs * targets + (1 - probs) * (1 - targets)
    # alpha_t = class-balancing factor per sample.
    # Positives get alpha (=0.75), negatives get 1-alpha (=0.25).
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    # Focal weight combines class balancing with hard-example mining.
    focal_weight = alpha_t * (1 - p_t).pow(gamma)
    return (focal_weight * bce).mean()


def train_one_epoch(model, loader, optimizer, scaler):
    """Train for one epoch. Returns mean loss across the dataset.

    Implements two paths:
      - AMP (CUDA): forward in fp16 via autocast, scale loss for stable
        fp16 backward, unscale gradients before clipping (clipping needs
        fp32-equivalent magnitudes), step, update scaler's loss-scale
        based on whether infs/nans appeared.
      - Non-AMP (MPS/CPU): straight fp32 path.

    Gradient clipping at norm 1.0 is identical on both paths — prevents
    occasional gradient explosions on unusual molecules from destabilizing
    the optimizer.
    """
    model.train()
    # Running loss accumulated on-device to avoid CPU-GPU sync per batch.
    total_loss = torch.zeros((), device=DEVICE)
    total_samples = 0

    for batch in loader:
        batch = batch.to(DEVICE)
        # set_to_none=True is faster than =0 — actually frees the gradient
        # tensors instead of zeroing them in-place.
        optimizer.zero_grad(set_to_none=True)

        if USE_AMP:
            # fp16 forward + loss
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                out = model(batch)
                loss = focal_bce_loss(out, batch.y)
            # GradScaler scales the loss before backward to keep fp16
            # gradients in a representable range.
            scaler.scale(loss).backward()
            # Unscale before clipping so the clip threshold is meaningful
            # in fp32 terms.
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()  # adjust loss scale up/down based on inf/nan check
        else:
            # Non-AMP path — straight fp32, simpler.
            out = model(batch)
            loss = focal_bce_loss(out, batch.y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Weight by graph count so the average is correct even on partial
        # final batches.
        total_loss += loss.detach() * batch.num_graphs
        total_samples += batch.num_graphs

    return (total_loss / total_samples).item()


@torch.no_grad()
def evaluate(model, loader):
    """Compute ROC-AUC on a dataset.

    AUC is the right metric for severe class imbalance — accuracy is
    misleading (a 96.5% always-negative classifier "wins"), and PR-AUC
    requires picking a threshold. ROC-AUC is threshold-free and bounded
    in [0,1] with random = 0.5.

    Returns 0.0 if labels are single-class (e.g. an empty val set) — this
    can happen on degenerate splits and we'd rather report a placeholder
    than crash.
    """
    model.eval()
    pred_chunks = []
    label_chunks = []

    for batch in loader:
        batch = batch.to(DEVICE)
        out = model(batch)
        pred_chunks.append(torch.sigmoid(out))
        label_chunks.append(batch.y)

    preds = torch.cat(pred_chunks).cpu().numpy()
    labels = torch.cat(label_chunks).cpu().numpy()

    # Guard against single-class arrays (sklearn raises on these).
    if len(np.unique(labels)) < 2:
        return 0.0
    return roc_auc_score(labels, preds)


def parse_args():
    p = argparse.ArgumentParser(description="Train HIV GNN (scaffold 5-fold CV).")
    p.add_argument(
        "--fold-limit", type=int, default=None,
        help="Only run the first N folds (useful for smoke testing).",
    )
    p.add_argument(
        "--start-fold", type=int, default=0,
        help="Skip folds before this index. Use to resume an interrupted "
             "5-fold run (e.g., --start-fold 4 to train only fold 4 when "
             "folds 0-3 already have saved checkpoints).",
    )
    p.add_argument(
        "--epoch-limit", type=int, default=None,
        help="Cap epochs per fold (useful for smoke testing).",
    )
    return p.parse_args()


def main():
    """Main training entry point — runs scaffold 5-fold CV and reports per-fold AUC."""
    args = parse_args()
    # Reproducibility — note that PyG ops on CUDA still have some inherent
    # nondeterminism (e.g. scatter operations), so runs aren't bit-exact
    # but per-fold AUCs are typically reproducible to within ±0.005.
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    max_epochs = args.epoch_limit if args.epoch_limit is not None else MAX_EPOCHS

    cache = load_or_build_cache()
    graphs = cache["graphs"]
    scaffolds = cache["scaffolds"]

    # Snapshot the original global feature vectors BEFORE any normalization.
    # We'll restart from this snapshot at the top of each fold so that
    # per-fold normalization stats don't compound across folds.
    raw_global_features = [g.global_features.clone() for g in graphs]

    labels = [g.y.item() for g in graphs]
    num_pos = sum(labels)
    print(
        f"Dataset: {len(graphs)} molecules, {int(num_pos)} active "
        f"({100*num_pos/len(graphs):.1f}%)"
    )
    print(f"Device: {DEVICE} | batch={BATCH_SIZE} | workers={NUM_WORKERS} | amp={USE_AMP}")
    print(f"Focal loss: alpha={FOCAL_ALPHA}, gamma={FOCAL_GAMMA}")
    print(f"Atom dim: {ATOM_FEATURE_DIM}, bond dim: {BOND_FEATURE_DIM}, global dim: {GLOBAL_FEATURE_DIM}")

    splits = scaffold_kfold_split(scaffolds, n_folds=NUM_FOLDS, seed=SEED)
    if args.fold_limit is not None:
        splits = splits[: args.fold_limit]
        print(f"Fold limit: running {len(splits)} of {NUM_FOLDS} folds")
    if args.epoch_limit is not None:
        print(f"Epoch limit: capping at {max_epochs} epochs per fold")
    fold_aucs = []

    # =====================================================================
    # Per-fold loop. For each fold:
    #   1. Print diagnostic block (sizes, positive counts, scaffold-leakage check).
    #   2. Compute per-fold descriptor normalization on training graphs.
    #   3. Train HIVGNN with focal loss + AdamW + cosine warm restarts.
    #   4. Track best val AUC; save checkpoint when it improves.
    #   5. Early stop after PATIENCE epochs of no improvement (but not before
    #      MIN_EPOCHS — protects against fold-0-style noise lock-in).
    #   6. Reload best checkpoint and report test AUC.
    # =====================================================================
    for fold_i, (train_idx, val_idx, test_idx) in enumerate(splits):
        if fold_i < args.start_fold:
            print(f"Skipping fold {fold_i} (--start-fold={args.start_fold})")
            continue
        # ---- Diagnostic block: print sizes, positive counts, leakage check ----
        # The leakage check is a sanity assertion — if our scaffold split logic
        # were buggy and let a scaffold appear in both train and val/test,
        # we'd see leak_val > 0 or leak_test > 0 and abort the run mentally.
        # In practice these should always print 0.
        train_pos = sum(int(graphs[i].y.item()) for i in train_idx)
        val_pos = sum(int(graphs[i].y.item()) for i in val_idx)
        test_pos = sum(int(graphs[i].y.item()) for i in test_idx)
        train_scaf = {scaffolds[i] for i in train_idx}
        val_scaf = {scaffolds[i] for i in val_idx}
        test_scaf = {scaffolds[i] for i in test_idx}
        leak_val = len(train_scaf & val_scaf)
        leak_test = len(train_scaf & test_scaf)
        print(f"\n{'='*60}")
        print(f"Fold {fold_i}/{NUM_FOLDS-1}")
        print(
            f"  train={len(train_idx)} (pos={train_pos}, {100*train_pos/max(1,len(train_idx)):.2f}%) "
            f"val={len(val_idx)} (pos={val_pos}, {100*val_pos/max(1,len(val_idx)):.2f}%) "
            f"test={len(test_idx)} (pos={test_pos}, {100*test_pos/max(1,len(test_idx)):.2f}%)"
        )
        print(f"  scaffold leakage: train∩val={leak_val} train∩test={leak_test} (must be 0)")
        # If val happens to land on a sub-bin with very few actives, AUC will be
        # noisy regardless of the split being correct — flag it so the user
        # knows to interpret val numbers with skepticism for this fold.
        if val_pos < 30:
            print(f"  WARNING: val_pos={val_pos} < 30 — AUC will be noisy regardless of split.")
        print(f"{'='*60}")

        # ---- Per-fold descriptor normalization ----
        # Step A: restore every graph's raw (unnormalized) global_features.
        #         This undoes any prior fold's normalization so we always
        #         normalize from a clean baseline.
        for g, raw in zip(graphs, raw_global_features):
            g.global_features = raw.clone()
        # Step B: fit z-score stats on TRAINING graphs only (no val/test
        #         contamination — that would be data leakage).
        train_graphs_for_stats = [graphs[i] for i in train_idx]
        gf_mean, gf_std = fit_global_feature_stats(train_graphs_for_stats)
        # Step C: apply (raw - mean)/std to all graphs (train, val, test
        #         all use the SAME stats — fitted only on train).
        apply_global_feature_norm(graphs, raw_global_features, gf_mean, gf_std)
        # Step D: persist per-fold stats so inference.py / ensemble can apply
        #         the same normalization at test time.
        torch.save(
            {"mean": gf_mean, "std": gf_std},
            DATA_DIR / f"global_feature_stats_v5_{VARIANT}_fold{fold_i}.pt",
        )

        # ---- Build per-fold datasets and data loaders ----
        train_data = [graphs[i] for i in train_idx]
        val_data = [graphs[i] for i in val_idx]
        test_data = [graphs[i] for i in test_idx]

        loader_kwargs = {
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            # pin_memory only helps on CUDA (page-locked host memory enables
            # async device transfer). No-op or harmful on MPS/CPU.
            "pin_memory": (DEVICE.type == "cuda"),
        }
        if NUM_WORKERS > 0:
            # persistent_workers keeps DataLoader workers alive across epochs
            # — saves ~10s/epoch by avoiding re-fork. prefetch_factor=4 means
            # each worker has 4 batches queued ready, hiding any data-prep
            # latency behind the GPU's compute.
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 4

        # PyG's DataLoader handles batching of variable-sized graphs by
        # concatenating them into one big graph with a `batch` index that
        # tells the model which atom belongs to which molecule. This is
        # what makes the "B" dimension implicit but recoverable in pooling.
        train_loader = DataLoader(train_data, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_data, **loader_kwargs)
        test_loader = DataLoader(test_data, **loader_kwargs)

        # ---- Build model, optimizer, scheduler, AMP scaler ----
        model = HIVGNN(
            atom_dim=ATOM_FEATURE_DIM,
            edge_dim=BOND_FEATURE_DIM,
            global_dim=GLOBAL_FEATURE_DIM,
            hidden_dim=HIDDEN_DIM,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
        ).to(DEVICE)

        # AdamW = Adam with decoupled weight decay (correct L2 regularization).
        # Default Adam's weight_decay couples to the adaptive moment estimates
        # in a way that effectively scales it by the gradient — bad. AdamW
        # applies decay directly to weights, which is what you want.
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
        )
        # Cosine annealing with warm restarts: LR drops smoothly from LR to
        # eta_min over T_0=20 epochs, then jumps back up to LR and decays
        # over the next T_0*T_mult=40 epochs, etc. The "restart" jolts help
        # escape local minima — but only if MIN_EPOCHS lets training run long
        # enough to *see* a restart, which is why MIN_EPOCHS=30 covers
        # 1.5 cosine cycles.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=1e-6,
        )
        # GradScaler is part of the AMP path — manages fp16 loss scaling.
        scaler = torch.amp.GradScaler("cuda") if USE_AMP else None

        # ---- Training loop with early stopping ----
        best_val_auc = 0.0
        patience_counter = 0
        checkpoint_path = DATA_DIR / f"best_gnn_fold{fold_i}_v5_{VARIANT}.pth"

        for epoch in range(max_epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, scaler)
            scheduler.step()  # advance LR per epoch (not per step)
            val_auc = evaluate(model, val_loader)

            # MPS doesn't auto-free unused tensors aggressively → periodic
            # manual cleanup prevents OOM on long runs.
            if DEVICE.type == "mps" and (epoch + 1) % 5 == 0:
                torch.mps.empty_cache()

            current_lr = optimizer.param_groups[0]["lr"]

            # Save checkpoint only when val AUC improves. We never overwrite
            # with worse weights — at fold end we reload from `checkpoint_path`
            # to get the best-val model for test evaluation.
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                torch.save(model.state_dict(), checkpoint_path)
            else:
                patience_counter += 1

            # Print every 5 epochs, OR every time we hit a new best val AUC.
            # Keeps logs informative without being too noisy.
            if (epoch + 1) % 5 == 0 or patience_counter == 0:
                print(
                    f"  Epoch {epoch+1:3d} | loss={train_loss:.4f} | "
                    f"val_auc={val_auc:.4f} | best={best_val_auc:.4f} | "
                    f"lr={current_lr:.2e} | patience={patience_counter}/{PATIENCE}"
                )

            # Early stop: only fire after MIN_EPOCHS=30 floor. Without the
            # floor, fold 0's epoch-1 noise spike was getting locked in as
            # "best" and the model never recovered. The floor covers 1.5
            # cosine cycles so the model has had time to see a restart.
            if patience_counter >= PATIENCE and (epoch + 1) >= MIN_EPOCHS:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        # ---- Reload best checkpoint and evaluate on test set ----
        # Test AUC is the actual reported number. Val AUC was just used to
        # pick the best checkpoint and decide when to stop.
        if checkpoint_path.exists():
            model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))

        test_auc = evaluate(model, test_loader)
        fold_aucs.append(test_auc)
        print(f"  Fold {fold_i} Test AUC: {test_auc:.4f}")

        # Free GPU memory between folds — important on T4's 16GB or any MPS
        # device. Without this, accumulated cache + model can OOM by fold 3-4.
        del model, optimizer, scheduler, train_loader, val_loader, test_loader
        del train_data, val_data, test_data
        if DEVICE.type == "mps":
            torch.mps.empty_cache()
        elif DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # ---- Save a shared inference-time normalization ----
    # The per-fold stats files above are great if you're matching predictions
    # to a specific fold's checkpoint. But for general single-model inference
    # (someone runs inference.py on a new SMILES), we don't know which fold's
    # checkpoint they're using — so save one set of stats fitted on the FULL
    # dataset as a sensible default.
    for g, raw in zip(graphs, raw_global_features):
        g.global_features = raw.clone()
    shared_mean, shared_std = fit_global_feature_stats(graphs)
    torch.save(
        {"mean": shared_mean, "std": shared_std},
        DATA_DIR / f"global_feature_stats_v5_{VARIANT}.pt",
    )

    print(f"\n{'='*60}")
    print("Cross-Validation Results")
    print(f"{'='*60}")
    for i, auc in enumerate(fold_aucs):
        print(f"  Fold {i}: {auc:.4f}")
    print(f"  Mean AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")


if __name__ == "__main__":
    main()
