"""Fine-tune MolFormer-XL on HIV activity using v5b's exact scaffold splits.

This is Phase 3 of the project: after the GNN tuning surface (focal loss,
val_pool=3, MIN_EPOCHS=30, etc.) was tapped out at ~0.77 mean test AUC, the
remaining bottleneck was scaffold generalization. The lever for that is a
foundation model — `ibm/MoLFormer-XL-both-10pct`, a 47M-param transformer
pretrained on 1.1B ZINC+PubChem molecules.

Critical design decision: identical scaffold splits as v5b.
  - load_smiles_and_scaffolds() loads molecules + scaffolds from the v5 GNN
    cache so the source-of-truth molecule list is identical.
  - scaffold_kfold_split() is imported from main.py and called with the same
    seed and val_pool, so fold-i's train/val/test sets are bit-for-bit
    identical between the GNN and MolFormer.
  - Why this matters: without identical splits, ensembling at inference time
    (ensemble_inference.py) would be averaging models trained on different
    test sets — fundamentally invalid. Identical splits also enable fair
    per-fold comparisons ("MolFormer F0=0.8182 vs GNN F0=0.7813").

Defaults sized for free Colab T4 (16GB VRAM):
  - batch=16 with grad_accum=4 → effective batch=64 (matches GNN's effective
    batch via grad accumulation since each forward is heavier).
  - LR: 1e-5 backbone, 1e-3 head — standard transformer fine-tune recipe.
  - 15 max epochs with patience=5 — pretrained models converge fast.
  - Focal loss matching v5b (α=0.75, γ=2.0) — same loss surface for both
    models keeps their probability outputs roughly comparable, important
    for the simple averaging done in ensemble_inference.py.

AMP is DISABLED. MolFormer's RoPE attention overflows in fp16 producing
NaN gradients (verified empirically — first epoch with AMP=True printed
loss=nan and val_auc=0.467 stuck at random). Fp32 path is ~2x slower but
trains correctly.
"""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from features import get_scaffold
from main import (  # type: ignore  (reuse v6's split + cache logic)
    DATA_DIR,
    CSV_PATH,
    NUM_FOLDS,
    SEED,
    csv_hash,
    scaffold_kfold_split,
)
from molformer_model import MolFormerClassifier, load_tokenizer

# ---------------------------------------------------------------------------
# Hyperparameters — standard transformer fine-tune recipe, sized for T4.
# ---------------------------------------------------------------------------
BACKBONE_LR = 1e-5      # TINY LR on backbone. Higher destroys pretraining.
HEAD_LR = 1e-3          # 100x higher LR on the freshly-initialized head.
WEIGHT_DECAY = 0.01     # Same as v5b, applied to both groups via AdamW.
WARMUP_FRAC = 0.1       # Linear LR warmup over first 10% of total steps.
                        # Critical for transformers — early gradients are
                        # wild and a cold-start at full LR can destabilize.
MAX_EPOCHS = 15         # Plenty for fine-tuning — usually converges in 3-8.
PATIENCE = 5            # Lower than GNN's 20 because transformer convergence
                        # is faster + smoother. 5 is enough to confirm plateau.
MIN_EPOCHS = 3          # Don't stop in the warmup window.
GRAD_CLIP = 1.0         # Same as v5b. Standard transformer practice.
MAX_SEQ_LEN = 202       # MolFormer's max context; longest HIV SMILES (~150 chars)
                        # fits comfortably so this rarely truncates anything.

# Focal loss hyperparameters — IDENTICAL to v5b's. Critical for ensemble
# compatibility: both models output probabilities calibrated against the
# same loss surface, making the simple (P_gnn + P_mf)/2 average meaningful.
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.75

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# T4-friendly defaults; override via --batch / --grad-accum.
# AMP disabled: MolFormer's RoPE attention overflows in fp16 → NaN loss.
# Trade fp16 speedup for stability; smaller batch + grad accum keeps eff. batch=64.
if DEVICE.type == "cuda":
    BATCH_SIZE = 16
    GRAD_ACCUM = 4  # effective batch = 64
    USE_AMP = False
    NUM_WORKERS = 2
else:
    BATCH_SIZE = 8
    GRAD_ACCUM = 1
    USE_AMP = False
    NUM_WORKERS = 0

CACHE_TAG = "molformer_xl"


# ---------------------------------------------------------------------------
# Data loading — primary path: reuse the v5 GNN cache so MolFormer trains
# on identical molecules + scaffolds, enabling fair ensembling. Fallback
# path rebuilds from CSV if the cache isn't present.
# ---------------------------------------------------------------------------
def load_smiles_and_scaffolds():
    """Load molecule list, scaffolds, and labels for MolFormer training.

    Tries two paths in order:

      1. Preferred: load from the v5 GNN's preprocessed cache. This guarantees:
         - Same molecule set (some molecules were skipped by RDKit; we want
           the GNN's exact post-skip list, not a fresh-from-CSV count).
         - Same scaffold values (computed by the same RDKit version under the
           same `includeChirality=False` setting).
         - Same canonicalized SMILES (RDKit's canonical SMILES output is
           deterministic but version-dependent).

      2. Fallback: read the CSV, recompute scaffolds. Slower (~30s) and may
         produce slightly different molecule counts than the GNN saw if RDKit
         versions differ between when the cache was built and now.

    Returns three parallel lists: smiles, scaffolds, labels.

    The cache lookup also verifies the CSV hash matches — if the source CSV
    has been modified since the cache was built, we fall through to rebuild
    rather than serving stale data.
    """
    # Try both descriptor-only and full (descriptors + Morgan) caches.
    for variant in ("desc", "full"):
        path = DATA_DIR / f"hiv_preprocessed_cache_v5_{variant}.pt"
        if path.exists():
            cache = torch.load(path, weights_only=False)
            meta = cache.get("meta", {})
            # Hash check guards against stale cache from old CSV.
            if meta.get("csv_hash") == csv_hash(CSV_PATH):
                smiles = list(cache["smiles"])
                scaffolds = list(cache["scaffolds"])
                # Labels live inside each Data object's `y` tensor.
                labels = [float(g.y.item()) for g in cache["graphs"]]
                print(f"Loaded splits source from v5 GNN cache ({variant}): "
                      f"{len(smiles)} molecules")
                return smiles, scaffolds, labels

    # Fallback: rebuild from CSV. Note that this may produce slightly
    # different splits if RDKit's behavior has changed since the cache
    # was built. For strict reproducibility, ship the cache file alongside
    # the code.
    print("v5 cache not found — building from CSV.")
    df = pd.read_csv(CSV_PATH).dropna(subset=["smiles", "HIV_active"])
    smiles = df["smiles"].tolist()
    labels = df["HIV_active"].astype(float).tolist()
    scaffolds = [get_scaffold(s) for s in tqdm(smiles, desc="Scaffolds")]
    return smiles, scaffolds, labels


# ---------------------------------------------------------------------------
# Dataset / collate — minimal wrappers since MolFormer takes raw text.
# ---------------------------------------------------------------------------
class SmilesDataset(Dataset):
    """A trivial dataset of (SMILES_string, label_float) pairs.

    No preprocessing happens here — tokenization is deferred to the collate
    function so it can pad each batch to that batch's longest sequence
    (saves compute on shorter batches vs padding everything to MAX_SEQ_LEN).
    """

    def __init__(self, smiles, labels):
        self.smiles = smiles
        self.labels = labels

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, i):
        return self.smiles[i], self.labels[i]


def make_collate(tokenizer):
    """Build a batch-collate function bound to a specific tokenizer.

    Returned function takes a list of (smiles, label) pairs and produces:
      enc:    BatchEncoding with input_ids and attention_mask, both [B, T]
              where T is the longest tokenized sequence in this batch.
      labels: [B] float tensor.

    Padding strategy: per-batch dynamic padding (`padding=True`). Each batch
    is padded to its own max sequence length, not to the global MAX_SEQ_LEN.
    This typically saves 2-5x compute on batches with short molecules.
    Truncation kicks in only on the rare molecule that exceeds MAX_SEQ_LEN
    (very long SMILES — most HIV molecules tokenize to <100 tokens).
    """
    def collate(batch):
        smiles, labels = zip(*batch)
        enc = tokenizer(
            list(smiles),
            padding=True,           # pad to longest in this batch
            truncation=True,        # safety net for unusually long SMILES
            max_length=MAX_SEQ_LEN,
            return_tensors="pt",
        )
        return enc, torch.tensor(labels, dtype=torch.float)
    return collate


# ---------------------------------------------------------------------------
# Loss / metrics
# ---------------------------------------------------------------------------
def focal_bce_loss(logits, targets, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA):
    # Force fp32 — focal loss underflows in fp16 (autocast) at init.
    logits = logits.float()
    targets = targets.float()
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1 - probs) * (1 - targets)
    # Clamp to keep (1 - p_t)**gamma and downstream ops finite.
    p_t = p_t.clamp(min=1e-7, max=1 - 1e-7)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    focal_weight = alpha_t * (1 - p_t).pow(gamma)
    return (focal_weight * bce).mean()


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    for enc, labels in loader:
        input_ids = enc["input_ids"].to(DEVICE)
        attn = enc["attention_mask"].to(DEVICE)
        logits = model(input_ids=input_ids, attention_mask=attn)
        all_probs.append(torch.sigmoid(logits).cpu())
        all_labels.append(labels)
    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    if len(np.unique(labels)) < 2:
        return 0.0
    return roc_auc_score(labels, probs)


# ---------------------------------------------------------------------------
# Learning rate schedule: linear warmup → cosine decay to zero.
#
# This is the standard fine-tuning schedule for HuggingFace transformers,
# different from the GNN's CosineAnnealingWarmRestarts. Differences:
#
#   GNN uses warm restarts (LR cycles back to max several times) because
#     stochastic restarts help small-from-scratch models escape local minima.
#
#   Transformers use ONE long monotonic cycle because the pretrained backbone
#     starts in an already-good region of parameter space — restarts risk
#     bouncing it out of that region.
#
#   Linear warmup is critical for transformers: at step 0, the head is
#     randomly initialized and gradients flowing back through the backbone
#     are noisy. Linearly ramping LR over the first 10% of steps lets the
#     head stabilize before the backbone starts moving meaningfully.
# ---------------------------------------------------------------------------
def make_scheduler(optimizer, total_steps, warmup_frac=WARMUP_FRAC):
    """Build a LambdaLR scheduler implementing warmup → cosine decay.

    Args:
        optimizer:   AdamW (or any) optimizer with one or more param groups.
                     The scheduler's lambda is applied as a multiplier to
                     EACH group's base LR — so backbone (1e-5) and head
                     (1e-3) both follow the same shape, just at different
                     scales.
        total_steps: Total optimizer steps across all training (epochs ×
                     steps_per_epoch / grad_accum). The schedule decays to
                     zero exactly at this step.
        warmup_frac: Fraction of total_steps to spend in linear warmup.
    """
    warmup_steps = max(1, int(total_steps * warmup_frac))

    def lr_lambda(step):
        # Phase 1: linear warmup from 0 to base_lr over `warmup_steps`.
        if step < warmup_steps:
            return step / warmup_steps
        # Phase 2: cosine decay from base_lr to 0 over the remaining steps.
        # progress goes 0 → 1; cos goes 1 → -1; output goes 1 → 0.
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Per-fold training loop. Mirrors main.py's structure but with transformer-
# specific touches (warmup scheduler, two-LR optimizer groups, gradient
# accumulation for effective batch sizing).
# ---------------------------------------------------------------------------
def train_fold(fold_i, train_data, val_data, test_data, tokenizer, args):
    """Train MolFormer for one fold, return test AUC.

    Uses gradient accumulation: effective batch size = args.batch * args.grad_accum.
    Each forward pass uses args.batch samples (memory bound by GPU VRAM), but
    we only call optimizer.step() every grad_accum forwards, so gradients
    accumulate for an "effective" batch of args.batch * args.grad_accum.
    Default 16 * 4 = 64, matching the GNN's effective batch.
    """
    collate = make_collate(tokenizer)
    loader_kwargs = {
        "batch_size": args.batch,
        "num_workers": NUM_WORKERS,
        "pin_memory": (DEVICE.type == "cuda"),
        "collate_fn": collate,
    }

    train_loader = DataLoader(train_data, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_data, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_data, shuffle=False, **loader_kwargs)

    model = MolFormerClassifier(
        gradient_checkpointing=args.grad_checkpoint,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.param_groups(BACKBONE_LR, HEAD_LR, WEIGHT_DECAY),
    )
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    scheduler = make_scheduler(optimizer, total_steps)
    scaler = torch.amp.GradScaler("cuda") if USE_AMP else None

    best_val_auc = 0.0
    patience_counter = 0
    checkpoint_path = DATA_DIR / f"best_molformer_fold{fold_i}.pth"
    has_checkpoint = False

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_count = 0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(
            train_loader,
            desc=f"  fold{fold_i} ep{epoch+1:02d}/{args.epochs}",
            leave=False,
        )
        for step, (enc, labels) in enumerate(pbar):
            input_ids = enc["input_ids"].to(DEVICE, non_blocking=True)
            attn = enc["attention_mask"].to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            if USE_AMP:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(input_ids=input_ids, attention_mask=attn)
                # Loss math runs in fp32 (outside autocast) — fp16 focal loss
                # underflows on near-zero p_t at init.
                loss = focal_bce_loss(logits.float(), labels.float()) / args.grad_accum
                scaler.scale(loss).backward()
            else:
                logits = model(input_ids=input_ids, attention_mask=attn)
                loss = focal_bce_loss(logits, labels) / args.grad_accum
                loss.backward()

            if (step + 1) % args.grad_accum == 0:
                if USE_AMP:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * labels.size(0) * args.grad_accum
            running_count += labels.size(0)

        train_loss = running_loss / max(1, running_count)
        val_auc = evaluate(model, val_loader)

        current_lr = optimizer.param_groups[0]["lr"]
        improved = val_auc > best_val_auc
        if improved:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            has_checkpoint = True
        else:
            patience_counter += 1

        marker = "*" if improved else " "
        print(
            f"  Epoch {epoch+1:2d}{marker} | loss={train_loss:.4f} | "
            f"val_auc={val_auc:.4f} | best={best_val_auc:.4f} | "
            f"backbone_lr={current_lr:.2e} | patience={patience_counter}/{PATIENCE}"
        )

        if patience_counter >= PATIENCE and (epoch + 1) >= MIN_EPOCHS:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Load best checkpoint, evaluate on test
    if has_checkpoint:
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    test_auc = evaluate(model, test_loader)
    print(f"  Fold {fold_i} Test AUC: {test_auc:.4f}")

    del model, optimizer, scheduler, train_loader, val_loader, test_loader
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE.type == "mps":
        torch.mps.empty_cache()

    return test_auc


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune MolFormer-XL on HIV (5-fold CV).")
    p.add_argument("--fold-limit", type=int, default=None)
    p.add_argument(
        "--start-fold", type=int, default=0,
        help="Skip folds before this index (for resuming after a crash).",
    )
    p.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    p.add_argument("--batch", type=int, default=BATCH_SIZE)
    p.add_argument("--grad-accum", type=int, default=GRAD_ACCUM)
    p.add_argument(
        "--grad-checkpoint", action="store_true",
        help="Enable gradient checkpointing (lower VRAM, ~25%% slower).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    smiles, scaffolds, labels = load_smiles_and_scaffolds()
    print(f"Device: {DEVICE} | batch={args.batch} (effective {args.batch*args.grad_accum}) | "
          f"amp={USE_AMP} | grad_ckpt={args.grad_checkpoint}")
    num_pos = int(sum(labels))
    print(f"Dataset: {len(smiles)} molecules, {num_pos} active "
          f"({100*num_pos/len(smiles):.2f}%)")

    splits = scaffold_kfold_split(scaffolds, n_folds=NUM_FOLDS, seed=SEED)
    if args.fold_limit is not None:
        splits = splits[: args.fold_limit]
        print(f"Fold limit: running {len(splits)} of {NUM_FOLDS} folds")

    # Preserve original fold indices, then optionally skip the first N
    # folds. This keeps fold-i's molecules identical to a fresh full run
    # (since scaffold_kfold_split is seeded), so checkpoints saved with
    # --start-fold remain compatible with the rest of the ensemble.
    indexed_splits = list(enumerate(splits))
    if args.start_fold > 0:
        skipped = indexed_splits[: args.start_fold]
        indexed_splits = indexed_splits[args.start_fold :]
        print(f"Resuming: skipping folds {[i for i, _ in skipped]}, "
              f"running folds {[i for i, _ in indexed_splits]}")

    print("Loading MolFormer tokenizer...")
    tokenizer = load_tokenizer()

    fold_aucs = []
    for fold_i, (train_idx, val_idx, test_idx) in indexed_splits:
        train_pos = sum(int(labels[i]) for i in train_idx)
        val_pos = sum(int(labels[i]) for i in val_idx)
        test_pos = sum(int(labels[i]) for i in test_idx)
        train_scaf = {scaffolds[i] for i in train_idx}
        val_scaf = {scaffolds[i] for i in val_idx}
        test_scaf = {scaffolds[i] for i in test_idx}
        leak_val = len(train_scaf & val_scaf)
        leak_test = len(train_scaf & test_scaf)

        print(f"\n{'='*60}")
        print(f"Fold {fold_i}/{NUM_FOLDS-1}")
        print(f"  train={len(train_idx)} (pos={train_pos}, "
              f"{100*train_pos/max(1,len(train_idx)):.2f}%) "
              f"val={len(val_idx)} (pos={val_pos}) "
              f"test={len(test_idx)} (pos={test_pos})")
        print(f"  scaffold leakage: train∩val={leak_val} train∩test={leak_test} (must be 0)")
        print(f"{'='*60}")

        train_data = SmilesDataset(
            [smiles[i] for i in train_idx],
            [labels[i] for i in train_idx],
        )
        val_data = SmilesDataset(
            [smiles[i] for i in val_idx],
            [labels[i] for i in val_idx],
        )
        test_data = SmilesDataset(
            [smiles[i] for i in test_idx],
            [labels[i] for i in test_idx],
        )

        test_auc = train_fold(fold_i, train_data, val_data, test_data, tokenizer, args)
        fold_aucs.append(test_auc)

    print(f"\n{'='*60}")
    print("Cross-Validation Results (MolFormer-XL fine-tuned)")
    print(f"{'='*60}")
    for i, auc in enumerate(fold_aucs):
        print(f"  Fold {i}: {auc:.4f}")
    print(f"  Mean AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")


if __name__ == "__main__":
    main()
