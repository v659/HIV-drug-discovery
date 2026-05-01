"""Evaluate a saved GNN (HIVGNN v5b) fold checkpoint on its test split.

GNN counterpart to eval_fold.py (which is MolFormer-only). Use this to recover
a missing test AUC for any saved GNN fold checkpoint without retraining.

Reproduces the exact split used during training: scaffold_kfold_split is
deterministic given the seed and molecule list. Per-fold normalization stats
are loaded from the saved global_feature_stats_v5_{VARIANT}_fold{i}.pt files
that main.py writes during training, so test predictions match exactly.

Usage:
  python src/eval_gnn_fold.py --fold 0
  python src/eval_gnn_fold.py --fold 2 --split val      # eval on val instead
  python src/eval_gnn_fold.py --fold 0 --ckpt src/best_gnn_fold0_v5_desc.pth
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from features import (
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
)
from main import (
    DROPOUT,
    HIDDEN_DIM,
    NUM_FOLDS,
    NUM_HEADS,
    NUM_LAYERS,
    SEED,
    VARIANT,
    apply_global_feature_norm,
    evaluate,
    load_or_build_cache,
    scaffold_kfold_split,
)
from model import HIVGNN

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
SRC_DIR = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser(description="Evaluate a GNN fold checkpoint.")
    parser.add_argument("--fold", type=int, required=True, help="Fold index (0..4).")
    parser.add_argument(
        "--ckpt", default=None,
        help="Path to checkpoint. Default: src/best_gnn_fold{fold}_v5_{VARIANT}.pth",
    )
    parser.add_argument(
        "--split", choices=["test", "val"], default="test",
        help="Which split to evaluate on (default: test).",
    )
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()

    ckpt_path = (
        Path(args.ckpt) if args.ckpt
        else SRC_DIR / f"best_gnn_fold{args.fold}_v5_{VARIANT}.pth"
    )
    if not ckpt_path.exists():
        sys.exit(f"Checkpoint not found: {ckpt_path}")

    stats_path = SRC_DIR / f"global_feature_stats_v5_{VARIANT}_fold{args.fold}.pt"
    if not stats_path.exists():
        sys.exit(
            f"Per-fold normalization stats not found: {stats_path}\n"
            f"These are written by main.py during training; without them we "
            f"can't reproduce the exact fold-specific input distribution."
        )

    print(f"Device: {DEVICE}")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("Loading cached graphs + splits...")
    cache = load_or_build_cache()
    graphs = cache["graphs"]
    scaffolds = cache["scaffolds"]
    raw_global_features = [g.global_features.clone() for g in graphs]

    splits = scaffold_kfold_split(scaffolds, n_folds=NUM_FOLDS, seed=SEED)
    if args.fold < 0 or args.fold >= len(splits):
        sys.exit(f"Fold {args.fold} out of range (0..{len(splits)-1})")
    train_idx, val_idx, test_idx = splits[args.fold]
    eval_idx = test_idx if args.split == "test" else val_idx

    # Apply this fold's normalization to all graphs (the saved stats are
    # train-only stats from training time, so applying them now reproduces
    # the exact preprocessing the saved checkpoint expects).
    stats = torch.load(stats_path, map_location="cpu", weights_only=False)
    apply_global_feature_norm(graphs, raw_global_features, stats["mean"], stats["std"])

    eval_data = [graphs[i] for i in eval_idx]
    eval_pos = sum(int(g.y.item()) for g in eval_data)
    print(
        f"Fold {args.fold} {args.split} set: {len(eval_data)} molecules, "
        f"{eval_pos} active ({100*eval_pos/max(1,len(eval_data)):.2f}%)"
    )

    loader = DataLoader(eval_data, batch_size=args.batch, shuffle=False)

    print("Loading model + checkpoint...")
    model = HIVGNN(
        atom_dim=ATOM_FEATURE_DIM,
        edge_dim=BOND_FEATURE_DIM,
        global_dim=GLOBAL_FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    model.eval()

    print(f"Evaluating on {args.split} set ({len(eval_data)} molecules)...")
    auc = evaluate(model, loader)
    print(f"\nFold {args.fold} {args.split} AUC: {auc:.4f}")


if __name__ == "__main__":
    main()
