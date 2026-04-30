"""Evaluate a saved MolFormer fold checkpoint on its corresponding test split.

Use this to recover a missing test AUC after a training run was interrupted —
the best-val-AUC checkpoint was saved during training, so we can compute the
test AUC at any time without re-training.

Reproduces the exact split that was used during training: scaffold_kfold_split
is deterministic given the seed and the molecule list, so the test set for
fold-i is bit-for-bit identical to what train_fold() saw at training time.

Usage:
  python src/eval_fold.py --fold 2
  python src/eval_fold.py --fold 2 --ckpt src/best_molformer_fold2.pth
  python src/eval_fold.py --fold 2 --split val      # eval on val instead of test
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from main import NUM_FOLDS, SEED, scaffold_kfold_split
from molformer_model import MolFormerClassifier, load_tokenizer
from molformer_train import (
    SmilesDataset,
    make_collate,
    evaluate,
    load_smiles_and_scaffolds,
    NUM_WORKERS,
)


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
SRC_DIR = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser(description="Evaluate a MolFormer fold checkpoint.")
    parser.add_argument("--fold", type=int, required=True, help="Fold index (0..4).")
    parser.add_argument(
        "--ckpt", default=None,
        help="Path to checkpoint. Default: src/best_molformer_fold{fold}.pth",
    )
    parser.add_argument(
        "--split", choices=["test", "val"], default="test",
        help="Which split to evaluate on (default: test).",
    )
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt) if args.ckpt else SRC_DIR / f"best_molformer_fold{args.fold}.pth"
    if not ckpt_path.exists():
        sys.exit(f"Checkpoint not found: {ckpt_path}")

    print(f"Device: {DEVICE}")
    print(f"Loading splits...")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    smiles, scaffolds, labels = load_smiles_and_scaffolds()
    splits = scaffold_kfold_split(scaffolds, n_folds=NUM_FOLDS, seed=SEED)

    if args.fold < 0 or args.fold >= len(splits):
        sys.exit(f"Fold {args.fold} out of range (0..{len(splits)-1})")

    train_idx, val_idx, test_idx = splits[args.fold]
    eval_idx = test_idx if args.split == "test" else val_idx
    eval_pos = sum(int(labels[i]) for i in eval_idx)
    print(
        f"Fold {args.fold} {args.split} set: {len(eval_idx)} molecules, "
        f"{eval_pos} active ({100*eval_pos/max(1,len(eval_idx)):.2f}%)"
    )

    eval_data = SmilesDataset(
        [smiles[i] for i in eval_idx],
        [labels[i] for i in eval_idx],
    )

    print("Loading MolFormer tokenizer + backbone...")
    tokenizer = load_tokenizer()
    model = MolFormerClassifier().to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    model.eval()

    loader = DataLoader(
        eval_data,
        batch_size=args.batch,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        collate_fn=make_collate(tokenizer),
    )

    print(f"Evaluating on {args.split} set ({len(eval_data)} molecules)...")
    auc = evaluate(model, loader)
    print(f"\nFold {args.fold} {args.split} AUC: {auc:.4f}")


if __name__ == "__main__":
    main()
