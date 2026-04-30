"""Evaluate the ensemble on a labeled mixed-set file.

Companion to ensemble_inference.py for honest threshold analysis. Unlike
--test-actives mode (recall-only on known positives), this script measures
precision, recall, F1, and ROC-AUC across a sweep of thresholds, so you
can pick a threshold based on the precision/recall trade-off you actually
want for your application.

Input file format (no header):
  - First N_POS lines = active SMILES (label=1)
  - Remaining lines  = inactive SMILES (label=0)

Usage:
  python src/eval_mixed.py --file mixed_test.txt --n-active 100 \\
         --stacker src/ensemble_stacker.pt
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))

from ensemble_inference import ensemble_predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="Mixed-set SMILES file.")
    ap.add_argument("--n-active", type=int, required=True,
                    help="Number of actives at the top of the file.")
    ap.add_argument("--gnn-glob", default="src/best_gnn_fold*_v5_desc.pth")
    ap.add_argument("--mf-glob", default="src/best_molformer_fold*.pth")
    ap.add_argument("--stacker", default=None)
    ap.add_argument("--gnn-weight", type=float, default=0.5)
    args = ap.parse_args()

    with open(args.file) as f:
        smiles = [ln.strip() for ln in f if ln.strip()]
    labels = np.array(
        [1] * args.n_active + [0] * (len(smiles) - args.n_active),
        dtype=np.int32,
    )
    print(f"Loaded {len(smiles)} molecules: {labels.sum()} actives, "
          f"{(labels == 0).sum()} inactives "
          f"({100 * labels.mean():.2f}% positive rate)")

    gnn_ckpts = sorted(glob.glob(args.gnn_glob))
    mf_ckpts = sorted(glob.glob(args.mf_glob))
    if not gnn_ckpts or not mf_ckpts:
        sys.exit("No checkpoints found.")

    stacker = None
    if args.stacker:
        stacker = torch.load(args.stacker, map_location="cpu", weights_only=False)

    final, _, _, errors = ensemble_predict(
        smiles, gnn_ckpts, mf_ckpts,
        gnn_weight=args.gnn_weight, stacker=stacker,
    )

    # Drop any errored rows for clean evaluation.
    valid = ~np.isnan(final)
    y = labels[valid]
    p = final[valid]
    if errors:
        print(f"\nDropped {len(errors)} errored rows.")

    # AUC is threshold-independent — the headline number.
    auc = roc_auc_score(y, p)
    print(f"\n=== Ensemble AUC on mixed set: {auc:.4f} ===")

    # Sweep candidate thresholds. Include the stacker's own auto-tuned ones
    # if available, so you can verify they actually behave as advertised.
    thresholds = [0.02, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    if stacker is not None:
        for k in ("threshold_youden", "threshold_f1", "threshold_baserate"):
            if k in stacker:
                thresholds.append(float(stacker[k]))
    thresholds = sorted(set(round(t, 4) for t in thresholds))

    print("\n=== Threshold sweep ===")
    print(f"{'thresh':>8} | {'TP':>4} {'FP':>4} {'TN':>5} {'FN':>4} | "
          f"{'recall':>7} {'prec':>6} {'F1':>6} {'FPR':>6}")
    print("-" * 70)
    for t in thresholds:
        pred = (p >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        prec, rec, f1, _ = precision_recall_fscore_support(
            y, pred, average="binary", zero_division=0
        )
        fpr = fp / max(1, fp + tn)
        print(f"{t:8.4f} | {tp:4d} {fp:4d} {tn:5d} {fn:4d} | "
              f"{rec:7.3f} {prec:6.3f} {f1:6.3f} {fpr:6.3f}")


if __name__ == "__main__":
    main()
