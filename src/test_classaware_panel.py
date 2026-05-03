"""Apply the class-aware stacker to a panel of SMILES and report results.

Runs the full 5-fold ensemble (P_gnn, P_mf), computes the all-actives
Tanimoto + 5 per-class Tanimoto features, and applies the new stacker.
Prints a per-molecule breakdown showing which class-Tanimoto features
fired, so you can see *why* a molecule was called active or inactive.

Usage:
  .venv/bin/python src/test_classaware_panel.py \\
      --smiles-file hiv_drugs_panel.txt \\
      --threshold-mode youden
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(SRC_DIR))

from ensemble_inference import gnn_predict, molformer_predict
from tanimoto_features import (
    build_active_fingerprints,
    smiles_to_ecfp4,
    tanimoto_features,
)

CLASSES = ["nrti", "nnrti", "pi", "insti", "entry"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smiles-file", required=True)
    p.add_argument(
        "--stacker",
        default=str(SRC_DIR / "ensemble_stacker_classaware.pt"),
    )
    p.add_argument(
        "--threshold-mode",
        choices=["youden", "f1", "baserate"], default="youden",
    )
    p.add_argument("--tta", type=int, default=5)
    p.add_argument(
        "--labels-file", default=None,
        help="Optional file with one class label per line "
             "(matching SMILES order) for per-class recall reporting.",
    )
    args = p.parse_args()

    smiles_list = [s.strip() for s in open(args.smiles_file) if s.strip()]
    print(f"Loaded {len(smiles_list)} SMILES.")

    panel_classes = None
    if args.labels_file:
        panel_classes = [l.strip().lower() for l in open(args.labels_file)
                         if l.strip()]
        if len(panel_classes) != len(smiles_list):
            sys.exit(f"Labels file has {len(panel_classes)} entries, "
                     f"SMILES file has {len(smiles_list)}.")

    # ---- Load stacker ----
    stacker = torch.load(args.stacker, map_location="cpu", weights_only=False)
    threshold = float(stacker[f"threshold_{args.threshold_mode}"])
    print(f"Threshold ({args.threshold_mode}): {threshold:.4f}")
    print(f"Stacker OOF AUC: {stacker.get('oof_auc_classaware', '?'):.4f}")

    # ---- Run GNN + MF ensembles ----
    gnn_ckpts = sorted(glob.glob(str(SRC_DIR / "best_gnn_fold*_v5_desc.pth")))
    mf_ckpts = sorted(glob.glob(str(SRC_DIR / "best_molformer_fold*.pth")))
    if not gnn_ckpts or not mf_ckpts:
        sys.exit("Missing checkpoints; check src/ for best_gnn_fold*.pth and "
                 "best_molformer_fold*.pth.")
    print(f"\nGNN folds: {len(gnn_ckpts)} | MolFormer folds: {len(mf_ckpts)} "
          f"| TTA: {args.tta}")
    gnn_probs, gnn_errors = gnn_predict(smiles_list, gnn_ckpts)
    mf_probs, _ = molformer_predict(smiles_list, mf_ckpts, tta_n=args.tta)

    # ---- Tanimoto features ----
    df = pd.read_csv(SRC_DIR / "hiv.csv")
    actives_smi = df[df["HIV_active"] == 1]["smiles"].tolist()
    all_active_fps = build_active_fingerprints(actives_smi, [1] * len(actives_smi))
    tani_all = tanimoto_features(
        smiles_list, all_active_fps, exclude_self_threshold=0.999
    )

    class_tani = {}
    for cls in CLASSES:
        cls_smis = stacker["class_drug_smiles"][cls]
        cls_fps = [smiles_to_ecfp4(s) for s in cls_smis]
        cls_fps = [f for f in cls_fps if f is not None]
        class_tani[cls] = tanimoto_features(
            smiles_list, cls_fps, exclude_self_threshold=0.99
        )

    # ---- Apply stacker ----
    final = np.zeros(len(smiles_list))
    for i in range(len(smiles_list)):
        if i in gnn_errors or np.isnan(gnn_probs[i]):
            final[i] = mf_probs[i]
            continue
        logit = (
            stacker["coef_gnn"] * gnn_probs[i]
            + stacker["coef_mf"] * mf_probs[i]
            + stacker["coef_tanimoto"] * tani_all[i]
            + stacker["coef_tani_nrti"] * class_tani["nrti"][i]
            + stacker["coef_tani_nnrti"] * class_tani["nnrti"][i]
            + stacker["coef_tani_pi"] * class_tani["pi"][i]
            + stacker["coef_tani_insti"] * class_tani["insti"][i]
            + stacker["coef_tani_entry"] * class_tani["entry"][i]
            + stacker["intercept"]
        )
        final[i] = 1.0 / (1.0 + np.exp(-logit))

    # ---- Print results ----
    hdr = (f"{'#':>3} {'Pred':9s} {'P':>7s} {'gnn':>5s} {'mf':>5s} "
           f"{'Tall':>5s} {'NRTI':>5s} {'NNRT':>5s} {'PI':>5s} "
           f"{'INST':>5s} {'ENTR':>5s}")
    if panel_classes:
        hdr += f"  {'cls':6s}"
    print("\n" + hdr)
    print("-" * len(hdr))
    n_active = 0
    per_class = {}
    for i, smi in enumerate(smiles_list):
        if i in gnn_errors:
            print(f"{i+1:3d} ERROR")
            continue
        is_active = final[i] >= threshold
        if is_active:
            n_active += 1
        pred = "ACTIVE" if is_active else "inactive"
        line = (
            f"{i+1:3d} {pred:9s} {final[i]:7.4f} "
            f"{gnn_probs[i]:5.3f} {mf_probs[i]:5.3f} "
            f"{tani_all[i]:5.3f} {class_tani['nrti'][i]:5.3f} "
            f"{class_tani['nnrti'][i]:5.3f} {class_tani['pi'][i]:5.3f} "
            f"{class_tani['insti'][i]:5.3f} {class_tani['entry'][i]:5.3f}"
        )
        if panel_classes:
            cls = panel_classes[i]
            line += f"  {cls:6s}"
            per_class.setdefault(cls, []).append(is_active)
        print(line)
    print("-" * len(hdr))
    print(f"\n{n_active}/{len(smiles_list)} predicted active.")

    if per_class:
        print("\nPer-class recall:")
        for cls in sorted(per_class):
            calls = per_class[cls]
            n_total = len(calls)
            n_pos = sum(calls)
            print(f"  {cls:8s}: {n_pos}/{n_total} = {n_pos/n_total:.2f}")


if __name__ == "__main__":
    main()
