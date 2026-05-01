"""Fit a logistic-regression stacker on out-of-fold (OOF) validation predictions.

This is Tier 2 of the ensemble-combination story. Instead of the naive
0.5 * P_gnn + 0.5 * P_mf weighted average used by ensemble_inference.py's
default, we learn:

    P_final = sigmoid(w_gnn * P_gnn + w_mf * P_mf + b)

where (w_gnn, w_mf, b) are fitted by logistic regression on out-of-fold
validation predictions. This is strictly more flexible than weighted
averaging because:

  - It learns the *relative reliability* of each model (the typical finding
    is that the stronger model gets a heavier coefficient).
  - It learns a bias term that corrects systematic over/underconfidence
    induced by focal loss training (focal pushes confident predictions
    closer to 0/1 than they should be — a calibration error).
  - It implicitly handles probability scaling differences between the
    two models (GNN sigmoid outputs vs MolFormer sigmoid outputs are
    on different distributions even when both are trained with the same
    loss).

Out-of-fold protocol — IMPORTANT for honest results:
  For fold i, we only use the GNN-fold-i and MolFormer-fold-i checkpoints
  to predict on fold-i's *val* set. Those checkpoints were trained on
  train_i (which excludes val_i), so val_i predictions are out-of-fold.
  Concatenating all 5 fold's val predictions gives ~5k molecules of
  honest OOF data — enough to fit a 3-parameter logistic regression
  without overfitting.

  Crucially we DO NOT touch the test sets here. The fitted stacker is
  applied at inference (ensemble_inference.py --stacker PATH) and its
  test-set AUC is the only honest evaluation.

Output: a small .pt file with coef_gnn, coef_mf, intercept, plus the
best-fixed-weight from a grid search for comparison.

Usage:
  python src/fit_ensemble_stacker.py \\
      --gnn-pattern 'best_gnn_fold{i}_v5_desc.pth' \\
      --mf-pattern  'best_molformer_fold{i}.pth' \\
      --out         'ensemble_stacker.pt'
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score, roc_curve
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader

sys.path.insert(0, str(Path(__file__).parent))

from features import (
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    mol_to_graph,
)
from main import NUM_FOLDS, SEED, scaffold_kfold_split
from model import HIVGNN
from molformer_model import MolFormerClassifier, load_tokenizer
from molformer_train import SmilesDataset, make_collate, load_smiles_and_scaffolds
from ensemble_inference import DEVICE, SRC_DIR, _load_norm_stats
from tanimoto_features import build_active_fingerprints, tanimoto_features


def predict_gnn_fold(smiles_list, ckpt_path, stats, batch_size=64):
    """Run a single GNN fold checkpoint on a list of SMILES, return per-molecule probs.

    Returns:
        probs: array of shape (len(smiles_list),). NaN where SMILES failed RDKit.
    """
    graphs = []
    valid_idx = []
    for i, smi in enumerate(smiles_list):
        g = mol_to_graph(smi)
        if g is None:
            continue
        g.global_features = (g.global_features - stats["mean"]) / stats["std"]
        graphs.append(g)
        valid_idx.append(i)

    out = np.full(len(smiles_list), np.nan, dtype=np.float32)
    if not graphs:
        return out

    model = HIVGNN(
        atom_dim=ATOM_FEATURE_DIM,
        edge_dim=BOND_FEATURE_DIM,
        global_dim=GLOBAL_FEATURE_DIM,
    ).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    model.eval()

    loader = PyGDataLoader(graphs, batch_size=batch_size, shuffle=False)
    chunks = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            chunks.append(torch.sigmoid(model(batch)).cpu())
    probs = torch.cat(chunks).numpy()
    for j, orig in enumerate(valid_idx):
        out[orig] = probs[j]

    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return out


def predict_mf_fold(smiles_list, ckpt_path, tokenizer, batch_size=32):
    """Run a single MolFormer fold checkpoint on SMILES, return per-molecule probs.

    MolFormer is permissive about SMILES (tokenization rarely fails), so we
    don't track per-molecule errors here.
    """
    if not smiles_list:
        return np.zeros((0,), dtype=np.float32)

    dataset = SmilesDataset(smiles_list, [0.0] * len(smiles_list))
    loader = TorchDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_collate(tokenizer),
    )
    model = MolFormerClassifier().to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    model.eval()
    chunks = []
    with torch.no_grad():
        for enc, _ in loader:
            input_ids = enc["input_ids"].to(DEVICE)
            attn = enc["attention_mask"].to(DEVICE)
            chunks.append(torch.sigmoid(model(input_ids=input_ids, attention_mask=attn)).cpu())
    probs = torch.cat(chunks).numpy()
    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return probs


def main():
    parser = argparse.ArgumentParser(description="Fit OOF logistic-regression stacker.")
    parser.add_argument(
        "--gnn-pattern", default="best_gnn_fold{i}_v5_desc.pth",
        help="Filename pattern for GNN checkpoints. {i} is replaced by fold index.",
    )
    parser.add_argument(
        "--mf-pattern", default="best_molformer_fold{i}.pth",
        help="Filename pattern for MolFormer checkpoints.",
    )
    parser.add_argument("--out", default="ensemble_stacker.pt")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    smiles, scaffolds, labels = load_smiles_and_scaffolds()
    splits = scaffold_kfold_split(scaffolds, n_folds=NUM_FOLDS, seed=SEED)
    stats = _load_norm_stats()
    print("Loading MolFormer tokenizer...")
    tokenizer = load_tokenizer()

    oof_gnn, oof_mf, oof_tani, oof_labels = [], [], [], []

    for fold_i, (train_idx, val_idx, _) in enumerate(splits):
        gnn_ckpt = SRC_DIR / args.gnn_pattern.format(i=fold_i)
        mf_ckpt = SRC_DIR / args.mf_pattern.format(i=fold_i)
        if not gnn_ckpt.exists():
            print(f"Skipping fold {fold_i}: missing GNN checkpoint {gnn_ckpt.name}")
            continue
        if not mf_ckpt.exists():
            print(f"Skipping fold {fold_i}: missing MF checkpoint {mf_ckpt.name}")
            continue

        val_smiles = [smiles[i] for i in val_idx]
        val_labels = np.array([labels[i] for i in val_idx], dtype=np.float32)

        # Build the per-fold Tanimoto reference set: training actives only.
        # Using train_idx (not the full dataset) is what makes the Tanimoto
        # feature an *honest* OOF feature — leakage would inflate AUC.
        train_actives_smi = [smiles[j] for j in train_idx if int(labels[j]) == 1]
        train_active_fps = build_active_fingerprints(
            train_actives_smi, [1] * len(train_actives_smi)
        )

        print(f"\nFold {fold_i}: {len(val_smiles)} val molecules "
              f"(reference: {len(train_active_fps)} train actives)")
        gnn_probs = predict_gnn_fold(val_smiles, gnn_ckpt, stats)
        mf_probs = predict_mf_fold(val_smiles, mf_ckpt, tokenizer)
        tani_probs = tanimoto_features(val_smiles, train_active_fps)

        # Drop molecules where GNN failed (RDKit rejection); the stacker
        # only sees molecules where both models produced predictions.
        keep = ~np.isnan(gnn_probs)
        n_dropped = int((~keep).sum())
        gnn_probs = gnn_probs[keep]
        mf_probs = mf_probs[keep]
        tani_probs = tani_probs[keep]
        v_labels = val_labels[keep]

        gnn_auc = roc_auc_score(v_labels, gnn_probs) if len(np.unique(v_labels)) > 1 else float("nan")
        mf_auc = roc_auc_score(v_labels, mf_probs) if len(np.unique(v_labels)) > 1 else float("nan")
        tani_auc = roc_auc_score(v_labels, tani_probs) if len(np.unique(v_labels)) > 1 else float("nan")
        print(f"  Dropped {n_dropped} GNN-failed | "
              f"GNN val AUC={gnn_auc:.4f} | MF val AUC={mf_auc:.4f} | "
              f"Tanimoto val AUC={tani_auc:.4f}")

        oof_gnn.append(gnn_probs)
        oof_mf.append(mf_probs)
        oof_tani.append(tani_probs)
        oof_labels.append(v_labels)

    if not oof_gnn:
        sys.exit("No folds processed — check --gnn-pattern and --mf-pattern.")

    oof_gnn = np.concatenate(oof_gnn)
    oof_mf = np.concatenate(oof_mf)
    oof_tani = np.concatenate(oof_tani)
    oof_labels = np.concatenate(oof_labels)
    print(f"\nTotal OOF molecules: {len(oof_labels)} "
          f"(positives: {int(oof_labels.sum())}, "
          f"{100*oof_labels.mean():.2f}%)")

    # ---- Fit logistic-regression stacker on OOF predictions ----
    # Three-feature stacker: P_gnn, P_mf, max-Tanimoto-to-train-actives.
    X = np.stack([oof_gnn, oof_mf, oof_tani], axis=1)
    stacker = LogisticRegression(max_iter=1000)
    stacker.fit(X, oof_labels)

    coef_gnn = float(stacker.coef_[0, 0])
    coef_mf = float(stacker.coef_[0, 1])
    coef_tani = float(stacker.coef_[0, 2])
    intercept = float(stacker.intercept_[0])

    print("\n=== Stacker fit ===")
    print(
        f"  P_final = sigmoid({coef_gnn:.4f} * P_gnn + "
        f"{coef_mf:.4f} * P_mf + {coef_tani:.4f} * P_tani + {intercept:.4f})"
    )

    # Implied weight contributions (rough — assumes probs roughly comparable):
    coef_sum = coef_gnn + coef_mf + coef_tani
    if coef_sum > 0:
        print(f"  Implied weights — GNN: {coef_gnn/coef_sum:.3f}  "
              f"MF: {coef_mf/coef_sum:.3f}  Tanimoto: {coef_tani/coef_sum:.3f}")

    # ---- Compare against simpler baselines on the OOF set ----
    stacked_probs = stacker.predict_proba(X)[:, 1]
    auc_gnn_only = roc_auc_score(oof_labels, oof_gnn)
    auc_mf_only = roc_auc_score(oof_labels, oof_mf)
    auc_tani_only = roc_auc_score(oof_labels, oof_tani)
    auc_avg = roc_auc_score(oof_labels, 0.5 * oof_gnn + 0.5 * oof_mf)
    auc_stacker = roc_auc_score(oof_labels, stacked_probs)

    # Also fit a 2-feature stacker (no Tanimoto) for honest comparison —
    # tells us whether the Tanimoto feature actually helps.
    X2 = np.stack([oof_gnn, oof_mf], axis=1)
    stacker2 = LogisticRegression(max_iter=1000).fit(X2, oof_labels)
    auc_stacker_2feat = roc_auc_score(oof_labels, stacker2.predict_proba(X2)[:, 1])

    # Tier-1 grid search for the best fixed weight, for comparison.
    best_w, best_auc = 0.5, 0.0
    for w in np.arange(0.0, 1.005, 0.05):
        auc = roc_auc_score(oof_labels, w * oof_gnn + (1 - w) * oof_mf)
        if auc > best_auc:
            best_w, best_auc = float(w), float(auc)

    print("\n=== OOF AUC comparison ===")
    print(f"  GNN only:                  {auc_gnn_only:.4f}")
    print(f"  MolFormer only:            {auc_mf_only:.4f}")
    print(f"  Tanimoto-NN only:          {auc_tani_only:.4f}")
    print(f"  Naive 50/50 average:       {auc_avg:.4f}")
    print(f"  Best fixed weight ({best_w:.2f}):   {best_auc:.4f}")
    print(f"  Stacker (GNN+MF):          {auc_stacker_2feat:.4f}")
    print(f"  Stacker (GNN+MF+Tanimoto): {auc_stacker:.4f}  ← saved")

    # ---- Threshold tuning on OOF stacker probabilities ----
    # The stacker's intercept calibrates probabilities to the ~3.5% base rate,
    # so the natural 0.5 cutoff is way too high. We compute three principled
    # thresholds, each optimal under a different operating point:
    #
    #   1. F1-max — best balance of precision and recall. Default for general
    #      classification reports; what most papers cite.
    #   2. Youden's J — argmax(TPR - FPR). Maximizes "informedness" — useful
    #      for screening where you care about discriminating actives from
    #      inactives independent of class balance. Tends to give higher recall
    #      than F1-max.
    #   3. Base-rate calibration — pick the threshold s.t. the positive rate
    #      among predictions equals the positive rate in the data (~3.5%).
    #      Useful for virtual screening: you want roughly the same fraction
    #      of "hits" as the prior rate.
    base_rate = float(oof_labels.mean())

    # F1-max via grid search over a fine grid of stacker probabilities.
    f1_grid = np.linspace(0.001, 0.999, 999)
    f1_scores = [f1_score(oof_labels, stacked_probs >= t, zero_division=0) for t in f1_grid]
    f1_best_idx = int(np.argmax(f1_scores))
    threshold_f1 = float(f1_grid[f1_best_idx])
    f1_at_best = float(f1_scores[f1_best_idx])
    prec_f1, rec_f1, _, _ = precision_recall_fscore_support(
        oof_labels, stacked_probs >= threshold_f1, average="binary", zero_division=0
    )

    # Youden's J via the ROC curve — the threshold returned by roc_curve
    # corresponds exactly to a unique (FPR, TPR) point.
    fpr, tpr, roc_thr = roc_curve(oof_labels, stacked_probs)
    j_scores = tpr - fpr
    j_best_idx = int(np.argmax(j_scores))
    threshold_youden = float(roc_thr[j_best_idx])
    j_at_best = float(j_scores[j_best_idx])
    prec_y, rec_y, _, _ = precision_recall_fscore_support(
        oof_labels, stacked_probs >= threshold_youden, average="binary", zero_division=0
    )

    # Base-rate calibration: sort probs descending, take the top ~base_rate
    # fraction. The cutoff probability gives the threshold.
    sorted_probs = np.sort(stacked_probs)[::-1]
    n_positive = max(1, int(round(base_rate * len(stacked_probs))))
    threshold_baserate = float(sorted_probs[n_positive - 1])
    prec_b, rec_b, _, _ = precision_recall_fscore_support(
        oof_labels, stacked_probs >= threshold_baserate, average="binary", zero_division=0
    )

    print(f"\n=== Threshold tuning (on OOF predictions, n={len(oof_labels)}) ===")
    print(
        f"  Base-rate ({100*base_rate:.2f}%):   threshold={threshold_baserate:.4f}  "
        f"recall={rec_b:.3f}  precision={prec_b:.3f}"
    )
    print(
        f"  Best F1:                threshold={threshold_f1:.4f}  "
        f"F1={f1_at_best:.3f}  recall={rec_f1:.3f}  precision={prec_f1:.3f}"
    )
    print(
        f"  Best Youden's J:        threshold={threshold_youden:.4f}  "
        f"J={j_at_best:.3f}  recall={rec_y:.3f}  precision={prec_y:.3f}"
    )

    # ---- Save stacker (no sklearn dependency at inference) ----
    out_path = SRC_DIR / args.out
    torch.save({
        "coef_gnn": coef_gnn,
        "coef_mf": coef_mf,
        "coef_tanimoto": coef_tani,
        "intercept": intercept,
        "best_fixed_weight": best_w,
        "oof_auc_stacker": auc_stacker,
        "oof_auc_stacker_2feat": auc_stacker_2feat,
        "oof_auc_tanimoto_only": auc_tani_only,
        "oof_auc_best_fixed": best_auc,
        "oof_n_molecules": int(len(oof_labels)),
        "oof_base_rate": base_rate,
        "threshold_f1": threshold_f1,
        "threshold_youden": threshold_youden,
        "threshold_baserate": threshold_baserate,
        "f1_at_best": f1_at_best,
        "youden_j_at_best": j_at_best,
        # Raw OOF arrays — saved so make_figures.py can produce plots
        # without re-running per-fold inference (~30 min). Adds ~1MB to the
        # stacker .pt file but unlocks ROC/PR/calibration figures cheaply.
        "oof_gnn": oof_gnn.astype(np.float32),
        "oof_mf": oof_mf.astype(np.float32),
        "oof_tani": oof_tani.astype(np.float32),
        "oof_stacked": stacked_probs.astype(np.float32),
        "oof_labels": oof_labels.astype(np.int8),
    }, out_path)
    print(f"\nSaved stacker to {out_path}")


if __name__ == "__main__":
    main()
