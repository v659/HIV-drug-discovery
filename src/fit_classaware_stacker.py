"""Refit the OOF stacker with class-aware Tanimoto features.

Adds 5 mechanism-class Tanimoto-NN features (NRTI, NNRTI, PI, INSTI, entry)
on top of the existing 3 features (P_gnn, P_mf, T_all_actives), giving 8
features total.

Reuses OOF predictions saved inside the existing ensemble_stacker.pt — no
GPU model inference is re-run. Only RDKit fingerprinting + sklearn LR fit.
Runs in ~2 minutes on CPU.

Honest-OOF protocol notes:
  - The existing stacker.pt already saved oof_gnn, oof_mf, oof_tani,
    oof_labels for the ~41k val molecules across 5 folds.
  - We re-derive the SMILES order from deterministic splits (same SEED, same
    scaffold_kfold_split) so the per-class Tanimoto features line up with
    the saved arrays.
  - Class drug references are EXTERNAL to the training set (FDA-approved
    drugs from DrugBank / PubChem). To prevent identity leakage when an
    OOF molecule IS one of the listed drugs, we use exclude_self_threshold
    = 0.99 (any reference fp with similarity >= 0.99 is dropped from the
    max).

Outputs: src/ensemble_stacker_classaware.pt
"""
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(SRC_DIR))

from features import mol_to_graph
from main import NUM_FOLDS, SEED, scaffold_kfold_split
from molformer_train import load_smiles_and_scaffolds
from tanimoto_features import smiles_to_ecfp4, tanimoto_features

CLASSES = ["nrti", "nnrti", "pi", "insti", "entry"]
DRUG_CLASS_DIR = SRC_DIR / "drug_classes"


def load_class_fps():
    out = {}
    for cls in CLASSES:
        path = DRUG_CLASS_DIR / f"{cls}.txt"
        if not path.exists():
            sys.exit(f"Missing {path}. Run build_drug_classes.py first.")
        smis = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
        fps = []
        for s in smis:
            fp = smiles_to_ecfp4(s)
            if fp is not None:
                fps.append(fp)
        print(f"  {cls:6s}: {len(fps)} reference fps from {len(smis)} SMILES")
        out[cls] = fps
    return out


def main():
    # ---- 1. Load saved OOF predictions ----
    old_path = SRC_DIR / "ensemble_stacker.pt"
    print(f"Loading existing stacker from {old_path}")
    old = torch.load(old_path, map_location="cpu", weights_only=False)
    if "oof_gnn" not in old:
        sys.exit("Old stacker has no saved OOF arrays. "
                 "Re-run fit_ensemble_stacker.py first.")
    oof_gnn = np.asarray(old["oof_gnn"]).astype(np.float32)
    oof_mf = np.asarray(old["oof_mf"]).astype(np.float32)
    oof_tani_all = np.asarray(old["oof_tani"]).astype(np.float32)
    oof_labels = np.asarray(old["oof_labels"]).astype(np.float32)
    print(f"  OOF molecules: {len(oof_labels)} "
          f"(positives: {int(oof_labels.sum())}, "
          f"{100*oof_labels.mean():.2f}%)")

    # ---- 2. Re-derive the OOF SMILES order ----
    print("\nRe-deriving OOF SMILES order from deterministic splits...")
    smiles, scaffolds, labels = load_smiles_and_scaffolds()
    splits = scaffold_kfold_split(scaffolds, n_folds=NUM_FOLDS, seed=SEED)
    oof_smiles = []
    for fold_i, (train_idx, val_idx, _) in enumerate(splits):
        kept = 0
        for i in val_idx:
            smi = smiles[i]
            # Mirror the GNN-fail filter from fit_ensemble_stacker.py
            if mol_to_graph(smi) is not None:
                oof_smiles.append(smi)
                kept += 1
        print(f"  Fold {fold_i}: {kept} GNN-parseable val molecules")
    if len(oof_smiles) != len(oof_gnn):
        sys.exit(f"SMILES count mismatch: derived {len(oof_smiles)}, "
                 f"saved oof_gnn has {len(oof_gnn)}. Did the splits change?")
    print(f"  Total: {len(oof_smiles)} (matches saved oof_gnn).")

    # ---- 3. Load class drug fingerprints ----
    print("\nLoading per-class drug reference fingerprints...")
    class_fps = load_class_fps()

    # ---- 4. Compute per-class Tanimoto features ----
    print("\nComputing per-class Tanimoto features (this may take ~1 min)...")
    class_feats = {}
    for cls, fps in class_fps.items():
        feat = tanimoto_features(oof_smiles, fps, exclude_self_threshold=0.99)
        class_feats[cls] = feat
        # Quick sanity: feature should fire on a few molecules but not all
        n_high = int((feat > 0.4).sum())
        print(f"  {cls:6s}: mean={feat.mean():.4f}  "
              f"max={feat.max():.4f}  "
              f"#(>0.4)={n_high}")

    # ---- 5. Build feature matrix ----
    feature_names = ["gnn", "mf", "tani_all"] + [f"tani_{c}" for c in CLASSES]
    X = np.stack(
        [oof_gnn, oof_mf, oof_tani_all] + [class_feats[c] for c in CLASSES],
        axis=1,
    )
    print(f"\nFeature matrix: {X.shape}")

    # Per-feature standalone AUC — diagnostic only, not used in fit.
    print("\nPer-feature standalone AUC (higher = more individually predictive):")
    for i, n in enumerate(feature_names):
        try:
            a = roc_auc_score(oof_labels, X[:, i])
        except Exception:
            a = float("nan")
        print(f"  {n:14s}: {a:.4f}")

    # ---- 6. Fit LR ----
    print("\nFitting logistic regression...")
    stacker = LogisticRegression(max_iter=2000)
    stacker.fit(X, oof_labels)

    coefs = {n: float(stacker.coef_[0, i]) for i, n in enumerate(feature_names)}
    intercept = float(stacker.intercept_[0])

    print("\n=== Class-aware stacker coefficients ===")
    for n in feature_names:
        bar = "█" * max(0, min(40, int(abs(coefs[n]) * 5)))
        sign = "+" if coefs[n] >= 0 else "-"
        print(f"  coef[{n:14s}] = {coefs[n]:+.4f}  {sign}{bar}")
    print(f"  intercept       = {intercept:+.4f}")

    stacked_probs = stacker.predict_proba(X)[:, 1]
    auc_new = roc_auc_score(oof_labels, stacked_probs)
    auc_old = float(old.get("oof_auc_stacker", float("nan")))
    print(f"\nOOF AUC, class-aware stacker:    {auc_new:.4f}")
    print(f"OOF AUC, old single-Tani stacker: {auc_old:.4f}")
    print(f"Delta:                            {auc_new - auc_old:+.4f}")

    # ---- 7. Threshold tuning ----
    base_rate = float(oof_labels.mean())

    f1_grid = np.linspace(0.001, 0.999, 999)
    f1_scores = [f1_score(oof_labels, stacked_probs >= t, zero_division=0)
                 for t in f1_grid]
    f1_best_idx = int(np.argmax(f1_scores))
    threshold_f1 = float(f1_grid[f1_best_idx])
    f1_at_best = float(f1_scores[f1_best_idx])

    fpr, tpr, roc_thr = roc_curve(oof_labels, stacked_probs)
    j_scores = tpr - fpr
    j_best_idx = int(np.argmax(j_scores))
    threshold_youden = float(roc_thr[j_best_idx])
    j_at_best = float(j_scores[j_best_idx])

    sorted_probs = np.sort(stacked_probs)[::-1]
    n_positive = max(1, int(round(base_rate * len(stacked_probs))))
    threshold_baserate = float(sorted_probs[n_positive - 1])

    print(f"\nThresholds:")
    print(f"  Youden's J: {threshold_youden:.4f} (J={j_at_best:.3f})")
    print(f"  F1-max:     {threshold_f1:.4f} (F1={f1_at_best:.3f})")
    print(f"  Base-rate:  {threshold_baserate:.4f} ({100*base_rate:.2f}%)")

    # ---- 8. Save ----
    out = SRC_DIR / "ensemble_stacker_classaware.pt"
    payload = {
        # Backward-compatible keys (so an old loader can use the basic stacker)
        "coef_gnn": coefs["gnn"],
        "coef_mf": coefs["mf"],
        "coef_tanimoto": coefs["tani_all"],
        "intercept": intercept,
        # New class-aware keys
        "coef_tani_nrti": coefs["tani_nrti"],
        "coef_tani_nnrti": coefs["tani_nnrti"],
        "coef_tani_pi": coefs["tani_pi"],
        "coef_tani_insti": coefs["tani_insti"],
        "coef_tani_entry": coefs["tani_entry"],
        # Class drug SMILES embedded for self-contained inference
        "class_drug_smiles": {
            cls: [ln.strip() for ln in (DRUG_CLASS_DIR / f"{cls}.txt")
                  .read_text().splitlines() if ln.strip()]
            for cls in CLASSES
        },
        "feature_names": feature_names,
        "threshold_f1": threshold_f1,
        "threshold_youden": threshold_youden,
        "threshold_baserate": threshold_baserate,
        "f1_at_best": f1_at_best,
        "youden_j_at_best": j_at_best,
        "oof_auc_classaware": auc_new,
        "oof_auc_old": auc_old,
        "oof_n_molecules": int(len(oof_labels)),
        "oof_base_rate": base_rate,
    }
    torch.save(payload, out)
    print(f"\nSaved class-aware stacker to {out}")


if __name__ == "__main__":
    main()
