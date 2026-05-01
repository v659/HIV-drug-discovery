"""Generate publication-quality figures for the paper.

Produces five PDFs in docs/figures/, plus prints bootstrap 95% confidence
intervals to stdout for inclusion in the paper text. PDFs are vector format
so they scale cleanly at any resolution.

Sources of data:
  - OOF arrays: read from src/ensemble_stacker.pt (saved by fit_ensemble_stacker.py).
  - Mixed-set: re-runs the ensemble on mixed_test.txt for held-out figures.
  - Per-fold test AUCs: hardcoded from the saved checkpoints' eval results
    (running 10 fold evaluations from scratch would take ~1 hour; the
    numbers are deterministic from the saved checkpoints anyway).

Usage:
  python src/make_figures.py
  python src/make_figures.py --skip-mixed   # faster, only OOF-based figures
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc as sk_auc,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

sys.path.insert(0, str(Path(__file__).parent))

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = Path(__file__).parent
FIG_DIR = PROJECT_ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Per-fold test AUCs (deterministic from saved checkpoints — see eval_fold.py
# and eval_gnn_fold.py to reproduce). Hardcoded here to keep this script fast.
# ---------------------------------------------------------------------------
GNN_FOLD_AUCS = [0.7955, 0.7676, 0.7780, 0.7392, 0.7851]
MF_FOLD_AUCS = [0.8182, 0.7955, 0.8097, 0.7798, 0.8255]

# Publication style — single-column figures sized for a 2-column paper.
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ---------------------------------------------------------------------------
# Bootstrap CI helpers
# ---------------------------------------------------------------------------
def bootstrap_auc_ci(y_true, y_pred, n_boot=2000, seed=42):
    """Bootstrap percentile 95% CI for ROC-AUC.

    Stratified resampling preserves class balance per draw — important for
    imbalanced data where unstratified bootstraps occasionally produce a
    sample with zero positives, breaking AUC computation.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)
    aucs = []
    for _ in range(n_boot):
        bp = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        bn = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([bp, bn])
        aucs.append(roc_auc_score(y_true[idx], y_pred[idx]))
    aucs = np.asarray(aucs)
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def paired_auc_bootstrap(y_true, p_a, p_b, n_boot=2000, seed=42):
    """Paired bootstrap: 95% CI for AUC(p_a) - AUC(p_b) and one-sided p-value.

    Resamples row indices (not predictions independently), so the same draw
    sees the same molecules under both predictors — the standard test for
    "is model A significantly better than model B on this dataset?".

    Returns (delta_observed, ci_lo, ci_hi, p_value_one_sided).
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    p_a = np.asarray(p_a)
    p_b = np.asarray(p_b)
    delta_obs = roc_auc_score(y_true, p_a) - roc_auc_score(y_true, p_b)

    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        bp = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        bn = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([bp, bn])
        deltas[i] = (
            roc_auc_score(y_true[idx], p_a[idx])
            - roc_auc_score(y_true[idx], p_b[idx])
        )
    lo = float(np.percentile(deltas, 2.5))
    hi = float(np.percentile(deltas, 97.5))
    # One-sided p-value: fraction of bootstrap draws where A is NOT better
    # than B. Small p ⇒ A is consistently better across resamples.
    p_value = float((deltas <= 0).mean())
    return float(delta_obs), lo, hi, p_value


def fold_mean_ci(values, n_boot=2000, seed=42):
    """Bootstrap CI for the mean of N fold-level AUCs.

    With only 5 folds this is undeniably small-sample, but it's the
    standard reporting practice for k-fold CV mean ± uncertainty.
    """
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, size=len(values), replace=True).mean()
             for _ in range(n_boot)]
    return (
        float(values.mean()),
        float(values.std(ddof=1)),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


# ---------------------------------------------------------------------------
# Figure 1 — per-fold AUC bar chart
# ---------------------------------------------------------------------------
def fig_fold_aucs():
    """Side-by-side bars for GNN vs MolFormer per-fold test AUC."""
    folds = np.arange(5)
    width = 0.38
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.bar(folds - width/2, GNN_FOLD_AUCS, width, label="GNN (v5b)", color="#5B9BD5")
    ax.bar(folds + width/2, MF_FOLD_AUCS, width, label="MolFormer-XL", color="#ED7D31")

    # Mean lines for visual reference.
    ax.axhline(np.mean(GNN_FOLD_AUCS), color="#5B9BD5", linestyle=":", alpha=0.6, lw=1)
    ax.axhline(np.mean(MF_FOLD_AUCS), color="#ED7D31", linestyle=":", alpha=0.6, lw=1)

    ax.set_xticks(folds)
    ax.set_xticklabels([f"Fold {i}" for i in folds])
    ax.set_ylabel("Scaffold-held-out test AUC")
    ax.set_ylim(0.70, 0.85)
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("Per-fold test AUC across scaffold splits")

    out = FIG_DIR / "fig1_fold_aucs.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Figure 2 — OOF ROC curves (GNN, MF, Tanimoto, Stacker)
# ---------------------------------------------------------------------------
def fig_oof_roc(stacker):
    """ROC curves for each component + the stacker, on OOF predictions."""
    y = np.asarray(stacker["oof_labels"]).astype(int)
    series = [
        ("Tanimoto-NN",    stacker["oof_tani"],    "#A5A5A5"),
        ("GNN (v5b)",      stacker["oof_gnn"],     "#5B9BD5"),
        ("MolFormer-XL",   stacker["oof_mf"],      "#ED7D31"),
        ("Stacker (final)", stacker["oof_stacked"], "#000000"),
    ]
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    for name, p, color in series:
        fpr, tpr, _ = roc_curve(y, p)
        a = roc_auc_score(y, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {a:.3f})",
                color=color, lw=1.6 if "Stacker" in name else 1.2)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"OOF ROC curves (n = {len(y):,})")
    ax.legend(loc="lower right", frameon=False)
    ax.set_aspect("equal")

    out = FIG_DIR / "fig2_oof_roc.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Figure 3 — OOF Precision-Recall curves
# ---------------------------------------------------------------------------
def fig_oof_pr(stacker):
    """PR curves are more informative than ROC for imbalanced data (~3.7% pos)."""
    y = np.asarray(stacker["oof_labels"]).astype(int)
    base_rate = y.mean()
    series = [
        ("Tanimoto-NN",    stacker["oof_tani"],    "#A5A5A5"),
        ("GNN (v5b)",      stacker["oof_gnn"],     "#5B9BD5"),
        ("MolFormer-XL",   stacker["oof_mf"],      "#ED7D31"),
        ("Stacker (final)", stacker["oof_stacked"], "#000000"),
    ]
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    for name, p, color in series:
        prec, rec, _ = precision_recall_curve(y, p)
        ap = sk_auc(rec, prec)
        ax.plot(rec, prec, label=f"{name} (AP = {ap:.3f})",
                color=color, lw=1.6 if "Stacker" in name else 1.2)
    ax.axhline(base_rate, color="k", linestyle="--", lw=0.8, alpha=0.5,
               label=f"Random ({base_rate:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"OOF Precision-Recall (n = {len(y):,}, {100*base_rate:.1f}% pos)")
    ax.legend(loc="upper right", frameon=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    out = FIG_DIR / "fig3_oof_pr.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Figure 4 — Reliability diagram (calibration plot)
# ---------------------------------------------------------------------------
def fig_calibration(stacker):
    """How well do predicted probabilities match empirical positive rates?"""
    y = np.asarray(stacker["oof_labels"]).astype(int)
    p_stacker = stacker["oof_stacked"]
    p_mf = stacker["oof_mf"]

    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    # Quantile-binning handles imbalance better than uniform binning at this
    # base rate (most uniform bins above 0.3 are nearly empty).
    for name, p, color in [
        ("MolFormer-XL (raw)", p_mf, "#ED7D31"),
        ("Stacker (calibrated)", p_stacker, "#000000"),
    ]:
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, "o-", label=name, color=color, lw=1.4, ms=4)

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Empirical positive rate")
    ax.set_title("Reliability diagram (OOF, quantile-binned)")
    ax.legend(loc="upper left", frameon=False)
    # Zoom to the populated region: with 3.7% positives, OOF predictions
    # rarely exceed ~0.5, so the upper-right of the unit square is empty.
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 0.3)
    ax.set_aspect("equal")

    out = FIG_DIR / "fig4_calibration.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Figure 5 — Threshold sweep on OOF
# ---------------------------------------------------------------------------
def fig_threshold_sweep(stacker):
    """Recall, precision, F1 vs threshold — visualizes the operating-point trade-off."""
    y = np.asarray(stacker["oof_labels"]).astype(int)
    p = stacker["oof_stacked"]
    grid = np.linspace(0.001, 0.999, 200)

    recalls, precs, f1s = [], [], []
    for t in grid:
        pred = (p >= t).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y, pred, average="binary", zero_division=0
        )
        recalls.append(rec)
        precs.append(prec)
        f1s.append(f1)

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.plot(grid, recalls, label="Recall", color="#5B9BD5", lw=1.4)
    ax.plot(grid, precs, label="Precision", color="#ED7D31", lw=1.4)
    ax.plot(grid, f1s, label="F1", color="#000000", lw=1.6)

    # Mark the principled thresholds. Stagger label y-positions inside the
    # plot area so they don't collide with each other or with the title.
    threshold_marks = [
        ("Youden's J", stacker["threshold_youden"], "#70AD47", 0.93),
        ("Base-rate",  stacker["threshold_baserate"], "#C00000", 0.86),
        ("F1-max",     stacker["threshold_f1"], "#7030A0", 0.93),
    ]
    for label, t, color, y_frac in threshold_marks:
        ax.axvline(t, color=color, linestyle=":", lw=1, alpha=0.7)
        ax.text(t + 0.005, y_frac, label, color=color, fontsize=8,
                ha="left", va="center", rotation=0,
                transform=ax.get_xaxis_transform())

    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Metric value")
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="center right", frameon=False)
    ax.set_title("Threshold sweep on OOF predictions")

    out = FIG_DIR / "fig5_threshold_sweep.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Bootstrap CI report (printed text — paste into paper)
# ---------------------------------------------------------------------------
def report_bootstrap_cis(stacker):
    """Print 95% bootstrap CIs for all the headline numbers."""
    print("\n=== Bootstrap 95% confidence intervals ===\n")

    # 1. Per-fold mean (small-sample bootstrap of fold-level values).
    for name, vals in [("GNN v5b", GNN_FOLD_AUCS), ("MolFormer-XL", MF_FOLD_AUCS)]:
        mean, std, lo, hi = fold_mean_ci(vals)
        print(f"  {name:15s} 5-fold mean = {mean:.4f} ± {std:.4f}  "
              f"[95% CI: {lo:.4f}–{hi:.4f}]")

    # 2. OOF AUC bootstrap (large-sample, much tighter).
    y = np.asarray(stacker["oof_labels"]).astype(int)
    print()
    for name, key in [
        ("GNN-only OOF",         "oof_gnn"),
        ("MolFormer-only OOF",   "oof_mf"),
        ("Tanimoto-NN OOF",      "oof_tani"),
        ("Stacker OOF",          "oof_stacked"),
    ]:
        p = stacker[key]
        a = roc_auc_score(y, p)
        lo, hi = bootstrap_auc_ci(y, p)
        print(f"  {name:25s} AUC = {a:.4f}  [95% CI: {lo:.4f}–{hi:.4f}]")

    # 3. Paired bootstrap: Stacker vs MolFormer-only on same OOF rows.
    # Marginal CIs above are conservative; the paired test answers
    # "is the stacker significantly better than MF alone?" directly.
    print("\n  Paired bootstrap (same OOF rows, stratified resampling):")
    for name_a, key_a, name_b, key_b in [
        ("Stacker", "oof_stacked", "MolFormer-only", "oof_mf"),
        ("Stacker", "oof_stacked", "GNN-only",       "oof_gnn"),
        ("MolFormer-only", "oof_mf", "GNN-only",     "oof_gnn"),
        ("Tanimoto-NN", "oof_tani", "GNN-only",      "oof_gnn"),
    ]:
        d, lo, hi, pv = paired_auc_bootstrap(y, stacker[key_a], stacker[key_b])
        sig = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else "ns"
        print(f"    Δ({name_a} − {name_b}) = {d:+.4f}  "
              f"[95% CI: {lo:+.4f}, {hi:+.4f}]  p = {pv:.4f}  {sig}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stacker", default="src/ensemble_stacker.pt")
    args = ap.parse_args()

    stacker_path = PROJECT_ROOT / args.stacker
    if not stacker_path.exists():
        sys.exit(f"Stacker file not found: {stacker_path}\n"
                 f"Run: python src/fit_ensemble_stacker.py")
    stacker = torch.load(stacker_path, map_location="cpu", weights_only=False)

    if "oof_labels" not in stacker:
        sys.exit("Stacker file does not contain raw OOF arrays. "
                 "Re-run: python src/fit_ensemble_stacker.py "
                 "(after the make_figures.py changes were applied).")

    print(f"Loaded stacker: {stacker['oof_n_molecules']:,} OOF molecules\n")
    print("Generating figures...")
    fig_fold_aucs()
    fig_oof_roc(stacker)
    fig_oof_pr(stacker)
    fig_calibration(stacker)
    fig_threshold_sweep(stacker)

    report_bootstrap_cis(stacker)

    print(f"\nAll figures saved to {FIG_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
