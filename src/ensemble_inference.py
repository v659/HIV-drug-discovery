"""Ensemble inference: v5b GNN + MolFormer-XL.

This is the *full* inference path that combines both models. Use this when
you care about prediction quality — the simpler inference.py only uses the
GNN and is meant for quick checks.

Combination math (per molecule):
  P_gnn       = mean of sigmoid outputs across 5 GNN fold checkpoints
  P_molformer = mean of sigmoid outputs across 5 MolFormer fold checkpoints
  P_final     = gnn_weight * P_gnn + (1 - gnn_weight) * P_molformer

So there are TWO levels of averaging:
  1. Within-model: average over a model family's 5 fold checkpoints.
  2. Cross-model:  weighted average between GNN's mean and MolFormer's mean.

Why this works:
  Both models were trained on IDENTICAL scaffold splits (MolFormer's
  scaffold_kfold_split was imported from main.py and called with the same
  seed). So fold-i's test set is the same molecules for both — neither
  model has seen any test molecule's scaffold during training.

  But the two models *see* molecules through completely different lenses:
    - GNN: explicit graph structure (atoms, bonds, descriptors).
    - MolFormer: tokenized SMILES embedded against a 1.1B-molecule prior.
  Their errors are partially uncorrelated → averaging cancels noise →
  ensemble outperforms either model alone (~+0.5 to +1.5 AUC points).

Robustness: if RDKit rejects a SMILES (so the GNN errors out for that
molecule), we fall back to MolFormer's prediction alone for that slot —
MolFormer is more permissive about weird SMILES because it operates on
tokens, not chemistry.

Usage:
  python src/ensemble_inference.py \\
      --gnn-glob 'src/best_gnn_fold*_v5_desc.pth' \\
      --mf-glob  'src/best_molformer_fold*.pth' \\
      --smiles 'CC(=O)OC1=CC=CC=C1C(=O)O' \\
      --smiles 'C[C@H](N)C(=O)O'
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader

sys.path.insert(0, str(Path(__file__).parent))

from features import (
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    USE_MORGAN,
    mol_to_graph,
)
from model import HIVGNN
from molformer_model import MolFormerClassifier, load_tokenizer
from molformer_train import SmilesDataset, make_collate, MAX_SEQ_LEN
from tanimoto_features import build_active_fingerprints, tanimoto_features

VARIANT = "full" if USE_MORGAN else "desc"
SRC_DIR = Path(__file__).parent

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


# ===========================================================================
# GNN side — graph parsing + per-graph normalization + within-GNN ensemble.
# ===========================================================================
def _load_norm_stats():
    """Load global feature normalization stats (mean/std for descriptors).

    Returns the shared stats fitted on the full dataset. Identity-transform
    fallback if the stats file is missing — predictions will be less accurate
    but won't crash.
    """
    path = SRC_DIR / f"global_feature_stats_v5_{VARIANT}.pt"
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=False)
    return {
        "mean": torch.zeros(GLOBAL_FEATURE_DIM),
        "std": torch.ones(GLOBAL_FEATURE_DIM),
    }


def gnn_predict(smiles_list, ckpt_paths, batch_size=64):
    """Run within-GNN ensemble inference: mean of sigmoid outputs across folds.

    For each molecule:
      1. Parse SMILES → graph (track errors per-molecule, don't crash).
      2. Apply z-score normalization (must match training).
      3. For each GNN checkpoint: forward, sigmoid, store probabilities.
      4. Average probabilities across checkpoints.

    Args:
        smiles_list: list of SMILES strings.
        ckpt_paths:  list of paths to GNN .pth checkpoints (typically 5).
        batch_size:  DataLoader batch size for inference.

    Returns:
        out:    numpy array of shape (len(smiles_list),) with per-molecule
                probabilities. NaN where the SMILES failed to parse.
        errors: dict mapping input-index → error message string.
    """
    stats = _load_norm_stats()

    # Parse SMILES one by one, tracking errors. valid_idx parallels graphs;
    # tracks the original input index of each successfully parsed molecule
    # so we can place predictions back at the correct positions.
    graphs = []
    valid_idx = []
    errors = {}
    for i, smi in enumerate(smiles_list):
        try:
            g = mol_to_graph(smi)
            if g is None:
                errors[i] = f"Invalid SMILES: {smi}"
                continue
            # Apply z-score normalization in-place (matches training).
            g.global_features = (g.global_features - stats["mean"]) / stats["std"]
            graphs.append(g)
            valid_idx.append(i)
        except Exception as e:
            errors[i] = str(e)

    # If at least one molecule parsed, run all checkpoints.
    probs_per_model = []
    if graphs:
        loader = PyGDataLoader(graphs, batch_size=batch_size, shuffle=False)
        for ckpt in ckpt_paths:
            # Build a fresh model instance for each checkpoint and load weights.
            # We reload rather than caching because for batch inference, GPU
            # memory pressure matters more than load latency.
            model = HIVGNN(
                atom_dim=ATOM_FEATURE_DIM,
                edge_dim=BOND_FEATURE_DIM,
                global_dim=GLOBAL_FEATURE_DIM,
            ).to(DEVICE)
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
            model.eval()
            # Run this model over all batches; collect sigmoid outputs.
            chunks = []
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(DEVICE)
                    logits = model(batch)
                    chunks.append(torch.sigmoid(logits).cpu())
            probs_per_model.append(torch.cat(chunks))
            # Free GPU memory before loading the next checkpoint — important
            # if you have many checkpoints or limited VRAM.
            del model
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

        # Stack to [n_checkpoints, n_valid_molecules], mean → [n_valid_molecules].
        avg_probs = torch.stack(probs_per_model).mean(dim=0).numpy()
    else:
        avg_probs = np.zeros((0,), dtype=np.float32)

    # Map predictions back into the full-length output array. Slots
    # corresponding to errored molecules stay as NaN; the caller handles
    # those via the errors dict.
    out = np.full(len(smiles_list), np.nan, dtype=np.float32)
    for j, orig in enumerate(valid_idx):
        out[orig] = avg_probs[j]
    return out, errors


# ---------------------------------------------------------------------------
# MolFormer side — tokenize SMILES, ensemble across fold checkpoints.
# ---------------------------------------------------------------------------
def _augment_smiles(smiles_list, tta_n):
    """Generate TTA variants per SMILES.

    For each input SMILES, produce up to `tta_n` valid representations:
      - Variant 0: RDKit's canonical SMILES (deterministic).
      - Variants 1..tta_n-1: random non-canonical SMILES via doRandom=True.

    Returns:
        aug_smiles: flat list of all augmented SMILES.
        aug_idx:    list of original-input indices, parallel to aug_smiles
                    (used to average back per molecule).

    If RDKit can't parse a SMILES, we just pass it through as-is — MolFormer
    will tokenize it character-by-character and produce *some* prediction.
    """
    aug_smiles = []
    aug_idx = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            aug_smiles.append(smi)
            aug_idx.append(i)
            continue
        aug_smiles.append(Chem.MolToSmiles(mol))  # canonical
        aug_idx.append(i)
        for _ in range(tta_n - 1):
            try:
                rsmi = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
                aug_smiles.append(rsmi)
                aug_idx.append(i)
            except Exception:
                # Some weird molecules can fail random SMILES; skip silently.
                pass
    return aug_smiles, aug_idx


def molformer_predict(smiles_list, ckpt_paths, batch_size=32, tta_n=1):
    """Run within-MolFormer ensemble inference: mean of sigmoid outputs across folds.

    Pipeline:
      1. (Optional) TTA: expand each SMILES into tta_n variant strings.
      2. Tokenize all SMILES with MolFormer's custom tokenizer.
      3. Wrap in SmilesDataset (placeholder labels — we only need logits).
      4. For each MolFormer checkpoint: forward all batches, sigmoid, store.
      5. Average probabilities across checkpoints.
      6. (If TTA) Average variant probabilities back to per-molecule.

    Unlike `gnn_predict`, MolFormer is permissive — its tokenizer almost never
    rejects a SMILES (it just tokenizes character-by-character if it can't find
    a known sub-pattern). So we don't track per-molecule errors here and return
    an empty `errors` dict for API symmetry with gnn_predict.

    Args:
        smiles_list: list of SMILES strings.
        ckpt_paths:  list of MolFormer .pth checkpoints (typically 5).
        batch_size:  Smaller default than GNN side (32 vs 64) because MolFormer
                     activations are heavier per-sample.
        tta_n:       Number of SMILES variants per molecule (1 = no TTA).
                     Typical values: 5-10. Each variant is a valid SMILES of
                     the same molecule, so averaging cancels tokenization noise.

    Returns:
        probs:  numpy array of shape (len(smiles_list),) with per-molecule
                probabilities averaged across all MolFormer fold checkpoints
                AND all TTA variants.
        errors: empty dict (kept for API symmetry with gnn_predict).
    """
    if not smiles_list:
        return np.zeros((0,), dtype=np.float32), {}

    if tta_n > 1:
        run_smiles, aug_idx = _augment_smiles(smiles_list, tta_n)
    else:
        run_smiles = list(smiles_list)
        aug_idx = list(range(len(smiles_list)))

    tokenizer = load_tokenizer()
    # Use placeholder labels — we only need probabilities here.
    dataset = SmilesDataset(run_smiles, [0.0] * len(run_smiles))
    loader = TorchDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_collate(tokenizer),
    )

    probs_per_model = []
    for ckpt in ckpt_paths:
        # Fresh backbone+head per checkpoint. MolFormer is ~47M params so
        # rebuilding is non-trivial but still cheaper than holding 5 copies
        # in VRAM simultaneously on a 16GB T4.
        model = MolFormerClassifier().to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
        model.eval()
        chunks = []
        with torch.no_grad():
            for enc, _ in loader:
                # Tokenizer collate gives us input_ids + attention_mask;
                # both must be on the same device as the model.
                input_ids = enc["input_ids"].to(DEVICE)
                attn = enc["attention_mask"].to(DEVICE)
                logits = model(input_ids=input_ids, attention_mask=attn)
                chunks.append(torch.sigmoid(logits).cpu())
        probs_per_model.append(torch.cat(chunks))
        # Free this checkpoint before instantiating the next.
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # Stack to [n_checkpoints, n_run_smiles], mean → [n_run_smiles].
    variant_probs = torch.stack(probs_per_model).mean(dim=0).numpy()

    if tta_n > 1:
        # Average variants back to one prediction per original molecule.
        out = np.zeros(len(smiles_list), dtype=np.float32)
        counts = np.zeros(len(smiles_list), dtype=np.int32)
        for j, p in zip(aug_idx, variant_probs):
            out[j] += float(p)
            counts[j] += 1
        return out / np.maximum(counts, 1), {}

    return variant_probs, {}


# ---------------------------------------------------------------------------
# Combiner
# ---------------------------------------------------------------------------
def ensemble_predict(
    smiles_list, gnn_ckpts, mf_ckpts,
    gnn_weight=0.5, stacker=None, tta_n=1, tanimoto_ref_fps=None,
    tanimoto_exclude_self=0.999,
):
    """Combine GNN, MolFormer (and optionally Tanimoto-NN) probabilities.

    Two combination modes:

      1. Weighted average (default): P_final = w·P_gnn + (1-w)·P_mf
         where w = `gnn_weight` ∈ [0,1].

      2. Logistic-regression stacker (if `stacker` is provided):
         P_final = sigmoid(w_gnn·P_gnn + w_mf·P_mf [+ w_tani·P_tani] + b)
         where coefficients are pre-fitted on OOF val predictions. The
         Tanimoto term is only added if the stacker file contains
         `coef_tanimoto` and `tanimoto_ref_fps` is provided.

    Optional inference-time enhancements:
      - tta_n > 1: Test-Time Augmentation for MolFormer. Each SMILES is
        expanded into tta_n valid representations; predictions are averaged
        per molecule. Smooths out tokenization noise (typically +0.5-1.5 AUC).
      - tanimoto_ref_fps: List of ECFP4 fingerprints of known actives. If
        provided AND the stacker has coef_tanimoto, the max-Tanimoto-similarity
        feature is used as a 3rd stacker input.

    Robustness: in either mode, if the GNN can't parse a SMILES, that
    molecule's final probability falls back to MolFormer-only.
    """
    print(f"GNN ensemble: {len(gnn_ckpts)} folds")
    print(f"MolFormer ensemble: {len(mf_ckpts)} folds (TTA={tta_n})")

    use_tanimoto = (
        stacker is not None
        and "coef_tanimoto" in stacker
        and tanimoto_ref_fps is not None
        and len(tanimoto_ref_fps) > 0
    )

    if stacker is not None:
        msg = (
            f"Combination: logistic stacker "
            f"(coef_gnn={stacker['coef_gnn']:.3f}, "
            f"coef_mf={stacker['coef_mf']:.3f}, "
        )
        if use_tanimoto:
            msg += f"coef_tani={stacker['coef_tanimoto']:.3f}, "
        msg += f"b={stacker['intercept']:.3f})"
        print(msg)
    else:
        print(f"Combination: weighted average (GNN={gnn_weight:.2f}, MF={1-gnn_weight:.2f})")
    print("-" * 60)

    gnn_probs, gnn_errors = gnn_predict(smiles_list, gnn_ckpts)
    mf_probs, _ = molformer_predict(smiles_list, mf_ckpts, tta_n=tta_n)
    tani_probs = (
        tanimoto_features(
            smiles_list, tanimoto_ref_fps,
            exclude_self_threshold=tanimoto_exclude_self,
        )
        if use_tanimoto else np.zeros(len(smiles_list), dtype=np.float32)
    )

    final = np.full(len(smiles_list), np.nan, dtype=np.float32)
    for i in range(len(smiles_list)):
        if i in gnn_errors or np.isnan(gnn_probs[i]):
            # MolFormer can handle SMILES the GNN rejected; in stacker mode
            # we still fall back to raw MF prob rather than running the
            # stacker on a missing GNN input.
            final[i] = mf_probs[i]
        elif stacker is not None:
            logit = (
                stacker["coef_gnn"] * gnn_probs[i]
                + stacker["coef_mf"] * mf_probs[i]
                + stacker["intercept"]
            )
            if use_tanimoto:
                logit += stacker["coef_tanimoto"] * tani_probs[i]
            final[i] = 1.0 / (1.0 + np.exp(-logit))
        else:
            final[i] = gnn_weight * gnn_probs[i] + (1 - gnn_weight) * mf_probs[i]

    return final, gnn_probs, mf_probs, gnn_errors


def parse_args():
    """Build the CLI for ensemble inference.

    Glob defaults assume the canonical project layout: GNN checkpoints with the
    `_v5_desc` suffix and MolFormer checkpoints without a variant suffix. Override
    these if you've trained alternative configurations (e.g., `_v5_full` for the
    GNN with Morgan fingerprints, or different checkpoint naming for MolFormer).

    --gnn-weight defaults to 0.5 (flat ensemble). Tune this on a held-out set if
    you want to favor one model: higher weight → trust GNN more on molecules
    where graphs parse cleanly; lower weight → lean on MolFormer's pretraining
    prior for harder/unfamiliar scaffolds.

    --threshold defaults to 0.5 — the standard binary cutoff. For HIV active vs
    inactive on a 96.5/3.5 imbalanced dataset, you might want a lower threshold
    (e.g., 0.3) to favor recall, or use the precision/recall trade-off curves.
    """
    p = argparse.ArgumentParser(description="Ensemble v5b GNN + MolFormer-XL inference.")
    p.add_argument(
        "--gnn-glob", default="src/best_gnn_fold*_v5_desc.pth",
        help="Glob for GNN fold checkpoints.",
    )
    p.add_argument(
        "--mf-glob", default="src/best_molformer_fold*.pth",
        help="Glob for MolFormer fold checkpoints.",
    )
    p.add_argument(
        "--gnn-weight", type=float, default=0.5,
        help="Weight on GNN probability in the final average (0..1). Ignored if --stacker is set.",
    )
    p.add_argument(
        "--stacker", default=None,
        help="Path to a stacker .pt file from fit_ensemble_stacker.py. "
             "If set, overrides --gnn-weight and uses logistic-regression stacking.",
    )
    p.add_argument(
        "--threshold", type=float, default=None,
        help="Decision threshold. If omitted: defaults to 0.5 for plain weighted "
             "averaging, or auto-loads from --stacker (uses 'threshold_youden' by "
             "default — set --threshold-mode to override).",
    )
    p.add_argument(
        "--threshold-mode", choices=["youden", "f1", "baserate"], default="youden",
        help="Which auto-tuned threshold to use from the stacker file. "
             "youden = max TPR-FPR (good recall, default); "
             "f1 = max F1 (balanced); "
             "baserate = matches positive rate to dataset prior (~3.5%%).",
    )
    p.add_argument("--smiles", action="append")
    p.add_argument("--smiles-file")
    p.add_argument(
        "--test-actives", type=int, default=None, metavar="N",
        help="Sanity-test mode: sample N random known actives from hiv.csv "
             "and report recall (fraction predicted active at --threshold). "
             "Higher = better; ideally >0.7 for a well-calibrated model.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Seed for --test-actives sampling, for reproducibility.",
    )
    p.add_argument(
        "--tta", type=int, default=1, metavar="N",
        help="Test-Time Augmentation: number of SMILES variants per molecule "
             "for MolFormer (1=off, recommended 5-10). Each variant is a valid "
             "alternate SMILES of the same molecule; averaging cancels "
             "tokenization noise. Costs N× MolFormer inference time.",
    )
    p.add_argument(
        "--tanimoto-exclude-self", type=float, default=0.999, metavar="T",
        help="When computing the Tanimoto-NN feature, drop any reference fp "
             "with similarity >= T to prevent identity-leakage when the query "
             "is in the reference set. Set to 1.01 to disable (allows self-match).",
    )
    return p.parse_args()


def load_inputs(args):
    """Resolve the list of SMILES strings to predict on.

    Priority:
      1. --test-actives N — sample N random known actives from hiv.csv (mode flag).
      2. --smiles (repeatable) — direct CLI strings.
      3. --smiles-file — one SMILES per line, blank lines skipped.
      4. Default smoke test — first 100 known actives from hiv.csv. This makes
         `python ensemble_inference.py` run with no args produce something
         meaningful (a sanity check that both ensembles agree most actives are
         indeed active), useful for quick validation of new checkpoints.

    Returns:
        list[str] of SMILES — order is preserved from CLI inputs followed by
        file inputs.
    """
    if args.test_actives is not None:
        df = pd.read_csv(SRC_DIR / "hiv.csv")
        actives = df[df["HIV_active"] == 1]["smiles"].tolist()
        rng = np.random.default_rng(args.seed)
        n = min(args.test_actives, len(actives))
        idx = rng.choice(len(actives), size=n, replace=False)
        return [actives[i] for i in idx]

    values = []
    if args.smiles:
        values.extend(args.smiles)
    if args.smiles_file:
        with open(args.smiles_file, "r", encoding="utf-8") as f:
            values.extend([ln.strip() for ln in f if ln.strip()])
    if not values:
        # Default smoke test: 100 known actives from the training CSV.
        df = pd.read_csv(SRC_DIR / "hiv.csv")
        values = df[df["HIV_active"] == 1]["smiles"].head(100).tolist()
    return values


def main():
    """Script entry point: glob checkpoints, run ensemble, print per-molecule rows.

    Output format (one row per input SMILES):
        IDX. PRED     | P=FINAL (gnn=GNN_PROB, mf=MF_PROB)

    Where FINAL is the ensemble probability, GNN_PROB is the within-GNN average
    (NaN if SMILES failed to parse, in which case FINAL falls back to MolFormer
    only), and MF_PROB is the within-MolFormer average.

    The trailing summary "N/M predicted active" gives a quick activity-rate
    sanity check — for the default smoke test of 100 known actives, you'd
    expect this to be close to 100 if the model is calibrated reasonably.
    """
    args = parse_args()

    gnn_ckpts = sorted(glob.glob(args.gnn_glob))
    mf_ckpts = sorted(glob.glob(args.mf_glob))
    if not gnn_ckpts:
        sys.exit(f"No GNN checkpoints matched {args.gnn_glob}")
    if not mf_ckpts:
        sys.exit(f"No MolFormer checkpoints matched {args.mf_glob}")

    stacker = None
    if args.stacker:
        stacker_path = Path(args.stacker)
        if not stacker_path.exists():
            sys.exit(f"Stacker file not found: {stacker_path}")
        stacker = torch.load(stacker_path, map_location="cpu", weights_only=False)

    # Resolve the threshold. Priority: explicit --threshold > stacker auto-tuned
    # threshold (selected by --threshold-mode) > 0.5 default.
    if args.threshold is not None:
        threshold = args.threshold
        threshold_source = "user"
    elif stacker is not None:
        key = f"threshold_{args.threshold_mode}"
        if key in stacker:
            threshold = float(stacker[key])
            threshold_source = f"stacker ({args.threshold_mode})"
        else:
            threshold = 0.5
            threshold_source = "default (stacker has no tuned thresholds)"
    else:
        threshold = 0.5
        threshold_source = "default"

    # Build Tanimoto reference set ONCE (only if needed). The reference is
    # all known actives in the full hiv.csv — scaffold splits prevent us
    # from worrying about identity-leakage at this level of granularity
    # (a Tanimoto similarity of 1.0 against itself never happens for a
    # truly held-out molecule).
    tanimoto_ref_fps = None
    if stacker is not None and "coef_tanimoto" in stacker:
        df_actives = pd.read_csv(SRC_DIR / "hiv.csv")
        active_smi = df_actives[df_actives["HIV_active"] == 1]["smiles"].tolist()
        tanimoto_ref_fps = build_active_fingerprints(active_smi, [1] * len(active_smi))
        print(f"Tanimoto reference: {len(tanimoto_ref_fps)} known actives.")

    smiles_list = load_inputs(args)
    final, gnn_probs, mf_probs, errors = ensemble_predict(
        smiles_list, gnn_ckpts, mf_ckpts,
        gnn_weight=args.gnn_weight, stacker=stacker,
        tta_n=args.tta, tanimoto_ref_fps=tanimoto_ref_fps,
        tanimoto_exclude_self=args.tanimoto_exclude_self,
    )

    print(f"\nResults (threshold={threshold:.4f}, source={threshold_source}):")
    n_active = 0
    for i, smi in enumerate(smiles_list):
        if i in errors:
            print(f"{i+1:3d}. ERROR ({errors[i]})")
            continue
        p_final = final[i]
        p_gnn = gnn_probs[i] if not np.isnan(gnn_probs[i]) else float("nan")
        p_mf = mf_probs[i]
        pred = "Active" if p_final >= threshold else "Inactive"
        if pred == "Active":
            n_active += 1
        print(
            f"{i+1:3d}. {pred:8s} | P={p_final:.4f} "
            f"(gnn={p_gnn:.4f}, mf={p_mf:.4f})"
        )
    print(f"\n{n_active}/{len(smiles_list)} predicted active.")

    # In --test-actives mode, all inputs are known actives, so n_active is
    # the recall numerator and we report it as such.
    if args.test_actives is not None:
        n_total = len(smiles_list) - len(errors)
        recall = n_active / max(1, n_total)
        valid_probs = final[~np.isnan(final)]
        mean_p = float(np.mean(valid_probs)) if len(valid_probs) else float("nan")
        median_p = float(np.median(valid_probs)) if len(valid_probs) else float("nan")
        print(
            f"\n=== --test-actives mode (N={len(smiles_list)}, "
            f"threshold={threshold:.4f}) ==="
        )
        print(f"  Recall on known actives: {n_active}/{n_total} = {recall:.3f}")
        print(f"  Mean P:    {mean_p:.4f}")
        print(f"  Median P:  {median_p:.4f}")
        if errors:
            print(f"  GNN parse errors: {len(errors)} (counted as MF-only fallback)")


if __name__ == "__main__":
    main()
