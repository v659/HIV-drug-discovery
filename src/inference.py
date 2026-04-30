"""GNN-only inference for HIV bioactivity prediction.

This is the lightweight inference script — uses ONLY the GNN side of the
ensemble. For best accuracy at inference time, prefer ensemble_inference.py
which combines GNN + MolFormer predictions.

Two usage modes:
  1. Single model: --model-path src/best_gnn_fold0_v5_desc.pth
  2. Ensemble:     --ensemble-glob 'src/best_gnn_fold*_v5_desc.pth'
                   Loads all matching checkpoints, averages their sigmoid
                   probabilities per molecule (within-GNN ensemble).

Performance optimizations:
  - Module-level model cache so repeated calls don't reload weights
  - torch.compile() on CUDA for ~2x faster forward
  - Batch DataLoader instead of per-molecule predict
  - Single normalization-stats load reused across all calls

Errors are caught per-molecule (RDKit-invalid SMILES return an error dict
rather than crashing the whole inference call). The output preserves input
order and slot-by-slot maps to the input SMILES list.
"""
import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

import torch
from torch_geometric.loader import DataLoader

# Make sibling modules importable when running this file directly.
sys.path.insert(0, str(Path(__file__).parent))

from features import (
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    DESCRIPTOR_DIM,
    USE_MORGAN,
    mol_to_graph,
)

# "full" if Morgan FP is included in features, else "desc" (descriptors only).
# Must match the training-time setting — checkpoints from training with one
# variant won't load into a model built for the other.
_VARIANT = "full" if USE_MORGAN else "desc"
from model import HIVGNN

# Auto-detect best available compute. CUDA > MPS > CPU.
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
SRC_DIR = Path(__file__).parent

# Module-level caches. _MODEL_CACHE keys by (path, dims) tuple so the same
# physical checkpoint loaded with different model dim args wouldn't collide.
# _NORM_STATS is a single dict shared across all calls.
_MODEL_CACHE = {}
_NORM_STATS = None


def _load_norm_stats():
    """Load global feature normalization stats (z-score mean/std) from disk.

    Cached after first call — subsequent calls return the cached dict.
    Falls back to identity normalization (mean=0, std=1) if the stats file
    is missing, so the script works even on a fresh checkout without a
    completed training run.

    Note: this loads the SHARED stats (fitted on the full dataset). For
    fold-specific predictions, ensemble_inference.py uses fold-specific
    stats files instead.
    """
    global _NORM_STATS
    if _NORM_STATS is not None:
        return _NORM_STATS
    path = SRC_DIR / f"global_feature_stats_v5_{_VARIANT}.pt"
    if path.exists():
        _NORM_STATS = torch.load(path, map_location="cpu", weights_only=False)
    else:
        # Identity transform fallback. Predictions will still work but be
        # less accurate since training applied real normalization.
        _NORM_STATS = {
            "mean": torch.zeros(GLOBAL_FEATURE_DIM),
            "std": torch.ones(GLOBAL_FEATURE_DIM),
        }
    return _NORM_STATS


def _apply_norm(graph):
    """Apply z-score normalization to a graph's global_features in-place.

    Critical for inference accuracy: the model was trained on normalized
    features, so passing raw features at inference time would shift its
    decision boundary away from where it was trained.
    """
    stats = _load_norm_stats()
    graph.global_features = (graph.global_features - stats["mean"]) / stats["std"]
    return graph


def load_model(model_path, atom_dim=ATOM_FEATURE_DIM, edge_dim=BOND_FEATURE_DIM,
               global_dim=GLOBAL_FEATURE_DIM):
    """Load a HIVGNN checkpoint with caching and CUDA compile.

    The cache lets repeated inference calls share a single loaded-and-compiled
    model instance — important for serving scenarios where you call the
    inference function many times with the same checkpoint.

    `weights_only=True` is the safer torch.load mode introduced in PyTorch 2.x
    that prevents arbitrary code execution from a malicious checkpoint.

    `torch.compile` on CUDA gives ~2x forward speedup. Wrapped in try/except
    because not every CUDA driver supports it (older Triton, custom kernels,
    etc.). Falls back to eager execution silently.
    """
    cache_key = (model_path, atom_dim, edge_dim, global_dim)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    model = HIVGNN(
        atom_dim=atom_dim, edge_dim=edge_dim, global_dim=global_dim,
    ).to(DEVICE)
    model.load_state_dict(
        torch.load(model_path, map_location=DEVICE, weights_only=True)
    )
    model.eval()

    if DEVICE.type == "cuda":
        try:
            # "reduce-overhead" mode: minimal compilation latency, good for
            # repeated short inference calls (vs "max-autotune" which spends
            # time finding the absolute fastest kernel).
            model = torch.compile(model, mode="reduce-overhead")
        except Exception:
            pass

    _MODEL_CACHE[cache_key] = model
    return model


def format_prediction(smiles, probability, threshold=0.5):
    """Build a human-readable result dict for a single molecule.

    Returns:
        {
            smiles:                the input SMILES,
            prediction:            "Active" or "Inactive" (binary decision at threshold),
            probability_active:    P(active),
            probability_inactive:  1 - P(active),
            confidence:            max(P(active), P(inactive))  ∈ [0.5, 1.0]
        }

    `confidence` is "how far from the decision boundary"; useful for sorting
    predictions by certainty and flagging ambiguous calls (confidence ~0.5).
    """
    prediction = "Active" if probability >= threshold else "Inactive"
    confidence = probability if probability > 0.5 else (1 - probability)
    return {
        "smiles": smiles,
        "prediction": prediction,
        "probability_active": round(probability, 4),
        "probability_inactive": round(1 - probability, 4),
        "confidence": round(confidence, 4),
    }


def resolve_model_paths(model_path, ensemble_glob=None):
    """Decide which checkpoint(s) to load — single or glob-matched ensemble.

    If `ensemble_glob` is provided, expand it and return all matched paths
    (sorted for determinism). Otherwise return [model_path] as a singleton.
    """
    if ensemble_glob:
        paths = sorted(glob.glob(ensemble_glob))
        if not paths:
            raise FileNotFoundError(
                f"No checkpoints matched pattern: {ensemble_glob}"
            )
        return paths
    return [model_path]


def run_inference(smiles_list, model_path, threshold=0.5, ensemble_glob=None):
    """Run GNN inference on a list of SMILES strings, return per-molecule results.

    Pipeline:
      1. Convert each SMILES to a PyG Data graph; record per-molecule errors.
      2. Load the GNN model(s) — single or ensemble glob.
      3. Batch through DataLoader; for each batch, run all models, sigmoid,
         and average across the ensemble per molecule.
      4. Assemble results in input order, filling error slots with error dicts.

    Args:
        smiles_list:    list of SMILES strings.
        model_path:     path to a single .pth checkpoint (used if no glob).
        threshold:      Decision threshold for Active/Inactive labelling.
        ensemble_glob:  Optional glob pattern for multi-checkpoint ensemble.

    Returns:
        List of dicts (length == len(smiles_list)). Each is either a
        prediction dict (from format_prediction) or an error dict
        ({"error": ..., "smiles": ...}).
    """
    if not smiles_list:
        return []

    # Step 1: convert SMILES to graphs. Per-molecule errors are tracked in
    # `errors` (dict from input index → error dict) so they can be reinserted
    # into the output at the correct positions later.
    graphs = []
    graph_indices = []  # parallel to graphs; tracks original input index
    errors = {}
    for i, smi in enumerate(smiles_list):
        try:
            g = mol_to_graph(smi)
            if g is None:
                # RDKit couldn't parse — record error, skip this molecule.
                errors[i] = {"error": f"Invalid SMILES: {smi}", "smiles": smi}
                continue
            _apply_norm(g)  # apply training-time normalization to descriptors
            graphs.append(g)
            graph_indices.append(i)
        except Exception as e:
            # Catch-all for unexpected RDKit / descriptor failures.
            errors[i] = {"error": str(e), "smiles": smi}

    # Edge case: every input was invalid. Return error dicts for everything.
    if not graphs:
        return [errors.get(i, {"error": "Unknown", "smiles": s}) for i, s in enumerate(smiles_list)]

    # Step 2: load model(s). Returns user-friendly error message on common
    # failure modes (e.g. checkpoint trained with different feature dims).
    try:
        model_paths = resolve_model_paths(model_path, ensemble_glob)
        models = [load_model(p) for p in model_paths]
    except Exception as e:
        msg = str(e)
        if "size mismatch" in msg or "Missing key" in msg:
            msg += " | Model/features mismatch. Check checkpoint version."
        return [{"error": f"Failed to load model: {msg}", "smiles": smiles_list[0]}]

    # Step 3: batched inference. For each batch, run every model in the
    # ensemble and average their sigmoid outputs per molecule. This is
    # the within-GNN ensemble — see ensemble_inference.py for the full
    # GNN+MolFormer cross-model ensemble.
    loader = DataLoader(graphs, batch_size=64, shuffle=False)
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            batch_probs = []
            for model in models:
                logits = model(batch)
                batch_probs.append(torch.sigmoid(logits).cpu())
            # Stack along a new dim 0 → [n_models, batch_size], then mean
            # across that dim → [batch_size]. This is the within-GNN average.
            avg_probs = torch.stack(batch_probs).mean(dim=0)
            all_probs.extend(avg_probs.numpy().tolist())

    # Step 4: assemble results in original input order. Successful predictions
    # go in their original slot; error slots get the error dict from earlier.
    results = [None] * len(smiles_list)
    for j, orig_idx in enumerate(graph_indices):
        results[orig_idx] = format_prediction(
            smiles_list[orig_idx], all_probs[j], threshold
        )
    for idx, err in errors.items():
        results[idx] = err

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run HIV activity inference for SMILES strings."
    )
    parser.add_argument(
        "--model-path",
        default=str(SRC_DIR / f"best_gnn_fold0_v5_{_VARIANT}.pth"),
        help="Path to trained checkpoint.",
    )
    parser.add_argument(
        "--ensemble-glob",
        help="Glob pattern for ensemble (e.g., 'src/best_gnn_fold*_v5_desc.pth').",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Decision threshold.",
    )
    parser.add_argument(
        "--smiles", action="append",
        help="SMILES string. Repeat for multiple inputs.",
    )
    parser.add_argument(
        "--smiles-file",
        help="Path to text file with one SMILES per line.",
    )
    return parser.parse_args()


def load_smiles_inputs(args):
    values = []
    if args.smiles:
        values.extend(args.smiles)
    if args.smiles_file:
        with open(args.smiles_file, "r", encoding="utf-8") as f:
            values.extend([line.strip() for line in f if line.strip()])
    if not values:
        csv_path = SRC_DIR / "hiv.csv"
        df = pd.read_csv(csv_path)
        values = df[df["HIV_active"] == 1]["smiles"].head(100).tolist()
    return values


if __name__ == "__main__":
    args = parse_args()
    smiles_inputs = load_smiles_inputs(args)

    results = run_inference(
        smiles_inputs,
        model_path=args.model_path,
        threshold=args.threshold,
        ensemble_glob=args.ensemble_glob,
    )

    if args.ensemble_glob:
        matched = sorted(glob.glob(args.ensemble_glob))
        print(f"Model ensemble: {len(matched)} checkpoints from {args.ensemble_glob}")
    else:
        print(f"Model path: {args.model_path}")
    print(f"Threshold: {args.threshold:.2f}")
    print("-" * 60)

    for i, res in enumerate(results, start=1):
        if "error" in res:
            print(f"{i}. Error for {res.get('smiles', '<unknown>')}: {res['error']}")
        else:
            print(
                f"{i}. {res['prediction']} | P(active)={res['probability_active']:.4f} "
                f"| confidence={res['confidence']:.4f}"
            )
