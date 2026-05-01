"""ECFP4 fingerprint + Tanimoto similarity utilities.

For each query molecule, we compute the maximum Tanimoto similarity to any
known active in a reference set. This becomes a 3rd feature for the
stacker, alongside P_gnn and P_mf.

Why this helps the ensemble:
  - It's a non-NN signal — uncorrelated errors with the GNN and MolFormer.
  - It's interpretable: "this molecule looks like active X (Tanimoto=0.83)".
  - It's strong on near-duplicates of known actives, which transformers
    sometimes miss when the SMILES tokenizes oddly.
  - Free to compute (no training, ~1s for 41k molecules on CPU).

Reference-set protocol — IMPORTANT for honest OOF features:
  For fold i, the reference set is the *training* actives of fold i only
  (the same actives the GNN-fold-i and MolFormer-fold-i checkpoints saw).
  This prevents leakage into the val_i Tanimoto feature.

  At final inference time on truly-unseen molecules, the reference set is
  all known actives in hiv.csv.
"""
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def smiles_to_ecfp4(smi, n_bits=2048):
    """Convert a SMILES to an ECFP4 (Morgan radius=2, 2048-bit) fingerprint.

    Returns None if RDKit can't parse the SMILES (so callers can default
    to a 0.0 similarity score gracefully).
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)


def max_tanimoto(query_fp, ref_fps, exclude_self_threshold=None):
    """Maximum Tanimoto similarity between a query fp and a list of reference fps.

    Returns 0.0 if query is None or ref_fps is empty — this lets the
    feature gracefully default to "not similar to anything" without
    needing special-case logic in callers.

    Args:
        exclude_self_threshold: If set (e.g., 0.999), any reference fp with
            similarity >= this value is dropped before taking the max. This
            prevents identity-leakage when the query molecule is itself
            present in the reference set (common when evaluating on training
            data: every active has Tanimoto=1.0 to itself in the active
            reference set, which inflates predictions trivially).
    """
    if query_fp is None or not ref_fps:
        return 0.0
    sims = DataStructs.BulkTanimotoSimilarity(query_fp, ref_fps)
    if exclude_self_threshold is not None:
        sims = [s for s in sims if s < exclude_self_threshold]
        if not sims:
            return 0.0
    return float(max(sims))


def build_active_fingerprints(smiles_list, labels):
    """Build the reference fingerprint set: ECFP4 for every labeled active.

    Skips molecules where RDKit fails. Caller can determine the kept count
    from len() of the returned list if it matters.
    """
    fps = []
    for smi, lbl in zip(smiles_list, labels):
        if int(lbl) != 1:
            continue
        fp = smiles_to_ecfp4(smi)
        if fp is not None:
            fps.append(fp)
    return fps


def tanimoto_features(query_smiles_list, ref_fps, exclude_self_threshold=None):
    """Vector of max-Tanimoto-to-reference for each query SMILES.

    Pass exclude_self_threshold=0.999 (or similar) when the query molecules
    might overlap the reference set, to prevent identity-leakage. Leave as
    None when you know the query is held-out from the reference (e.g., the
    OOF protocol in fit_ensemble_stacker.py).
    """
    import numpy as np
    out = np.zeros(len(query_smiles_list), dtype=np.float32)
    for i, smi in enumerate(query_smiles_list):
        out[i] = max_tanimoto(
            smiles_to_ecfp4(smi), ref_fps,
            exclude_self_threshold=exclude_self_threshold,
        )
    return out
