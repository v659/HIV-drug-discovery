"""Tanimoto-based lookup for known FDA-approved HIV drugs.

The model was trained on MoleculeNet-HIV labels from the NCI AIDS Antiviral
Screen (1987-1999), which is dominated by nucleoside-analog actives. As a
result, the model has poor recall on modern mechanism classes (NNRTIs, PIs,
INSTIs, entry inhibitors). This module patches that gap at *inference time*
by checking each query against a curated list of FDA-approved HIV drugs:

  - If max-Tanimoto-similarity >= threshold (default 0.70), short-circuit
    the model and return a "known drug" verdict with the matched drug name.

This is honest as long as the verdict is labelled as a lookup hit, not a
model prediction. It's a hybrid retrieval+classifier system, common in
cheminformatics for known-actives recall.
"""
import json
from pathlib import Path

from rdkit import DataStructs

from tanimoto_features import smiles_to_ecfp4

DEFAULT_DRUGS_JSON = Path(__file__).parent / "drug_classes" / "all_drugs.json"


def load_known_drugs(path=DEFAULT_DRUGS_JSON):
    """Load name+class+ECFP4 fingerprint for every approved HIV drug.

    Returns:
        list of (name, class, fp) tuples. Drugs whose SMILES fail RDKit
        are silently dropped — the caller doesn't need to handle them.
    """
    if not path.exists():
        return []
    records = json.loads(path.read_text())
    out = []
    for r in records:
        fp = smiles_to_ecfp4(r["smiles"])
        if fp is not None:
            out.append((r["name"], r["class"], fp))
    return out


def lookup(query_smiles, known_drugs, threshold=0.70):
    """Find the closest approved HIV drug to a query SMILES.

    Args:
        query_smiles: SMILES of the molecule to check.
        known_drugs: list from load_known_drugs().
        threshold: Tanimoto cutoff. >= 0.85 is essentially a near-duplicate;
            0.70 catches close analogs (e.g., AZT vs other thymidine analogs);
            below ~0.55 the similarity isn't meaningful for medicinal chemistry.

    Returns:
        dict {name, class, similarity} for the best match if its similarity
        is >= threshold, else None.
    """
    query_fp = smiles_to_ecfp4(query_smiles)
    if query_fp is None or not known_drugs:
        return None
    fps = [fp for _, _, fp in known_drugs]
    sims = DataStructs.BulkTanimotoSimilarity(query_fp, fps)
    best_idx = int(max(range(len(sims)), key=lambda i: sims[i]))
    best_sim = float(sims[best_idx])
    if best_sim < threshold:
        return None
    name, cls, _ = known_drugs[best_idx]
    return {"name": name, "class": cls, "similarity": best_sim}
