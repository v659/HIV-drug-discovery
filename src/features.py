"""Feature extraction for HIV bioactivity classification.

This module is the bridge between SMILES strings (text) and tensors the GNN can
consume. For every molecule we produce three things:

  1. **Per-atom feature vectors** (28 dims each) — atom type, hybridization,
     formal charge, ring membership, chirality, etc. Stacked into `Data.x`
     with shape [num_atoms, 28].
  2. **Per-bond feature vectors** (8 dims each) — bond order, conjugation,
     ring membership, stereochemistry. Stacked into `Data.edge_attr` with
     shape [2*num_bonds, 8] (each bond appears twice — once per direction —
     because PyG GNNs work on directed graphs).
  3. **Whole-molecule "global" descriptors** (54 dims, optionally + 2048 Morgan
     bits) — molecular weight, logP, TPSA, ring counts, drug-likeness score, etc.
     Computed by RDKit. Gives the model a "summary view" of the molecule that
     doesn't depend on graph traversal succeeding.

The final `mol_to_graph()` function packages everything into a PyG `Data`
object that is ready to be batched and fed to the GNN.

Scaffold extraction (Murcko scaffolds) lives here too because both training
and inference need to compute scaffolds the same way.
"""
import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.data import Data

# RDKit emits noisy warnings when computing descriptors on weird molecules
# (radicals, unusual valences). They don't affect the actual numbers we get
# back — we either get a finite value or NaN, and `_safe_descriptor` handles
# both. Suppressing here so training logs stay readable.
RDLogger.DisableLog("rdApp.*")

# ---------------------------------------------------------------------------
# Categorical vocabularies. Each list defines a one-hot encoding scheme for a
# specific atom/bond property. The order is fixed so that feature dimension
# `i` always means the same thing across train/val/test/inference — if you
# reorder these, every saved checkpoint becomes invalid.
# ---------------------------------------------------------------------------

# 9 explicit element types found in HIV.csv plus "Other" as a catch-all for
# rare elements. Covers >99.9% of atoms in the dataset.
ATOM_TYPES = ["C", "N", "O", "S", "F", "P", "Cl", "Br", "I", "Other"]

# Standard hybridization states. SP3 is most common (saturated carbons),
# SP2 is for double bonds and aromatic rings, SP is for triple bonds.
# Anything else (SP3D, etc.) gets bucketed into a 4th "Other" dim by code below.
HYBRIDIZATION_TYPES = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
]

# Chirality tags (R/S configuration at stereocenters). HIV bioactivity can
# depend strongly on stereochemistry — many drugs have a single active
# enantiomer, so we encode this explicitly.
CHIRAL_TAGS = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]

# Bond order: single, double, triple, aromatic (Kekulized aromatic bonds get
# their own type rather than being treated as alternating double/single).
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

# Total feature dimensions — must match what `get_atom_features` /
# `get_bond_features` produce. The model's input projection layers
# (atom_proj, edge_proj in model.py) are sized against these constants,
# so any change here requires retraining from scratch.
#
# Atom: 10 type + 6 degree + 1 charge + 4 hybridization + 1 aromatic
#       + 1 H-count + 4 chirality + 1 in-ring = 28
ATOM_FEATURE_DIM = 28
# Bond: 4 bond type + 1 conjugated + 1 in-ring + 2 stereo = 8
BOND_FEATURE_DIM = 8

# ---------------------------------------------------------------------------
# Whole-molecule "global" descriptors.
#
# RDKit ships ~200 descriptors total. Not all are useful, and some are slow
# or numerically unstable. This curated subset gives the model a rich summary
# of physicochemical properties without bogging down preprocessing:
#
#   - Molecular weight family (MolWt, HeavyAtomMolWt, ExactMolWt) — size signals
#   - Charge descriptors (Max/Min PartialCharge) — electrostatic profile
#   - Lipophilicity (MolLogP) — membrane permeability proxy, key for drugs
#   - TPSA, LabuteASA — surface area / polar surface area
#   - Connectivity indices (Chi0..Chi4, Kappa1..3, BalabanJ, BertzCT) —
#     graph-theoretic complexity / topology
#   - Functional group counts (NumHAcceptors/Donors, NumRotatableBonds,
#     NumAromaticRings, etc.) — pharmacophore counts
#   - QED — composite drug-likeness score in [0,1]
#
# These complement the GNN's local message passing by giving it whole-molecule
# context that doesn't depend on graph traversal. The model can use them as a
# "summary view" alongside the learned graph representation.
# ---------------------------------------------------------------------------
_DESCRIPTOR_FUNCS = [
    ("MolWt", Descriptors.MolWt),
    ("HeavyAtomMolWt", Descriptors.HeavyAtomMolWt),
    ("ExactMolWt", Descriptors.ExactMolWt),
    ("NumValenceElectrons", Descriptors.NumValenceElectrons),
    ("NumRadicalElectrons", Descriptors.NumRadicalElectrons),
    ("MaxPartialCharge", Descriptors.MaxPartialCharge),
    ("MinPartialCharge", Descriptors.MinPartialCharge),
    ("MaxAbsPartialCharge", Descriptors.MaxAbsPartialCharge),
    ("MinAbsPartialCharge", Descriptors.MinAbsPartialCharge),
    ("FpDensityMorgan1", Descriptors.FpDensityMorgan1),
    ("FpDensityMorgan2", Descriptors.FpDensityMorgan2),
    ("FpDensityMorgan3", Descriptors.FpDensityMorgan3),
    ("MolLogP", Descriptors.MolLogP),
    ("MolMR", Descriptors.MolMR),
    ("BalabanJ", Descriptors.BalabanJ),
    ("BertzCT", Descriptors.BertzCT),
    ("HallKierAlpha", Descriptors.HallKierAlpha),
    ("Kappa1", Descriptors.Kappa1),
    ("Kappa2", Descriptors.Kappa2),
    ("Kappa3", Descriptors.Kappa3),
    ("Chi0", Descriptors.Chi0),
    ("Chi1", Descriptors.Chi1),
    ("Chi0n", Descriptors.Chi0n),
    ("Chi1n", Descriptors.Chi1n),
    ("Chi2n", Descriptors.Chi2n),
    ("Chi3n", Descriptors.Chi3n),
    ("Chi4n", Descriptors.Chi4n),
    ("Chi0v", Descriptors.Chi0v),
    ("Chi1v", Descriptors.Chi1v),
    ("Chi2v", Descriptors.Chi2v),
    ("Chi3v", Descriptors.Chi3v),
    ("Chi4v", Descriptors.Chi4v),
    ("TPSA", Descriptors.TPSA),
    ("LabuteASA", Descriptors.LabuteASA),
    ("NumHAcceptors", Descriptors.NumHAcceptors),
    ("NumHDonors", Descriptors.NumHDonors),
    ("NumHeteroatoms", Descriptors.NumHeteroatoms),
    ("NumRotatableBonds", Descriptors.NumRotatableBonds),
    ("NumAromaticRings", Descriptors.NumAromaticRings),
    ("NumSaturatedRings", Descriptors.NumSaturatedRings),
    ("NumAliphaticRings", Descriptors.NumAliphaticRings),
    ("NumAromaticHeterocycles", Descriptors.NumAromaticHeterocycles),
    ("NumSaturatedHeterocycles", Descriptors.NumSaturatedHeterocycles),
    ("NumAliphaticHeterocycles", Descriptors.NumAliphaticHeterocycles),
    ("NumAromaticCarbocycles", Descriptors.NumAromaticCarbocycles),
    ("NumSaturatedCarbocycles", Descriptors.NumSaturatedCarbocycles),
    ("NumAliphaticCarbocycles", Descriptors.NumAliphaticCarbocycles),
    ("RingCount", Descriptors.RingCount),
    ("FractionCSP3", Descriptors.FractionCSP3),
    ("NHOHCount", Descriptors.NHOHCount),
    ("NOCount", Descriptors.NOCount),
    ("HeavyAtomCount", Descriptors.HeavyAtomCount),
    ("NumHeterocycles", lambda m: Descriptors.NumAromaticHeterocycles(m) + Descriptors.NumSaturatedHeterocycles(m) + Descriptors.NumAliphaticHeterocycles(m)),
    ("qed", Descriptors.qed),
]

DESCRIPTOR_DIM = len(_DESCRIPTOR_FUNCS)  # currently 54

# Morgan fingerprints (a.k.a. ECFP4): a bit-vector representation where each
# bit indicates the presence of a particular circular substructure. Radius=2
# means "neighborhoods up to 2 bonds away from each atom". They were the
# pre-deep-learning gold standard for molecular property prediction.
MORGAN_FP_DIM = 2048
MORGAN_RADIUS = 2

# Toggle: include Morgan fingerprint in global features?
# Experiment A (current default) found that descriptors alone outperform
# descriptors + Morgan on the scaffold-split test. Hypothesis: the Morgan
# bits leak in-fold scaffold info that helps val but hurts held-out test.
# Override via env var: HIV_USE_MORGAN=1 to include them.
import os
USE_MORGAN = os.environ.get("HIV_USE_MORGAN", "0") == "1"

# Final dimension of the global feature vector handed to the GNN.
GLOBAL_FEATURE_DIM = DESCRIPTOR_DIM + (MORGAN_FP_DIM if USE_MORGAN else 0)


def get_atom_features(atom):
    """Build a 28-dim feature vector for a single RDKit atom.

    Layout (in order):
        [0:9]    one-hot for C, N, O, S, F, P, Cl, Br, I (the 9 explicit types)
        [9]      one-hot for "Other" (any element not in the explicit list)
        [10:16]  one-hot for degree 0..5 (number of bonded neighbors)
        [16]     formal charge (raw integer, can be negative)
        [17:20]  one-hot for SP, SP2, SP3 hybridization
        [20]     one-hot for "Other" hybridization (SP3D, SP3D2, etc.)
        [21]     1 if aromatic else 0
        [22]     total number of attached hydrogens (raw integer)
        [23:27]  one-hot for chirality (unspecified, R, S, other)
        [27]     1 if atom is in any ring else 0

    Returns a Python list of length 28. The caller stacks these into
    `Data.x` of shape [num_atoms, 28].
    """
    # --- Atom type: 10 dims (9 explicit + 1 "Other") ---
    symbol = atom.GetSymbol()
    # ATOM_TYPES[:-1] excludes "Other" — we set those one-hots explicitly...
    features = [1 if symbol == t else 0 for t in ATOM_TYPES[:-1]]
    # ...and the "Other" slot is 1 only if no explicit type matched.
    features.append(1 if symbol not in ATOM_TYPES[:-1] else 0)

    # --- Degree: 6 dims (one-hot for 0..5 bonded neighbors) ---
    # Atoms with degree > 5 are vanishingly rare in HIV; they'd be all-zero
    # in this block, which is fine — model can learn that pattern.
    degree = atom.GetDegree()
    features.extend([1 if degree == i else 0 for i in range(6)])

    # --- Formal charge: 1 dim (raw integer, typically -2..+2) ---
    # Kept as raw integer rather than one-hot because the model can learn
    # that +1 and +2 are "more positive" — useful for ionic interactions.
    features.append(atom.GetFormalCharge())

    # --- Hybridization: 4 dims (3 explicit + 1 "Other") ---
    hyb = atom.GetHybridization()
    features.extend([1 if hyb == h else 0 for h in HYBRIDIZATION_TYPES])
    features.append(1 if hyb not in HYBRIDIZATION_TYPES else 0)

    # --- Aromaticity flag: 1 dim ---
    # Distinct from "in ring" — benzene atoms are aromatic AND in a ring;
    # cyclohexane atoms are in a ring but NOT aromatic.
    features.append(1 if atom.GetIsAromatic() else 0)

    # --- Hydrogen count: 1 dim (raw integer) ---
    # Total H atoms attached (explicit + implicit). Implicit Hs are not
    # in `mol.GetAtoms()` but RDKit tracks them for valence accounting.
    features.append(atom.GetTotalNumHs())

    # --- Chirality: 4 dims (one-hot) ---
    # Critical for drug activity — many drugs have a single active enantiomer.
    chiral = atom.GetChiralTag()
    features.extend([1 if chiral == t else 0 for t in CHIRAL_TAGS])

    # --- Ring membership: 1 dim ---
    # Useful as a global "is this part of a scaffold" signal.
    features.append(1 if atom.IsInRing() else 0)

    return features


def get_bond_features(bond):
    """Build an 8-dim feature vector for a single RDKit bond.

    Layout:
        [0:4]  one-hot for SINGLE, DOUBLE, TRIPLE, AROMATIC
        [4]    1 if conjugated (part of a delocalized π system) else 0
        [5]    1 if in any ring else 0
        [6]    1 if no stereo info (STEREONONE) else 0
        [7]    1 if has stereo info else 0

    Note: dims [6] and [7] are mutually exclusive — together they form a
    binary "is the stereochemistry of this bond defined" indicator. We don't
    one-hot the specific stereo (E/Z/cis/trans) because the categories are
    too sparse to learn from in this dataset.

    Returns a Python list of length 8.
    """
    # --- Bond order: 4 dims ---
    bt = bond.GetBondType()
    features = [1 if bt == t else 0 for t in BOND_TYPES]

    # --- Conjugation: 1 dim ---
    # Conjugated bonds (alternating single/double in a chain) delocalize
    # electrons — important for reactivity and absorption.
    features.append(1 if bond.GetIsConjugated() else 0)

    # --- Ring membership: 1 dim ---
    features.append(1 if bond.IsInRing() else 0)

    # --- Stereo presence: 2 mutually exclusive dims ---
    # We compress all stereo categories into "has stereo / no stereo" since
    # the specific E/Z/cis/trans distinctions are too rare to be informative
    # at this dataset size.
    stereo = bond.GetStereo()
    features.append(1 if stereo == Chem.rdchem.BondStereo.STEREONONE else 0)
    features.append(1 if stereo != Chem.rdchem.BondStereo.STEREONONE else 0)

    return features


def _safe_descriptor(fn, mol):
    """Run an RDKit descriptor function and return 0.0 on any failure.

    Some descriptors fail on molecules with radicals, weird valences, or
    huge ring systems. Rather than crashing the whole preprocessing, we
    return 0.0 and let the model learn that "0 in this column for some
    weird molecules" is just noise. NaN values from RDKit also get mapped
    to 0.0 so downstream normalization isn't poisoned.
    """
    try:
        v = fn(mol)
        if v is None or not np.isfinite(v):
            return 0.0
        return float(v)
    except Exception:
        return 0.0


def compute_global_features(mol):
    """Compute the whole-molecule global feature vector.

    Returns a numpy float32 array of length:
        DESCRIPTOR_DIM (54) if USE_MORGAN is False (default), or
        DESCRIPTOR_DIM + MORGAN_FP_DIM (54 + 2048 = 2102) if True.

    The descriptor block is on continuous scales (MolWt in g/mol, TPSA in
    Å², counts, etc.), so it gets z-score normalized per-fold during
    training (see main.py:fit_global_feature_stats). The Morgan FP block
    (when present) is binary and is left as-is.
    """
    # Run all 54 descriptor functions, swallowing failures into 0.0.
    desc = np.array(
        [_safe_descriptor(fn, mol) for _, fn in _DESCRIPTOR_FUNCS],
        dtype=np.float32,
    )

    if not USE_MORGAN:
        return desc

    # Morgan fingerprint path (used only when HIV_USE_MORGAN=1).
    # Returns a 2048-bit RDKit ExplicitBitVect; we convert to a dense 0/1
    # numpy array so it can be concatenated with the descriptor floats.
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_FP_DIM)
    fp_arr = np.zeros((MORGAN_FP_DIM,), dtype=np.float32)
    from rdkit.DataStructs import ConvertToNumpyArray
    ConvertToNumpyArray(fp, fp_arr)

    return np.concatenate([desc, fp_arr], axis=0)


def mol_to_graph(smiles, label=None):
    """Convert a SMILES string to a PyTorch Geometric `Data` object.

    Args:
        smiles: SMILES string for the molecule.
        label: Optional binary label (0.0 or 1.0). If provided, attached as `Data.y`.

    Returns:
        A PyG `Data` object with:
          x:               [num_atoms, 28]    atom features
          edge_index:      [2, 2*num_bonds]   bidirectional bond connectivity
          edge_attr:       [2*num_bonds, 8]   bond features (duplicated for both directions)
          global_features: [1, GLOBAL_FEATURE_DIM]  whole-molecule descriptors
          y:               [1]                label (only if `label` is given)
        OR `None` if the SMILES is invalid / produces an atomless molecule.

    Why each bond appears twice: PyG GNNs operate on directed graphs (a message
    flows from src→dst). A chemical bond is undirected, so we add both
    (i,j) and (j,i) so messages can flow in both directions during convolution.
    """
    # SMILES parsing — RDKit returns None for malformed strings. We surface
    # this as None to the caller rather than raising, because the caller
    # (main.py / inference.py) tracks how many molecules got skipped.
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Build atom features. Skip molecules that parsed but have zero atoms
    # (extremely rare — empty SMILES like "" — but possible).
    atom_features = [get_atom_features(a) for a in mol.GetAtoms()]
    if len(atom_features) == 0:
        return None

    # x: [num_atoms, 28]
    x = torch.tensor(atom_features, dtype=torch.float)

    # Build edge_index and edge_attr in parallel. Each chemical bond
    # contributes TWO directed edges (i→j AND j→i) with identical features
    # — message passing needs both directions.
    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = get_bond_features(bond)
        edge_indices.append([i, j])
        edge_indices.append([j, i])
        edge_attrs.append(bf)
        edge_attrs.append(bf)

    # Edge tensors. PyG expects edge_index in [2, num_edges] format (transposed
    # from the natural [num_edges, 2] list-of-pairs layout we built above).
    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        # Edge case: a single-atom "molecule" (e.g. just "C" for methane after
        # implicit-H stripping). No bonds means empty edge tensors with the
        # correct dtype/shape so PyG batching still works.
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, BOND_FEATURE_DIM), dtype=torch.float)

    # Whole-molecule features. Shape note: we store as [1, D] (not [D]) so that
    # when PyG's DataLoader batches B molecules, concatenating along dim 0 gives
    # [B, D] — exactly what the model's global encoder MLP expects as input.
    global_feat = compute_global_features(mol)
    global_features = torch.from_numpy(global_feat).unsqueeze(0)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        global_features=global_features,
    )
    if label is not None:
        # PyG batches scalar labels by stacking, so [1]-shaped tensors per
        # molecule become [B] after batching.
        data.y = torch.tensor([label], dtype=torch.float)

    return data


def get_scaffold(smiles):
    """Compute the Murcko scaffold of a molecule (returns canonical SMILES of scaffold).

    A Murcko scaffold is the molecule reduced to its core ring systems plus
    the linkers connecting them — side chains and terminal substituents are
    removed. Aspirin and salicylic acid share the same scaffold even though
    they're chemically different compounds.

    This is the unit of "structural class" we use to define scaffold-based
    splits. Two molecules with the same scaffold are considered structurally
    equivalent for evaluation purposes — they MUST end up in the same fold,
    or the model can cheat by memorizing scaffold→label associations.

    `includeChirality=False` means R/S stereo is ignored when defining the
    scaffold — two enantiomers share a scaffold. We accept this because
    the activity *signal* often follows scaffold (one isomer active, one
    inactive), and grouping enantiomers prevents trivial leakage where
    train sees one enantiomer and test sees the other.

    Falls back to the original SMILES on RDKit failure — that single weird
    molecule then becomes its own singleton scaffold group, which is the
    safest behavior (it can't accidentally cluster with anything else).
    """
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(
            smiles=smiles, includeChirality=False
        )
    except Exception:
        return smiles
