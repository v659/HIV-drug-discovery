# HIV-drug-discovery
A OpenSource model for researchers to use for finding new HIV drugs without having to test thousands of molecules in a lab. Currently in **Alpha stage**

# Motive while building model

To build a good, accurate model that researchers can use for free

# Dataset

https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv

Dataset contains around 41K molecules. Heavy class imbalance(2.5% actives only), some assay compounds, need to filter invalid molecules

**Cleaning dataset**

1. Remove invalid molecules:
  Remove invalid SMILES(Notation for molecules)
  Remove Invalid molecules after 3D embedding if embedding fails(not implemented yet)
  Remove if graph is invalid(num.features = 0)
  Invalid Edges (Empty Bond Graph) are handled by setting value to 0
  Remove molecule if graph or scaffold is None
**NOTE**: Molecules with failed 3D embedding are currently not removed.

3. PAINS filtering:
PAINS are chemical compounds that often give false positive results in high-throughput screens.
Filter using FilterCatalog from rdkit

Neural network type:
GNN with atoms as nodes and bonds as edges

ATOMS
FEATURE COUNT - 26

Atom type, Atoms Degrees, Hybridization type, IsAromatic, Num. Hydrogens
Nodes added later(Bonds)

Model:
self.mlp = nn.Sequential(
    nn.Linear(hidden_dim * 2, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 1)
)
**Note: Not optimized for 3D gnn yet**

Model uses BCEWITHLOGITSLOSS
Optimizer - Adam weight decay

**Model is not properly functional yet(Expexted results would be around 0.70 AUC compared to the 0.78 AUC with previously 2D model.) Parameters still under testing**

## Citation Requirement

If you use this software in research, you must cite:

Arjun Chandra Agarwal, “HIV drug classification”, 2026, https://github.com/v659/HIV-drug-discovery

If you deploy this software in a public service or distribute modified versions,
you must include this citation in the documentation or “About” page.
