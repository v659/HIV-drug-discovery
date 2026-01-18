"""
  Citation Required:
  Arjun Chandra Agarwal, "HIV drug Discovery", 2026
  https://github.com/v659/HIV-drug-discovery
"""


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from sklearn.metrics import roc_auc_score
from rdkit import Chem
from rdkit.Chem import AllChem, FilterCatalog
from rdkit.Chem.Scaffolds import MurckoScaffold

BATCH_SIZE = 128
EPOCHS = 100
LR = 1e-4
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
FOLDS = 5
USE_3D = True
FILTER_PAINS = True

params = FilterCatalog.FilterCatalogParams()
params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
pains_catalog = FilterCatalog.FilterCatalog(params)


def is_pains(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return pains_catalog.HasMatch(mol)


def get_atom_features(atom, conf=None):
    features = []
    # Atom type (one-hot)
    atom_types = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'Other']
    atom_symbol = atom.GetSymbol()
    features.extend([1 if atom_symbol == t else 0 for t in atom_types[:-1]])
    features.append(1 if atom_symbol not in atom_types[:-1] else 0)

    features.extend([1 if atom.GetDegree() == i else 0 for i in range(6)])

    features.append(atom.GetFormalCharge())

    hybrid_types = [Chem.rdchem.HybridizationType.SP,
                    Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3]
    features.extend([1 if atom.GetHybridization() == h else 0 for h in hybrid_types])
    features.append(1 if atom.GetHybridization() not in hybrid_types else 0)

    features.append(1 if atom.GetIsAromatic() else 0)

    features.append(atom.GetTotalNumHs())

    if USE_3D:
        if conf is not None:
            pos = conf.GetAtomPosition(atom.GetIdx())
            features.extend([pos.x, pos.y, pos.z])
        else:
            features.extend([0.0, 0.0, 0.0])

    return features

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    conf = None
    if USE_3D:
        try:
            mol = Chem.AddHs(mol)
            result = AllChem.EmbedMolecule(mol, randomSeed=42, maxAttempts=1)
            if result == 0:  # Success
                AllChem.MMFFOptimizeMolecule(mol, maxIters=100)  # Limit iterations
                mol = Chem.RemoveHs(mol)
                conf = mol.GetConformer()
        except:
            pass  # Fall back to 2D if 3D generation fails

    # Get atom features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom, conf))

    if len(atom_features) == 0:
        return None

    x = torch.tensor(atom_features, dtype=torch.float)

    # Get edge indices
    edge_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.append([i, j])
        edge_indices.append([j, i])

    if len(edge_indices) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)


# -------------------- Scaffold --------------------
def scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)


# -------------------- GNN Model --------------------
class HIVGNN(nn.Module):
    def __init__(self, num_features, hidden_dim=128, num_layers=3):
        super().__init__()

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # First layer
        self.convs.append(GCNConv(num_features, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # MLP for prediction
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Graph convolutions
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)

        # Global pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)

        # Prediction
        return self.mlp(x).squeeze(1)


import os

csv_path = "hiv.csv"
if not os.path.exists(csv_path):
    print("Downloading HIV dataset...")
    import urllib.request

    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv"
    urllib.request.urlretrieve(url, csv_path)
    print("Download complete!")

df = pd.read_csv(csv_path).dropna()

# Filter PAINS if enabled
if FILTER_PAINS:
    print("Filtering PAINS compounds...")
    df['is_pains'] = df['smiles'].apply(is_pains)
    pains_count = df['is_pains'].sum()
    df = df[~df['is_pains']].drop('is_pains', axis=1).reset_index(drop=True)
    print(f"Removed {pains_count} PAINS compounds ({pains_count / len(df) * 100:.1f}%)")

print(f"Loading molecules with {'3D' if USE_3D else '2D'} features...")
graphs, labels, scaffolds = [], [], []
for i, (s, y) in enumerate(zip(df["smiles"], df["HIV_active"])):
    if i % 5000 == 0:
        print(f"  Processed {i}/{len(df)} molecules...")
    graph = smiles_to_graph(s)
    scaf = scaffold(s)
    if graph is not None and scaf is not None:
        graph.y = torch.tensor([y], dtype=torch.float)
        graphs.append(graph)
        labels.append(y)
        scaffolds.append(scaf)

y = np.array(labels)
df = pd.DataFrame({"smiles": df["smiles"].iloc[:len(y)], "scaffold": scaffolds})

print(f"\nLoaded {len(graphs)} valid molecules")
print(f"Number of atom features: {graphs[0].x.shape[1]} ({'2D only' if not USE_3D else '2D + 3D coordinates'})")

# Check how many 3D generations failed
if USE_3D:
    failed_3d = 0
    for graph in graphs:
        # Check if all atoms have (0,0,0) coordinates
        coords = graph.x[:, -3:]  # Last 3 features are x,y,z
        if torch.all(coords == 0):
            failed_3d += 1

    success_3d = len(graphs) - failed_3d
    print(f"\n3D Structure Generation Results:")
    print(f"  Successful: {success_3d}/{len(graphs)} ({success_3d / len(graphs) * 100:.1f}%)")
    print(f"  Failed:     {failed_3d}/{len(graphs)} ({failed_3d / len(graphs) * 100:.1f}%)")
    print(f"  (Failed molecules use dummy coordinates and rely on 2D features)")

# -------------------- Create 5-Fold Scaffold Split --------------------
unique_scaffolds = list(df["scaffold"].unique())
np.random.shuffle(unique_scaffolds)

fold_scaffolds = np.array_split(unique_scaffolds, FOLDS)

fold_test_aucs = []

for fold in range(FOLDS):
    print(f"\n{'=' * 50}")
    print(f"FOLD {fold + 1}/{FOLDS}")
    print(f"{'=' * 50}")

    test_scaf = set(fold_scaffolds[fold])
    train_scaf = set(unique_scaffolds) - test_scaf

    train_idx = df[df["scaffold"].isin(train_scaf)].index.tolist()
    test_idx = df[df["scaffold"].isin(test_scaf)].index.tolist()

    # 10% of train becomes validation
    np.random.shuffle(train_idx)
    val_size = int(0.1 * len(train_idx))
    val_idx = train_idx[:val_size]
    train_idx = train_idx[val_size:]

    train_graphs = [graphs[i] for i in train_idx]
    val_graphs = [graphs[i] for i in val_idx]
    test_graphs = [graphs[i] for i in test_idx]

    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE)

    # Calculate positive weight for loss
    y_train = np.array([y[i] for i in train_idx])
    pos_weight = torch.tensor(
        (y_train == 0).sum() / (y_train == 1).sum(),
        device=DEVICE, dtype=torch.float32
    )

    # Initialize model
    num_features = graphs[0].x.shape[1]
    model = HIVGNN(num_features=num_features).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
    )

    best_val_auc = 0.0
    patience = 10
    counter = 0

    for epoch in range(EPOCHS):
        # Training
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                out = model(batch)
                probs = torch.sigmoid(out).cpu().numpy()
                val_preds.extend(probs)
                val_labels.extend(batch.y.cpu().numpy())

        val_auc = roc_auc_score(val_labels, val_preds)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS}: Loss = {total_loss / len(train_loader):.4f}, Val AUC = {val_auc:.4f}")

        scheduler.step(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            counter = 0
            torch.save(model.state_dict(), f"best_gnn_fold{fold}_{'3d' if USE_3D else '2d'}.pth")
        else:
            counter += 1

        if counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # Load best model and evaluate on test
    model.load_state_dict(torch.load(f"best_gnn_fold{fold}_{'3d' if USE_3D else '2d'}.pth"))
    model.eval()

    test_preds = []
    test_labels = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            out = model(batch)
            probs = torch.sigmoid(out).cpu().numpy()
            test_preds.extend(probs)
            test_labels.extend(batch.y.cpu().numpy())

    test_auc = roc_auc_score(test_labels, test_preds)

    print(f"\nFold {fold + 1} Test AUC = {test_auc:.4f}")
    fold_test_aucs.append(test_auc)

mean_auc = np.mean(fold_test_aucs)
std_auc = np.std(fold_test_aucs)

print(f"\n{'=' * 50}")
print(
    f"=== 5-Fold Cross-Validation Results ({'3D' if USE_3D else '2D'}, {'PAINS filtered' if FILTER_PAINS else 'No filter'}) ===")
print(f"{'=' * 50}")
print(f"Individual fold AUCs: {[f'{auc:.4f}' for auc in fold_test_aucs]}")
print(f"Mean Test AUC = {mean_auc:.4f}")
print(f"Std Test AUC  = {std_auc:.4f}")
