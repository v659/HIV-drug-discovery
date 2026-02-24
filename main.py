import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool

from sklearn.metrics import roc_auc_score
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import deepchem as dc

from multiprocessing import Pool, cpu_count

BATCH_SIZE = 128
EPOCHS = 100
LR = 1e-4
FOLDS = 5

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def safe_roc_auc(y_true, y_score):
    y_true = np.asarray(y_true)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return None
    return roc_auc_score(y_true, y_score)

def get_atom_features(atom):
    features = []

    atom_types = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I']
    symbol = atom.GetSymbol()
    features.extend([1 if symbol == t else 0 for t in atom_types])
    features.append(1 if symbol not in atom_types else 0)

    features.extend([1 if atom.GetDegree() == i else 0 for i in range(6)])
    features.append(atom.GetFormalCharge())

    hybrid_types = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3
    ]
    features.extend([1 if atom.GetHybridization() == h else 0 for h in hybrid_types])
    features.append(1 if atom.GetHybridization() not in hybrid_types else 0)

    features.append(int(atom.GetIsAromatic()))
    features.append(atom.GetTotalNumHs())

    return features

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    x = [get_atom_features(a) for a in mol.GetAtoms()]
    if len(x) == 0:
        return None

    x = torch.tensor(x, dtype=torch.float)
    edges = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        edges.append([i, j])
        edges.append([j, i])

    edge_index = (
        torch.tensor(edges, dtype=torch.long).t().contiguous()
        if edges else torch.empty((2, 0), dtype=torch.long)
    )

    return Data(x=x, edge_index=edge_index)

def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)

def process_smiles(args):
    smiles, label = args
    graph = smiles_to_graph(smiles)
    scaffold = get_scaffold(smiles)

    if graph is None or scaffold is None:
        return None

    graph.y = torch.tensor([label], dtype=torch.float)
    return graph, scaffold, label

class HIVGNN(nn.Module):
    def __init__(self, num_features, hidden_dim=128, num_layers=3):
        super().__init__()

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(GCNConv(num_features, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

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

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)

        x = torch.cat(
            [global_mean_pool(x, batch), global_max_pool(x, batch)],
            dim=1
        )
        return self.mlp(x).squeeze(1)

if __name__ == "__main__":

    data_dir = dc.utils.get_data_dir()
    df = pd.read_csv(f"{data_dir}/hiv.csv").dropna()

    inputs = list(zip(df["smiles"], df["HIV_active"]))

    print("Processing molecules (parallel)...")
    with Pool(processes=max(1, cpu_count() - 1)) as pool:
        results = pool.map(process_smiles, inputs)

    results = [r for r in results if r is not None]

    graphs, scaffolds, labels = zip(*results)

    graphs = list(graphs)
    scaffolds = list(scaffolds)
    labels = np.array(labels)

    df = pd.DataFrame({"scaffold": scaffolds})

    print(f"Loaded {len(graphs)} valid molecules")
    print(f"Atom feature size: {graphs[0].x.shape[1]}")

    unique_scaffolds = np.array(df["scaffold"].unique())
    np.random.shuffle(unique_scaffolds)
    fold_scaffolds = np.array_split(unique_scaffolds, FOLDS)

    fold_aucs = []

    for fold in range(FOLDS):
        print(f"\nFOLD {fold + 1}/{FOLDS}")

        test_scaf = set(fold_scaffolds[fold])
        train_scaf = set(unique_scaffolds) - test_scaf

        train_idx = df[df["scaffold"].isin(train_scaf)].index.to_list()
        test_idx = df[df["scaffold"].isin(test_scaf)].index.to_list()

        np.random.shuffle(train_idx)
        if len(train_idx) > 1:
            val_size = max(1, int(0.1 * len(train_idx)))
            val_size = min(val_size, len(train_idx) - 1)
        else:
            val_size = 0
        val_idx = train_idx[:val_size]
        train_idx = train_idx[val_size:]

        if len(train_idx) == 0 or len(test_idx) == 0:
            print("Skipping fold due to empty train/test split after scaffold partitioning.")
            fold_aucs.append(np.nan)
            continue

        train_loader = DataLoader([graphs[i] for i in train_idx], batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader([graphs[i] for i in val_idx], batch_size=BATCH_SIZE)
        test_loader  = DataLoader([graphs[i] for i in test_idx], batch_size=BATCH_SIZE)

        y_train = labels[train_idx]
        num_pos = (y_train == 1).sum()
        num_neg = (y_train == 0).sum()
        if num_pos == 0 or num_neg == 0:
            print("Warning: training split has only one class; using pos_weight=1.0 for stability.")
            pos_weight_value = 1.0
        else:
            pos_weight_value = num_neg / num_pos
        pos_weight = torch.tensor(pos_weight_value, device=DEVICE)

        model = HIVGNN(graphs[0].x.shape[1]).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

        best_auc = -np.inf
        patience, counter = 10, 0
        has_saved_checkpoint = False

        for epoch in range(EPOCHS):
            model.train()
            for batch in train_loader:
                batch = batch.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(batch), batch.y)
                loss.backward()
                optimizer.step()

            model.eval()
            preds, ys = [], []
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(DEVICE)
                    preds.extend(torch.sigmoid(model(batch)).cpu().numpy())
                    ys.extend(batch.y.cpu().numpy())

            auc = safe_roc_auc(ys, preds)
            if auc is None:
                if not has_saved_checkpoint:
                    torch.save(model.state_dict(), f"best_fold{fold}.pth")
                    has_saved_checkpoint = True
                counter += 1
            elif auc > best_auc:
                best_auc = auc
                counter = 0
                torch.save(model.state_dict(), f"best_fold{fold}.pth")
                has_saved_checkpoint = True
            else:
                counter += 1

            if counter >= patience:
                break

        if not has_saved_checkpoint:
            torch.save(model.state_dict(), f"best_fold{fold}.pth")

        model.load_state_dict(torch.load(f"best_fold{fold}.pth"))
        model.eval()

        preds, ys = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(DEVICE)
                preds.extend(torch.sigmoid(model(batch)).cpu().numpy())
                ys.extend(batch.y.cpu().numpy())

        test_auc = safe_roc_auc(ys, preds)
        if test_auc is None:
            print(f"Fold {fold + 1} Test AUC = N/A (single-class or empty test labels)")
            test_auc = np.nan
        else:
            print(f"Fold {fold + 1} Test AUC = {test_auc:.4f}")
        fold_aucs.append(test_auc)

    print("\n FINAL RESULTS ")
    print("Fold AUCs:", [("N/A" if np.isnan(a) else f"{a:.4f}") for a in fold_aucs])
    print(f"Mean AUC = {np.nanmean(fold_aucs):.4f}")
    print(f"Std  AUC = {np.nanstd(fold_aucs):.4f}")
