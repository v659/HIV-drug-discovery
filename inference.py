import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from rdkit import Chem

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def get_atom_features(atom):
    features = []

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

    return features


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # Get atom features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))

    if len(atom_features) == 0:
        raise ValueError(f"No atoms found in molecule: {smiles}")

    x = torch.tensor(atom_features, dtype=torch.float)

    # Get edge indices
    edge_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.append([i, j])
        edge_indices.append([j, i])  # Add reverse edge for undirected graph

    if len(edge_indices) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)

class HIVGNN(nn.Module):
    def __init__(self, num_features, hidden_dim=128, num_layers=3):
        super().__init__()

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        self.convs.append(GCNConv(num_features, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

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

        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)

        return self.mlp(x).squeeze(1)

def predict_hiv_activity(smiles, model_path):

    print(f"Processing SMILES: {smiles}")

    try:
        graph = smiles_to_graph(smiles)
    except Exception as e:
        return {"error": str(e), "smiles": smiles}

    # Add batch dimension
    graph.batch = torch.zeros(graph.x.shape[0], dtype=torch.long)

    # Load model
    num_features = graph.x.shape[1]
    model = HIVGNN(num_features=num_features).to(DEVICE)

    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"Loaded model from {model_path}")
    except Exception as e:
        return {"error": f"Failed to load model: {str(e)}", "smiles": smiles}

    model.eval()

    # Make prediction
    with torch.no_grad():
        graph = graph.to(DEVICE)
        logit = model(graph)
        probability = torch.sigmoid(logit).item()

    # Interpret results
    prediction = "Active" if probability > 0.5 else "Inactive"
    confidence = probability if probability > 0.5 else (1 - probability)

    return {
        "smiles": smiles,
        "prediction": prediction,
        "probability_active": round(probability, 4),
        "probability_inactive": round(1 - probability, 4),
        "confidence": round(confidence, 4)
    }


def predict_batch(smiles_list, model_path):

    results = []
    for smiles in smiles_list:
        result = predict_hiv_activity(smiles, model_path)
        results.append(result)

        if "error" not in result:
            print(f"  → Prediction: {result['prediction']} "
                  f"(P(active) = {result['probability_active']:.4f})")
        else:
            print(f"   Error: {result['error']}")
        print()

    return results

if __name__ == "__main__":
    test_smiles = "O=[N+]([O-])c1ccc(Cl)c(C2SCc3nc4ccccc4n32)c1"

    print("HIV Activity Prediction")
    print()

    result = predict_hiv_activity(test_smiles, model_path="best_fold2.pth")

    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"SMILES: {result['smiles']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Probability of Active: {result['probability_active']:.4f}")
        print(f"Probability of Inactive: {result['probability_inactive']:.4f}")
        print(f"Confidence: {result['confidence']:.4f}")

    print()
    print("=" * 60)

    print("\nBatch Prediction Example:")
    print("-" * 60)

    batch_smiles = [
        "C[n+]1c(-c2ccc(C=NNC(=O)c3cccc(C(=O)NN=Cc4ccc(-c5cn6ccccc6[n+]5C)cc4)c3)cc2)cn2ccccc21.Cc1ccc(S(=O)(=O)O)cc1"
    ]

    batch_results = predict_batch(batch_smiles, model_path="best_fold2.pth")

    # Summary
    print("\nSummary:")
    print("-" * 60)
    for i, res in enumerate(batch_results, 1):
        if "error" not in res:
            print(f"{i}. {res['prediction']} (P={res['probability_active']:.3f})")
        else:
            print(f"{i}. Error: {res['error']}")
