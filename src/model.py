"""GNN architecture for HIV bioactivity prediction.

Architecture overview (default hyperparameters):

    Atom features [N, 28] ────► atom_proj (Linear) ────► [N, 128]
                                                              │
                                                              ▼
    Edge features [E, 8] ─────► edge_proj (Linear) ────► [E, 128]
                                                              │
                                                              ▼
                                  ┌──── 3× GATv2 block ────┐
                                  │  GATv2Conv(128→32, h=4)│
                                  │  → BatchNorm → ReLU    │   (residual skip)
                                  │  → Dropout → +residual │
                                  └────────────┬───────────┘
                                               │  [N, 128]
                                               ▼
                            mean_pool ┐    max_pool ┐
                                      ▼             ▼
                                   [B, 128]      [B, 128]
                                            │
                                            ▼  concat → [B, 256]
                                            │
                                            │              global descriptors [B, 54]
                                            │                       │
                                            │                       ▼
                                            │              global_encoder MLP
                                            │                       │
                                            │                  [B, 256]
                                            ▼                       ▼
                                            └───── concat → [B, 512]
                                                          │
                                                          ▼
                                              MLP head: 512 → 256 → 128 → 1
                                                          │
                                                          ▼
                                                      logit per molecule

Why GATv2 (not GCN/GIN/MPNN)?
  - GATv2 ("dynamic graph attention v2") learns *which* neighbors matter for
    each atom, instead of treating all neighbors equally. For molecules,
    neighbor importance varies wildly: a carbonyl oxygen carries different
    weight than a methyl group attached to the same carbon.
  - GATv2 fixes a known limitation of GAT v1 where attention couldn't depend
    on the *target* node — only the source. v2's attention is fully expressive.
  - Native edge-feature support: bond properties (order, conjugation, ring
    membership) directly modulate attention weights.

Why residual + BatchNorm?
  - Without residuals, deep GNNs suffer "oversmoothing" — every node's
    embedding converges to the same vector after a few layers. Residuals
    let the model preserve fine-grained per-atom information across depth.
  - BatchNorm stabilizes training and lets us use a slightly higher LR.

Why mean+max pooling (not just mean)?
  - Mean captures average atom behavior across the molecule.
  - Max captures whether ANY atom has an extreme value on each feature
    dimension. Useful for identifying critical functional groups whose
    presence is dilutied by averaging.

Why a separate global feature pathway?
  - Whole-molecule descriptors (logP, TPSA, drug-likeness) are global
    properties that the GNN would have to reconstruct from per-atom info
    via deep message passing. Giving them directly is a strong shortcut
    that improves sample efficiency.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool

# Bumped whenever the architecture changes in a way that breaks checkpoint
# compatibility. Older checkpoints with a different MODEL_VERSION cannot be
# loaded into this class without retraining.
MODEL_VERSION = 3


class HIVGNN(nn.Module):
    """GATv2-based graph neural network for binary molecular property prediction.

    Args:
        atom_dim:    Dim of input atom features (default 28, matches features.py).
        edge_dim:    Dim of input bond features (default 8, matches features.py).
        global_dim:  Dim of whole-molecule descriptor vector. 0 disables the
                     global pathway entirely. Default 54 (descriptors only) or
                     2102 (descriptors + Morgan FP) depending on USE_MORGAN.
        hidden_dim:  Internal channel width. Must be divisible by num_heads
                     since GATv2 splits hidden_dim across heads.
        num_heads:   Number of attention heads in each GATv2 layer.
        num_layers:  Number of GATv2 blocks (each = conv + norm + activation
                     + dropout + residual). 3 is the sweet spot for this
                     dataset; deeper hurts due to oversmoothing.
        dropout:     Applied inside GATv2 attention AND between MLP layers.
        global_hidden: Hidden width of the global descriptor encoder MLP.

    Output: a single logit per molecule (apply sigmoid for probability).
    """

    def __init__(
        self,
        atom_dim=28,
        edge_dim=8,
        global_dim=0,
        hidden_dim=128,
        num_heads=4,
        num_layers=3,
        dropout=0.3,
        global_hidden=256,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.global_dim = global_dim

        # Project sparse one-hot-style atom/bond features to dense `hidden_dim`
        # vectors that GATv2 can operate on. Without this projection, the very
        # first GAT layer would have to do the heavy lifting of converting
        # one-hots to embeddings AND learning attention simultaneously.
        self.atom_proj = nn.Linear(atom_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)

        # Build the stack of GATv2 blocks. Each block: GATv2 conv → BN → ReLU
        # → Dropout → residual add. Residual requires that input/output dims
        # match (both `hidden_dim`), which is why we use concat=True with
        # `hidden_dim // num_heads` per head.
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                GATv2Conv(
                    hidden_dim,
                    hidden_dim // num_heads,  # per-head out dim
                    heads=num_heads,
                    edge_dim=hidden_dim,       # use projected edge features
                    concat=True,                # heads concatenate → hidden_dim
                    dropout=dropout,            # dropout inside attention weights
                )
            )
            self.norms.append(nn.BatchNorm1d(hidden_dim))

        # Optional global feature pathway. Encoded with a 2-layer MLP rather
        # than directly fed to the head — gives descriptors room to interact
        # before fusion with the graph representation.
        if global_dim > 0:
            self.global_encoder = nn.Sequential(
                nn.Linear(global_dim, global_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(global_hidden, global_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            # MLP head input = [mean_pool || max_pool || global_repr]
            mlp_in = hidden_dim * 2 + global_hidden
        else:
            self.global_encoder = None
            # MLP head input = [mean_pool || max_pool]
            mlp_in = hidden_dim * 2

        # Classification head. 512 → 256 → 128 → 1. Three layers gives the
        # head capacity to learn nonlinear combinations of graph + global
        # features without being so wide it overfits the small dataset.
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, data):
        """Forward pass.

        Args:
            data: A PyG Batch (DataLoader-produced) with attributes:
                  x:               [N_total, atom_dim]   atoms across all graphs
                  edge_index:      [2, E_total]          edges across all graphs
                  edge_attr:       [E_total, edge_dim]   bond features
                  batch:           [N_total]             which graph each atom belongs to
                  global_features: [B, global_dim]       per-molecule descriptors

        Returns:
            logits: [B] — one raw logit per molecule. Apply sigmoid for prob.
        """
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        # Lift sparse input features into the dense hidden space.
        x = self.atom_proj(x)
        edge_attr = self.edge_proj(edge_attr)

        # GATv2 blocks with residuals. Each iteration:
        #   1. Save current x as residual.
        #   2. Apply GATv2 attention (uses edge features in attention weights).
        #   3. BatchNorm normalizes per-feature across the batch.
        #   4. ReLU + Dropout.
        #   5. Add the residual back. Net effect: every layer is a "delta"
        #      added to the previous representation, preventing oversmoothing.
        for i in range(self.num_layers):
            residual = x
            x = self.convs[i](x, edge_index, edge_attr=edge_attr)
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + residual

        # Pool atom-level features to molecule-level. PyG's pooling functions
        # use the `batch` index to know which atoms belong to which molecule.
        # mean_pool: averages atom embeddings within each molecule.
        # max_pool:  takes the max along each feature dim within each molecule.
        x_mean = global_mean_pool(x, batch)  # [B, hidden_dim]
        x_max = global_max_pool(x, batch)    # [B, hidden_dim]
        graph_repr = torch.cat([x_mean, x_max], dim=1)  # [B, 2*hidden_dim]

        # Fuse the global descriptor pathway if enabled.
        if self.global_encoder is not None:
            g = data.global_features
            # Edge case: a single-graph batch could collapse to [D]. Promote
            # back to [1, D] so the encoder's Linear layer works.
            if g.dim() == 1:
                g = g.unsqueeze(0)
            g = self.global_encoder(g)             # [B, global_hidden]
            graph_repr = torch.cat([graph_repr, g], dim=1)  # [B, 2*hidden + global_hidden]

        # MLP head → single logit per molecule. Squeeze the trailing singleton
        # dim so the output is [B] not [B, 1] — matches the label shape.
        return self.mlp(graph_repr).squeeze(-1)
