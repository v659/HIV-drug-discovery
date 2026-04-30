"""MolFormer-XL classification wrapper for HIV bioactivity prediction.

This module is the transformer counterpart to model.py's GNN. Where the GNN
sees explicit molecular graph structure, MolFormer sees SMILES as text and
reuses representations learned by IBM's pretraining on 1.1 billion molecules
from ZINC and PubChem.

The bet: molecules in the HIV test set whose scaffolds the GNN has never seen
during HIV training may still be familiar to MolFormer because *something
structurally similar* appeared in its 1.1B-molecule pretraining corpus.
That foundation-model prior is the lever for crossing the scaffold-
generalization wall the GNN hits at ~0.78 mean test AUC.

Architecture:
  SMILES → tokenizer → [B, T] token IDs
                            │
                            ▼
                    MolFormer backbone (47M params, hidden=768)
                            │
                            ▼
              last_hidden_state [B, T, 768]
                            │
                            ▼  masked mean pool
                    pooled [B, 768]
                            │
                            ▼
              MLP head: 768 → 256 → 1
                            │
                            ▼
                  logit per molecule [B]

Why masked mean pooling (not CLS token)?
  MolFormer was pretrained with masked-LM, not next-sentence-style objectives,
  so its CLS-position token doesn't carry the kind of summary representation
  that BERT's [CLS] does. Masked mean over real (non-padding) tokens is the
  community-standard fallback for this family of molecular transformers.

Why a 2-layer MLP head, not just a Linear?
  The single-Linear "linear probe" approach assumes the backbone's pooled
  embedding is already linearly separable for the downstream task. For
  bioactivity that's a stretch — a 2-layer head with GELU gives the model
  capacity to learn nonlinear combinations of pretrained features without
  adding many parameters (768·256 + 256·1 ≈ 197k params total in the head).
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# IBM's MolFormer-XL pretrained on 10% of ZINC+PubChem (~1.1B molecules).
# `both` indicates the model was trained on both data sources combined.
# This is the standard checkpoint cited in the MolFormer paper.
MOLFORMER_NAME = "ibm/MoLFormer-XL-both-10pct"


def load_tokenizer():
    """Load MolFormer's custom SMILES tokenizer.

    `trust_remote_code=True` is required because MolFormer ships custom
    tokenizer/model code on HuggingFace (not part of stock transformers
    library). HF will download and execute that code — which is why HF
    prints a warning. The IBM repo is a known-good source.
    """
    return AutoTokenizer.from_pretrained(MOLFORMER_NAME, trust_remote_code=True)


def load_backbone():
    """Load the pretrained MolFormer-XL transformer backbone.

    `deterministic_eval=True` makes the rotary attention deterministic at
    eval time — important for reproducible inference results across runs.
    """
    return AutoModel.from_pretrained(
        MOLFORMER_NAME,
        deterministic_eval=True,
        trust_remote_code=True,
    )


class MolFormerClassifier(nn.Module):
    """MolFormer-XL backbone + 2-layer MLP head → single binary logit.

    Args:
        hidden_dim:  Hidden width of the MLP head. 256 is a good default —
                     enough capacity to learn nonlinear features without
                     overfitting on a 41k-molecule dataset.
        dropout:     Applied before each Linear in the head. Lower than the
                     GNN's 0.3 because MolFormer is already heavily regularized
                     by its pretraining + the small head.
        gradient_checkpointing: If True, the backbone re-runs forward during
                     backward to save activation memory at the cost of ~25%
                     slower training. T4 (16GB) handles batch=16 without it;
                     enable for batch≥64 or larger backbones.

    Output: per-molecule logit (apply sigmoid for probability).
    """

    def __init__(self, hidden_dim=256, dropout=0.1, gradient_checkpointing=False):
        super().__init__()
        self.backbone = load_backbone()
        if gradient_checkpointing:
            # Saves ~30% VRAM at the cost of ~25% slower backward.
            # try/except: not all backbone implementations expose this method.
            try:
                self.backbone.gradient_checkpointing_enable()
            except Exception:
                pass
        # MolFormer-XL hidden size = 768 (standard BERT-base width).
        backbone_hidden = self.backbone.config.hidden_size

        # 2-layer MLP head with GELU activation (matches transformer
        # convention; smoother than ReLU and what the backbone was
        # pretrained with).
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(backbone_hidden, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, input_ids, attention_mask):
        """Forward pass.

        Args:
            input_ids:      [B, T] tokenizer-produced token IDs (padded).
            attention_mask: [B, T] 1 for real tokens, 0 for padding.

        Returns:
            logits: [B] one logit per molecule.
        """
        # Run the transformer backbone. Output includes last_hidden_state
        # (per-token embeddings) and other artifacts we don't need here.
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = out.last_hidden_state  # [B, T, 768]

        # Masked mean pool: average ONLY over real (non-padding) tokens.
        # If we naively took mean(dim=1), padding zeros would bias the
        # average toward 0 for short molecules with lots of padding.
        mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)  # [B, T, 1]
        # Zero out padding token contributions, then sum.
        summed = (last_hidden * mask).sum(dim=1)        # [B, 768]
        # Divide by the number of real tokens per molecule (clamp to 1
        # to avoid 0-division on a hypothetical all-padding row).
        denom = mask.sum(dim=1).clamp(min=1.0)          # [B, 1]
        pooled = summed / denom                          # [B, 768]

        # MLP head → squeeze trailing singleton → [B] logits.
        return self.head(pooled).squeeze(-1)

    def param_groups(self, backbone_lr, head_lr, weight_decay):
        """Build two-LR optimizer parameter groups.

        Standard transformer fine-tune recipe:
          - Backbone gets a TINY learning rate (1e-5). It's pretrained, so we
            want to perturb it minimally — too high a LR causes "catastrophic
            forgetting" where the model loses its 1.1B-molecule prior.
          - Head gets a HIGH learning rate (1e-3). It's freshly initialized
            and needs to learn from scratch in a few epochs.

        Returning these as separate groups lets AdamW track separate moment
        estimates per group, which is necessary for stable optimization of
        parameters at very different scales.
        """
        return [
            {"params": self.backbone.parameters(), "lr": backbone_lr,
             "weight_decay": weight_decay},
            {"params": self.head.parameters(), "lr": head_lr,
             "weight_decay": weight_decay},
        ]
