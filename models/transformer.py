"""
BDI Transformer — Entangled Baseline
=====================================
A standard transformer encoder that reads a partial trajectory and the current
world state, then predicts the agent's goal, belief state, and desire profile
through three independent linear heads attached to a single [CLS] token.

This is the *entangled* baseline: all three predictions share the same latent
representation.  The hypothesis under test is that a BDI-factorised model
(which maintains separate latent subspaces z_b, z_d, z_i) will outperform
this model on compositional generalisation (test condition 4).

Architecture
------------
Input sequence  : [CLS] · [WS] · step_0 · step_1 · … · step_{T-1}

  [CLS]   — learned parameter, provides a pooled representation for heads
  [WS]    — world state W ∈ {0,1}^N projected to d_model via linear layer
  step_t  — grid position at step t, encoded as a learned embedding indexed
             by the flat position index (row × n_cols + col), 1-indexed so
             that index 0 is reserved for padding

Step order is encoded by a second learned embedding (step_embedding) added to
each step token before the transformer, giving the model explicit access to
temporal position within the trajectory prefix.

Prediction heads (all linear):
  goal    → (B, N)  N-class logits  → cross-entropy loss
  belief  → (B, N)  N independent logits  → BCE-with-logits loss
  desire  → (B, N)  N logits, softmax → KL-divergence loss

Note on agent ID
----------------
Agent ID is intentionally excluded.  For familiar agents it creates a
shortcut (agent_id → desires via memorisation) that bypasses desire
inference.  For novel agents it is unseen.  Including it would compromise
the fairness of the compositional generalisation evaluation.
The principled alternative — past trajectory context C_a — is left for
future work.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BDITransformer(nn.Module):
    """
    Parameters
    ----------
    n_pois : int
        Number of points of interest N (output dimensionality of all heads).
    grid_size : (int, int)
        (rows, cols) of the grid.  Used to compute the position vocabulary size.
    d_model : int
        Transformer hidden dimension.
    n_heads : int
        Number of self-attention heads.  Must divide d_model evenly.
    n_layers : int
        Number of transformer encoder layers.
    d_ff : int
        Feedforward sublayer inner dimension (typically 4 × d_model).
    dropout : float
    max_seq_len : int
        Maximum trajectory length supported by the step embedding.
    """

    def __init__(
        self,
        n_pois: int,
        grid_size: tuple[int, int],
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 100,
    ) -> None:
        super().__init__()

        self.n_pois = n_pois
        self.d_model = d_model
        n_rows, n_cols = grid_size
        n_cells = n_rows * n_cols  # vocabulary size for grid positions

        # ------------------------------------------------------------------
        # Input encoders
        # ------------------------------------------------------------------

        # Grid position embedding: index 0 = padding token (never attended to)
        self.pos_embedding = nn.Embedding(
            n_cells + 1, d_model, padding_idx=0
        )

        # Temporal position embedding: captures step order within the prefix
        self.step_embedding = nn.Embedding(max_seq_len, d_model)

        # World state projection: binary N-vector → single d_model-dim token
        self.ws_proj = nn.Sequential(
            nn.Linear(n_pois, d_model),
            nn.LayerNorm(d_model),
        )

        # Learnable [CLS] token prepended to every sequence
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ------------------------------------------------------------------
        # Transformer encoder
        # ------------------------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,          # pre-norm ("pre-LN") for training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,   # avoids shape issues with padding
        )

        # ------------------------------------------------------------------
        # Prediction heads
        # ------------------------------------------------------------------
        self.goal_head    = nn.Linear(d_model, n_pois)
        self.belief_head  = nn.Linear(d_model, n_pois)
        self.desire_head  = nn.Linear(d_model, n_pois)

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        trajectory: torch.Tensor,
        world_state: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        trajectory : (B, T) long
            Flat position indices for each trajectory step, 1-indexed.
            Padding positions should contain 0.
        world_state : (B, N) float
            Binary POI availability vector W.
        padding_mask : (B, T) bool, optional
            True where trajectory is padding (should be ignored by attention).

        Returns
        -------
        dict with keys 'goal_logits', 'belief_logits', 'desire_logits',
        each of shape (B, N).
        """
        B, T = trajectory.shape

        # --- Encode trajectory steps ---
        traj_emb = self.pos_embedding(trajectory)        # (B, T, d)
        steps = torch.arange(T, device=trajectory.device).unsqueeze(0)  # (1, T)
        traj_emb = traj_emb + self.step_embedding(steps) # (B, T, d)

        # --- Encode world state as a single token ---
        ws_tok = self.ws_proj(world_state).unsqueeze(1)  # (B, 1, d)

        # --- Prepend [CLS] and [WS] tokens ---
        cls = self.cls_token.expand(B, -1, -1)           # (B, 1, d)
        seq = torch.cat([cls, ws_tok, traj_emb], dim=1)  # (B, 2+T, d)

        # --- Build key padding mask for the full sequence ---
        # CLS and WS tokens are never masked (False = attend)
        if padding_mask is not None:
            prefix = torch.zeros(B, 2, dtype=torch.bool, device=trajectory.device)
            full_mask = torch.cat([prefix, padding_mask], dim=1)  # (B, 2+T)
        else:
            full_mask = None

        # --- Transformer ---
        out = self.transformer(seq, src_key_padding_mask=full_mask)  # (B, 2+T, d)

        # --- Pool via [CLS] token ---
        cls_out = out[:, 0]  # (B, d)

        return {
            "goal_logits":    self.goal_head(cls_out),    # (B, N)
            "belief_logits":  self.belief_head(cls_out),  # (B, N)
            "desire_logits":  self.desire_head(cls_out),  # (B, N)
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_config(cls, model_cfg: dict, sim_cfg: dict) -> "BDITransformer":
        """
        Construct a BDITransformer from the transformer config dict and the
        simulation metadata (for n_pois and grid_size).
        """
        return cls(
            n_pois=sim_cfg["n_pois"],
            grid_size=tuple(sim_cfg["grid_size"]),
            d_model=model_cfg["d_model"],
            n_heads=model_cfg["n_heads"],
            n_layers=model_cfg["n_layers"],
            d_ff=model_cfg["d_ff"],
            dropout=model_cfg["dropout"],
            max_seq_len=model_cfg["max_seq_len"],
        )
