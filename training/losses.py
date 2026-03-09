"""
Multi-Head BDI Loss
===================
Computes the weighted combination of three supervision signals:

  Goal   : CrossEntropyLoss over N POI classes
           Measures how accurately the model predicts which POI the agent
           intends to reach.

  Belief : BCEWithLogitsLoss over N independent binary variables
           Measures per-POI accuracy of the predicted belief state B_a.
           Each bit is an independent Bernoulli; the agent may believe any
           subset of POIs to be open, unconstrained by the world state.

  Desire : KL divergence KL(true || predicted)
           Treats desires as a probability distribution over POIs and
           penalises divergence between the predicted and true distributions.
           We minimise KL(p_true || p_pred) = E_{p_true}[log p_true - log p_pred],
           which encourages the model to assign high probability wherever
           the true distribution does.

Loss weights (λ_goal, λ_belief, λ_desire) are configurable.  With equal
weights, goal and belief losses dominate early training because they are
classification problems with larger gradients.  Tune λ_desire upward if
desire predictions lag behind.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class BDILoss(nn.Module):
    """
    Parameters
    ----------
    w_goal, w_belief, w_desire : float
        Loss weights λ for each head.
    label_smoothing : float
        Applied to the goal cross-entropy.  Helps prevent overconfident goal
        predictions, especially for familiar agents with many trajectories.
    """

    def __init__(
        self,
        w_goal: float = 1.0,
        w_belief: float = 1.0,
        w_desire: float = 1.0,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.w_goal = w_goal
        self.w_belief = w_belief
        self.w_desire = w_desire
        self.label_smoothing = label_smoothing

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        outputs : dict
            Model outputs with keys 'goal_logits', 'belief_logits', 'desire_logits'.
            All shapes (B, N).
        targets : dict
            Batch dict with keys 'goal' (B,), 'belief' (B, N), 'desires' (B, N).

        Returns
        -------
        dict with keys 'total', 'goal', 'belief', 'desire' — all scalar tensors.
        """
        # --- Goal: N-class classification ---
        goal_loss = F.cross_entropy(
            outputs["goal_logits"],
            targets["goal"],
            label_smoothing=self.label_smoothing,
        )

        # --- Belief: N independent binary predictions ---
        belief_loss = F.binary_cross_entropy_with_logits(
            outputs["belief_logits"],
            targets["belief"],
        )

        # --- Desire: KL(true || predicted) ---
        # log_softmax is numerically stable and required by F.kl_div
        log_pred = F.log_softmax(outputs["desire_logits"], dim=-1)
        desire_loss = F.kl_div(
            log_pred,
            targets["desires"],          # target is already a valid distribution
            reduction="batchmean",
            log_target=False,
        )

        total = (
            self.w_goal   * goal_loss
            + self.w_belief * belief_loss
            + self.w_desire * desire_loss
        )

        return {
            "total":   total,
            "goal":    goal_loss,
            "belief":  belief_loss,
            "desire":  desire_loss,
        }


# ---------------------------------------------------------------------------
# Accuracy helpers  (used by Trainer — kept here to centralise metric logic)
# ---------------------------------------------------------------------------

@torch.no_grad()
def goal_accuracy(goal_logits: torch.Tensor, goal_targets: torch.Tensor) -> float:
    """
    Top-1 accuracy: fraction of episodes where argmax(logits) == true goal.
    """
    preds = goal_logits.argmax(dim=-1)
    return (preds == goal_targets).float().mean().item()


@torch.no_grad()
def belief_metrics(
    belief_logits: torch.Tensor,
    belief_targets: torch.Tensor,
    world_states: torch.Tensor,
    is_veridical: torch.Tensor,
) -> dict:
    """
    Comprehensive belief evaluation metrics.

    The naïve per-POI accuracy has a misleading floor of ~50% because
    beliefs are balanced (P(belief=1) ≈ 0.5 for ρ=0.3, P(W=1) ≈ 0.5).
    An untrained model with near-zero logits achieves ~50% by chance.

    Meaningful baselines:
      - Random predictor       : ~50%  (uninformative floor)
      - Predict world state W  : ~85%  (50% veridical → 100%, 50% false → ~70%)
      - Oracle                 : 100%

    The paper's key diagnostic is Hamming distance: how many bits does the
    predicted belief differ from the true belief?  Baseline = ρ·N ≈ 30.

    Returns
    -------
    dict with keys:
      bit_acc_all      : raw per-POI accuracy (misleading on its own)
      bit_acc_veridical: per-POI accuracy on veridical episodes only
      bit_acc_false    : per-POI accuracy on false-belief episodes only (the key metric)
      ws_baseline_acc  : accuracy of always predicting world state (no-model baseline)
      hamming_pred     : mean Hamming distance between prediction and true belief (↓ better)
      hamming_baseline : mean Hamming dist. of always predicting W (model must beat this)
    """
    B, N = belief_targets.shape
    preds_binary = (belief_logits.sigmoid() > 0.5).float()

    # Raw per-POI accuracy (all episodes)
    correct = (preds_binary == belief_targets)
    bit_acc_all = correct.float().mean().item()

    # Stratify by veridical / false
    if is_veridical.any():
        bit_acc_veridical = correct[is_veridical].float().mean().item()
    else:
        bit_acc_veridical = float("nan")

    false_mask = ~is_veridical
    if false_mask.any():
        bit_acc_false = correct[false_mask].float().mean().item()
    else:
        bit_acc_false = float("nan")

    # World-state baseline: what accuracy does "always predict W" achieve?
    ws_correct = (world_states == belief_targets)
    ws_baseline_acc = ws_correct.float().mean().item()

    # Hamming distance: number of bits wrong per episode (↓ is better)
    hamming_pred     = (preds_binary != belief_targets).float().sum(dim=-1).mean().item()
    hamming_baseline = (world_states  != belief_targets).float().sum(dim=-1).mean().item()

    return {
        "bit_acc_all":       bit_acc_all,
        "bit_acc_veridical": bit_acc_veridical,
        "bit_acc_false":     bit_acc_false,
        "ws_baseline_acc":   ws_baseline_acc,
        "hamming_pred":      hamming_pred,
        "hamming_baseline":  hamming_baseline,
    }


@torch.no_grad()
def desire_accuracy(desire_logits: torch.Tensor, desire_targets: torch.Tensor) -> float:
    """
    Top-1 accuracy: fraction of episodes where the model correctly identifies
    the agent's single most-preferred POI (argmax of the desire distribution).

    Random baseline: 1/N = 1%.  A useful model should substantially exceed this.
    """
    pred_top = desire_logits.argmax(dim=-1)
    true_top = desire_targets.argmax(dim=-1)
    return (pred_top == true_top).float().mean().item()
