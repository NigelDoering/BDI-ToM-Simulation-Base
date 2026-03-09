"""
Trainer
=======
Model-agnostic training loop for all BDI-ToM models.

The Trainer accepts any model that returns a dict with keys
'goal_logits', 'belief_logits', 'desire_logits' and processes
batches from the shared BDIDataset/collate_fn pipeline.

WandB logging structure
-----------------------
Per epoch:
  train/loss_{total,goal,belief,desire}
  val/loss_{total,goal,belief,desire}

Prefix-fraction accuracy curves (eval_prefix_fracs, default 6 fracs):
  val/{goal,belief,desire}_acc/pf{N}   — logged every epoch
  train/{goal,belief,desire}_acc/pf{N} — logged every eval_train_every epochs

End of training (test conditions):
  test_c{1,2,3,4}/{goal,belief,desire}_acc/pf{N}

Metric naming uses WandB's "/" grouping so each head's accuracy curve
appears as a separate panel with all prefix fractions overlaid.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .dataloader import BDIDataset, collate_fn
from .losses import BDILoss, belief_metrics, desire_accuracy, goal_accuracy


class Trainer:
    """
    Parameters
    ----------
    model : nn.Module
        Any model returning {goal_logits, belief_logits, desire_logits}.
    train_loader : DataLoader
        Training split loader (random prefix sampling).
    eval_datasets : dict
        {split_name: {prefix_frac: BDIDataset}}
        Built by training.dataloader.build_eval_datasets().
    loss_fn : BDILoss
    optimizer : torch.optim.Optimizer
    scheduler : optional LR scheduler (called every epoch after optimiser step)
    cfg : merged config dict (model + training sections)
    device : torch.device
    wandb_run : optional wandb run object (None → no WandB logging)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        eval_datasets: Dict[str, Dict[float, BDIDataset]],
        loss_fn: BDILoss,
        optimizer: torch.optim.Optimizer,
        scheduler,
        cfg: dict,
        device: torch.device,
        wandb_run=None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.eval_datasets = eval_datasets
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg
        self.device = device
        self.wandb_run = wandb_run

        train_cfg = cfg["training"]
        self.n_epochs = train_cfg["n_epochs"]
        self.grad_clip = train_cfg.get("grad_clip", 1.0)
        self.eval_train_every = train_cfg.get("eval_train_every", 5)
        self.checkpoint_dir = os.path.abspath(train_cfg["checkpoint_dir"])
        self.save_every = train_cfg.get("save_every", 10)
        self.save_best = train_cfg.get("save_best", True)
        self.eval_batch_size = train_cfg.get("batch_size", 64)
        self.eval_prefix_fracs: List[float] = cfg["data"]["eval_prefix_fracs"]
        self.num_workers = train_cfg.get("num_workers", 0)

        self.best_val_loss = float("inf")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        totals: Dict[str, float] = {"total": 0, "goal": 0, "belief": 0, "desire": 0}
        n_batches = 0

        for batch in tqdm(self.train_loader, desc=f"  Epoch {epoch}", leave=False):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            self.optimizer.zero_grad()
            outputs = self.model(
                trajectory=batch["trajectory"],
                world_state=batch["world_state"],
                padding_mask=batch["padding_mask"],
            )
            losses = self.loss_fn(outputs, batch)
            losses["total"].backward()

            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            for k in totals:
                totals[k] += losses[k].item()
            n_batches += 1

        return {k: v / n_batches for k, v in totals.items()}

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate_split_at_frac(
        self, dataset: BDIDataset, prefix_frac: float
    ) -> Dict[str, float]:
        """
        Run one full evaluation pass at a fixed prefix fraction.
        Returns loss dict + top-1 accuracies for all three heads.
        """
        self.model.eval()
        loader = DataLoader(
            dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        loss_totals = {"total": 0.0, "goal": 0.0, "belief": 0.0, "desire": 0.0}
        all_goal_logits, all_belief_logits, all_desire_logits = [], [], []
        all_goals, all_beliefs, all_desires = [], [], []
        all_world_states, all_is_veridical = [], []
        n_batches = 0

        for batch in loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            outputs = self.model(
                trajectory=batch["trajectory"],
                world_state=batch["world_state"],
                padding_mask=batch["padding_mask"],
            )
            losses = self.loss_fn(outputs, batch)
            for k in loss_totals:
                loss_totals[k] += losses[k].item()

            all_goal_logits.append(outputs["goal_logits"].cpu())
            all_belief_logits.append(outputs["belief_logits"].cpu())
            all_desire_logits.append(outputs["desire_logits"].cpu())
            all_goals.append(batch["goal"].cpu())
            all_beliefs.append(batch["belief"].cpu())
            all_desires.append(batch["desires"].cpu())
            all_world_states.append(batch["world_state"].cpu())
            all_is_veridical.append(batch["is_veridical"].cpu())
            n_batches += 1

        goal_logits   = torch.cat(all_goal_logits)
        belief_logits = torch.cat(all_belief_logits)
        desire_logits = torch.cat(all_desire_logits)
        goals         = torch.cat(all_goals)
        beliefs       = torch.cat(all_beliefs)
        desires       = torch.cat(all_desires)
        world_states  = torch.cat(all_world_states)
        is_veridical  = torch.cat(all_is_veridical)

        metrics = {k: v / n_batches for k, v in loss_totals.items()}
        metrics["goal_acc"]  = goal_accuracy(goal_logits, goals)
        metrics["desire_acc"] = desire_accuracy(desire_logits, desires)
        # Richer belief metrics — replaces the misleading single bit_acc number
        bm = belief_metrics(belief_logits, beliefs, world_states, is_veridical)
        metrics.update({f"belief_{k}": v for k, v in bm.items()})
        return metrics

    def evaluate_prefix_curve(
        self, split: str
    ) -> Dict[float, Dict[str, float]]:
        """
        Evaluate all prefix fractions for a named split.
        Returns {prefix_frac: metrics_dict}.
        """
        datasets_at_fracs = self.eval_datasets[split]
        return {
            pf: self.evaluate_split_at_frac(ds, pf)
            for pf, ds in datasets_at_fracs.items()
        }

    # ------------------------------------------------------------------
    # WandB logging helpers
    # ------------------------------------------------------------------

    def _log(self, metrics: dict, epoch: int) -> None:
        if self.wandb_run is not None:
            self.wandb_run.log(metrics, step=epoch)

    def _log_losses(self, losses: Dict[str, float], prefix: str, epoch: int) -> None:
        self._log({f"{prefix}/loss_{k}": v for k, v in losses.items()}, epoch)

    def _log_prefix_curve(
        self,
        curve: Dict[float, Dict[str, float]],
        split: str,
        epoch: int,
    ) -> None:
        """
        Log accuracy at each prefix fraction under grouped WandB metric names.
        E.g. "val/goal_acc/pf10", "val/belief_acc/pf50", etc.
        """
        log_dict = {}
        for pf, metrics in curve.items():
            pf_tag = f"pf{int(pf * 100):02d}"
            # Goal and desire top-1 accuracy
            log_dict[f"{split}/goal_acc/{pf_tag}"]           = metrics["goal_acc"]
            log_dict[f"{split}/desire_acc/{pf_tag}"]          = metrics["desire_acc"]
            # Belief: report false-episode accuracy (primary), plus Hamming distances
            log_dict[f"{split}/belief_acc_false/{pf_tag}"]   = metrics["belief_bit_acc_false"]
            log_dict[f"{split}/belief_acc_verid/{pf_tag}"]   = metrics["belief_bit_acc_veridical"]
            log_dict[f"{split}/belief_hamming/{pf_tag}"]     = metrics["belief_hamming_pred"]
            # Log the no-model baseline once per split so it's visible on the same panel
            log_dict[f"{split}/belief_ws_baseline/{pf_tag}"] = metrics["belief_ws_baseline_acc"]
        self._log(log_dict, epoch)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save(self, tag: str) -> None:
        path = os.path.join(self.checkpoint_dir, f"{tag}.pt")
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
                "cfg": self.cfg,
            },
            path,
        )

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        print(f"\n{'=' * 60}")
        print(f"  Training: {self.cfg['model']['name']}")
        print(f"  Device  : {self.device}")
        print(f"  Params  : {self.model.count_parameters():,}")
        print(f"  Epochs  : {self.n_epochs}")
        print(f"  Splits  : {list(self.eval_datasets.keys())}")
        print(f"{'=' * 60}\n")

        for epoch in range(1, self.n_epochs + 1):
            t0 = time.time()

            # --- Train ---
            train_losses = self.train_epoch(epoch)
            self._log_losses(train_losses, "train", epoch)

            if self.scheduler is not None:
                self.scheduler.step()

            # --- Val loss (every epoch) ---
            val_losses_by_frac = self.evaluate_prefix_curve("val")
            # Use median prefix frac (≈0.5) for the loss scalar
            mid_pf = sorted(self.eval_datasets["val"].keys())[len(self.eval_datasets["val"]) // 2]
            val_losses = {k: v for k, v in val_losses_by_frac[mid_pf].items()
                          if not k.endswith("_acc")}
            self._log_losses(val_losses, "val", epoch)

            # --- Val prefix-fraction accuracy curve (every epoch) ---
            self._log_prefix_curve(val_losses_by_frac, "val", epoch)

            # --- Train prefix-fraction accuracy curve (every N epochs) ---
            if epoch % self.eval_train_every == 0:
                train_curve = self.evaluate_prefix_curve("train") if "train" in self.eval_datasets else {}
                if train_curve:
                    self._log_prefix_curve(train_curve, "train", epoch)

            # --- Checkpoint: best val ---
            val_total = val_losses.get("total", float("inf"))
            if self.save_best and val_total < self.best_val_loss:
                self.best_val_loss = val_total
                self._save("best")

            # --- Checkpoint: periodic ---
            if epoch % self.save_every == 0:
                self._save(f"epoch_{epoch:04d}")

            elapsed = time.time() - t0
            mid_metrics = val_losses_by_frac[mid_pf]
            print(
                f"  Epoch {epoch:3d}/{self.n_epochs} | "
                f"train_loss={train_losses['total']:.4f} | "
                f"val_loss={val_total:.4f} | "
                f"goal={mid_metrics['goal_acc']:.3f} | "
                f"belief(false)={mid_metrics['belief_bit_acc_false']:.3f} "
                f"[ws={mid_metrics['belief_ws_baseline_acc']:.3f}] "
                f"H={mid_metrics['belief_hamming_pred']:.1f} | "
                f"desire={mid_metrics['desire_acc']:.3f} | "
                f"{elapsed:.1f}s"
            )

        # --- Final: evaluate all four test conditions ---
        print("\nEvaluating test conditions…")
        test_splits = [s for s in self.eval_datasets if s.startswith("test_")]
        for split in test_splits:
            curve = self.evaluate_prefix_curve(split)
            self._log_prefix_curve(curve, split, self.n_epochs)
            mid_pf = sorted(self.eval_datasets[split].keys())[len(self.eval_datasets[split]) // 2]
            m = curve[mid_pf]
            print(
                f"  {split}: goal={m['goal_acc']:.3f} | "
                f"belief(false)={m['belief_bit_acc_false']:.3f} "
                f"[ws_baseline={m['belief_ws_baseline_acc']:.3f}] "
                f"H={m['belief_hamming_pred']:.1f} | "
                f"desire={m['desire_acc']:.3f}"
            )

        self._save("final")
        print("\nTraining complete.")
