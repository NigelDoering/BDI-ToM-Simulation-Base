"""
Shared Data Pipeline
====================
Every model in this project uses the same dataset class and collate function.
This guarantees that all models receive identical inputs and that accuracy
comparisons are fair.

Design
------
BDIDataset
  Wraps a list of episode dicts (loaded from a split JSON file).
  At __getitem__ time it samples a prefix length (random during training,
  fixed during evaluation) and returns a dict of tensors.

  Position encoding
    Each grid cell (row, col) is mapped to a 1-indexed flat integer:
      pos_idx = row * n_cols + col + 1
    Index 0 is the padding token (never attended to by the transformer).

  Prefix sampling
    Training  : prefix_len ~ Uniform[round(T * min_pf), round(T * max_pf)]
    Evaluation: prefix_len = round(T * fixed_pf)   (deterministic)

collate_fn
  Pads variable-length trajectory tensors within a batch to the batch's
  maximum trajectory length.  Returns a boolean padding_mask (True = pad)
  aligned with the trajectory axis.

build_loaders
  Factory that creates a DataLoader for a named split and, for evaluation,
  one BDIDataset per prefix fraction (no DataLoader — the Trainer calls
  evaluate() with a dataset, not a loader, to avoid re-creating workers).
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BDIDataset(Dataset):
    """
    Parameters
    ----------
    episodes : list of episode dicts
        As loaded from a split JSON file.
    n_cols : int
        Number of grid columns (needed to compute flat position index).
    min_prefix_frac, max_prefix_frac : float
        Used when fixed_prefix_frac is None (training mode).
    fixed_prefix_frac : float or None
        When set, every item uses this exact fraction of its trajectory.
        Used for evaluation at a specific observation level.
    seed : int
        Seed for the per-dataset RNG (ensures reproducible val evaluation).
    """

    def __init__(
        self,
        episodes: List[Dict[str, Any]],
        n_cols: int,
        min_prefix_frac: float = 0.20,
        max_prefix_frac: float = 0.80,
        fixed_prefix_frac: Optional[float] = None,
        seed: int = 0,
    ) -> None:
        self.episodes = episodes
        self.n_cols = n_cols
        self.min_pf = min_prefix_frac
        self.max_pf = max_prefix_frac
        self.fixed_pf = fixed_prefix_frac
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep = self.episodes[idx]
        traj: List[List[int]] = ep["trajectory"]
        T = len(traj)

        # --- Prefix length ---
        if self.fixed_pf is not None:
            prefix_len = max(1, round(T * self.fixed_pf))
        else:
            lo = max(1, round(T * self.min_pf))
            hi = max(lo, round(T * self.max_pf))
            prefix_len = self.rng.randint(lo, hi)

        prefix = traj[:prefix_len]

        # --- Trajectory tensor (1-indexed flat position, 0 = padding) ---
        pos_indices = [r * self.n_cols + c + 1 for r, c in prefix]
        traj_tensor = torch.tensor(pos_indices, dtype=torch.long)

        return {
            "trajectory": traj_tensor,                                          # (prefix_len,)
            "world_state": torch.tensor(ep["world_state"], dtype=torch.float32), # (N,)
            "goal":        torch.tensor(ep["goal_idx"],   dtype=torch.long),     # scalar
            "belief":      torch.tensor(ep["belief"],     dtype=torch.float32),  # (N,)
            "desires":     torch.tensor(ep["desires"],    dtype=torch.float32),  # (N,)
            # Diagnostics (not used by loss, but useful for per-condition eval)
            "prefix_frac":       torch.tensor(prefix_len / T,          dtype=torch.float32),
            "is_veridical":      torch.tensor(ep["is_veridical"],       dtype=torch.bool),
            "is_novel_agent":    torch.tensor(ep["is_novel_agent"],     dtype=torch.bool),
            "hamming_distance":  torch.tensor(ep["hamming_distance"],   dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Pad variable-length trajectory tensors to the longest in the batch.
    Returns a padding_mask tensor (True = padding, ignore during attention).
    """
    B = len(batch)
    max_len = max(b["trajectory"].shape[0] for b in batch)

    padded_traj = torch.zeros(B, max_len, dtype=torch.long)
    padding_mask = torch.ones(B, max_len, dtype=torch.bool)   # True = padded

    for i, b in enumerate(batch):
        L = b["trajectory"].shape[0]
        padded_traj[i, :L] = b["trajectory"]
        padding_mask[i, :L] = False                            # not padded

    return {
        "trajectory":      padded_traj,
        "padding_mask":    padding_mask,
        "world_state":     torch.stack([b["world_state"]    for b in batch]),
        "goal":            torch.stack([b["goal"]           for b in batch]),
        "belief":          torch.stack([b["belief"]         for b in batch]),
        "desires":         torch.stack([b["desires"]        for b in batch]),
        # diagnostics
        "prefix_frac":     torch.stack([b["prefix_frac"]      for b in batch]),
        "is_veridical":    torch.stack([b["is_veridical"]     for b in batch]),
        "is_novel_agent":  torch.stack([b["is_novel_agent"]   for b in batch]),
        "hamming_distance":torch.stack([b["hamming_distance"] for b in batch]),
    }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def load_split(datasets_dir: str, split_name: str) -> List[Dict]:
    path = os.path.join(datasets_dir, f"{split_name}.json")
    with open(path) as f:
        return json.load(f)


def load_metadata(datasets_dir: str) -> Dict:
    with open(os.path.join(datasets_dir, "metadata.json")) as f:
        return json.load(f)


def build_train_loader(
    datasets_dir: str,
    data_cfg: dict,
    batch_size: int,
    num_workers: int = 0,
    seed: int = 0,
) -> DataLoader:
    """DataLoader for the training split with random prefix sampling."""
    metadata = load_metadata(datasets_dir)
    n_cols = metadata["environment"]["grid_size"][1]

    episodes = load_split(datasets_dir, "train")
    dataset = BDIDataset(
        episodes=episodes,
        n_cols=n_cols,
        min_prefix_frac=data_cfg["min_prefix_frac"],
        max_prefix_frac=data_cfg["max_prefix_frac"],
        fixed_prefix_frac=None,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )


def build_eval_datasets(
    datasets_dir: str,
    eval_prefix_fracs: List[float],
    splits: Optional[List[str]] = None,
    seed: int = 0,
) -> Dict[str, Dict[float, BDIDataset]]:
    """
    Build eval datasets for each split × prefix fraction combination.

    Returns
    -------
    {split_name: {prefix_frac: BDIDataset}}
    e.g. {"val": {0.1: ds, 0.2: ds, ...}, "test_c1": {...}, ...}
    """
    metadata = load_metadata(datasets_dir)
    n_cols = metadata["environment"]["grid_size"][1]

    if splits is None:
        splits = ["val", "test_c1", "test_c2", "test_c3", "test_c4"]

    result: Dict[str, Dict[float, BDIDataset]] = {}
    for split in splits:
        episodes = load_split(datasets_dir, split)
        result[split] = {
            pf: BDIDataset(
                episodes=episodes,
                n_cols=n_cols,
                fixed_prefix_frac=pf,
                seed=seed,
            )
            for pf in eval_prefix_fracs
        }
    return result
