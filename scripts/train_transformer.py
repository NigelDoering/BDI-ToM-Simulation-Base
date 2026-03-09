"""
train_transformer.py
====================
Entry point for training the BDI Transformer (entangled baseline).

Usage
-----
  # From the project root (with venv active):
  python scripts/train_transformer.py

  # Override common settings:
  python scripts/train_transformer.py \\
    --datasets_dir datasets/run_1 \\
    --n_epochs 100 \\
    --batch_size 128 \\
    --lr 5e-4 \\
    --run_name my_experiment

  # Dry run (no WandB, no checkpoints):
  python scripts/train_transformer.py --dry_run --n_epochs 2

CLI flags
---------
  --sim_config      Path to simulation YAML  (default: config/default.yaml)
  --model_config    Path to model YAML       (default: config/transformer.yaml)
  --datasets_dir    Override datasets path   (default: from model_config)
  --n_epochs        Override number of epochs
  --batch_size      Override batch size
  --lr              Override learning rate
  --seed            Random seed
  --run_name        WandB run name (default: auto-generated)
  --dry_run         Skip WandB init; useful for quick local testing
  --resume          Path to checkpoint .pt to resume from
  --device          'cpu', 'cuda', 'mps', or 'auto' (default: auto)
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import yaml

# Make project root importable when called as `python scripts/...`
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.transformer import BDITransformer
from training.dataloader import build_eval_datasets, build_train_loader, load_metadata
from training.losses import BDILoss
from training.trainer import Trainer


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(requested)


def build_scheduler(optimizer, cfg: dict, n_train_steps_per_epoch: int):
    """
    Linear warmup → cosine annealing.
    Implemented as SequentialLR over two sub-schedulers.
    """
    train_cfg = cfg["training"]
    n_epochs = train_cfg["n_epochs"]
    warmup = train_cfg.get("warmup_epochs", 5)

    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-3,
        end_factor=1.0,
        total_iters=warmup,
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=n_epochs - warmup,
        eta_min=cfg["training"]["lr"] * 0.01,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[warmup],
    )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train BDI Transformer baseline.")
    p.add_argument("--sim_config",   default=os.path.join(ROOT, "config", "default.yaml"))
    p.add_argument("--model_config", default=os.path.join(ROOT, "config", "transformer.yaml"))
    p.add_argument("--datasets_dir", default=None)
    p.add_argument("--n_epochs",     type=int,   default=None)
    p.add_argument("--batch_size",   type=int,   default=None)
    p.add_argument("--lr",           type=float, default=None)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--run_name",     default=None)
    p.add_argument("--dry_run",      action="store_true",
                   help="Disable WandB and checkpointing; useful for smoke-testing.")
    p.add_argument("--resume",       default=None,
                   help="Path to a checkpoint .pt file to resume training from.")
    p.add_argument("--device",       default="auto")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # --- Load configs and apply CLI overrides ---
    model_cfg = load_cfg(args.model_config)
    if args.datasets_dir is not None:
        model_cfg["data"]["datasets_dir"] = args.datasets_dir
    if args.n_epochs is not None:
        model_cfg["training"]["n_epochs"] = args.n_epochs
    if args.batch_size is not None:
        model_cfg["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        model_cfg["training"]["lr"] = args.lr

    datasets_dir = os.path.join(ROOT, model_cfg["data"]["datasets_dir"])
    metadata = load_metadata(datasets_dir)
    sim_cfg = metadata["environment"]   # grid_size, n_pois, etc.

    torch.manual_seed(args.seed)
    device = select_device(args.device)

    print(f"\nDevice : {device}")
    print(f"Dataset: {datasets_dir}")

    # --- Data loaders ---
    train_cfg = model_cfg["training"]
    data_cfg  = model_cfg["data"]
    eval_fracs = data_cfg["eval_prefix_fracs"]

    train_loader = build_train_loader(
        datasets_dir=datasets_dir,
        data_cfg=data_cfg,
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg.get("num_workers", 0),
        seed=args.seed,
    )

    # Eval datasets: val + all four test conditions + train prefix curve
    eval_splits = ["val", "test_c1", "test_c2", "test_c3", "test_c4"]
    eval_datasets = build_eval_datasets(
        datasets_dir=datasets_dir,
        eval_prefix_fracs=eval_fracs,
        splits=eval_splits,
        seed=args.seed,
    )

    # --- Model ---
    model = BDITransformer.from_config(model_cfg["model"], sim_cfg).to(device)
    print(f"Parameters: {model.count_parameters():,}")

    # --- Optimiser & scheduler ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = build_scheduler(optimizer, model_cfg, len(train_loader))

    # --- Resume from checkpoint ---
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if ckpt.get("scheduler_state") and scheduler:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        print(f"Resumed from {args.resume}")

    # --- Loss ---
    lw = train_cfg["loss_weights"]
    loss_fn = BDILoss(
        w_goal=lw["goal"],
        w_belief=lw["belief"],
        w_desire=lw["desire"],
    )

    # --- WandB ---
    wandb_run = None
    if not args.dry_run:
        try:
            import wandb
            wandb_cfg = model_cfg.get("wandb", {})
            run_name = args.run_name or model_cfg["model"]["name"]
            wandb_run = wandb.init(
                project=wandb_cfg.get("project", "simulation-tom"),
                entity=wandb_cfg.get("entity", None),
                name=run_name,
                config={
                    "model": model_cfg["model"],
                    "training": train_cfg,
                    "data": data_cfg,
                    "sim": sim_cfg,
                    "seed": args.seed,
                },
            )
            if wandb_cfg.get("watch_model", False):
                wandb.watch(model, log="all", log_freq=100)
        except Exception as e:
            print(f"WandB init failed ({e}). Continuing without logging.")

    # --- Train ---
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        eval_datasets=eval_datasets,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=model_cfg,
        device=device,
        wandb_run=wandb_run if not args.dry_run else None,
    )

    try:
        trainer.train()
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
