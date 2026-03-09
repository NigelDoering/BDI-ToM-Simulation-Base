"""
generate_dataset.py
===================
Entry point for generating the BDI Theory of Mind simulation dataset.

Usage
-----
  # From the project root:
  python scripts/generate_dataset.py

  # Override any config value:
  python scripts/generate_dataset.py --config config/default.yaml \
      --seed 7 \
      --n_train 50000 \
      --output_dir datasets/large

CLI arguments
-------------
  --config      Path to YAML config file (default: config/default.yaml)
  --seed        Random seed (overrides value in config)
  --n_train     Training episodes (overrides config)
  --n_val       Validation episodes (overrides config)
  --n_test      Test episodes per condition (overrides config)
  --output_dir  Output directory (overrides config)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import yaml

# Make project root importable when called as `python scripts/...`
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulation.dataset_gen import generate_dataset, save_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate BDI-ToM simulation dataset."
    )
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "..", "config", "default.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument("--seed",       type=int,   default=None)
    parser.add_argument("--n_train",    type=int,   default=None)
    parser.add_argument("--n_val",      type=int,   default=None)
    parser.add_argument("--n_test",     type=int,   default=None)
    parser.add_argument("--output_dir", type=str,   default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load config
    config_path = os.path.abspath(args.config)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Apply CLI overrides
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.n_train is not None:
        cfg["dataset"]["n_train"] = args.n_train
    if args.n_val is not None:
        cfg["dataset"]["n_val"] = args.n_val
    if args.n_test is not None:
        cfg["dataset"]["n_test_per_condition"] = args.n_test
    if args.output_dir is not None:
        cfg["dataset"]["output_dir"] = args.output_dir

    output_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", cfg["dataset"]["output_dir"])
    )

    print("=" * 60)
    print("  BDI Theory of Mind — Dataset Generator")
    print("=" * 60)
    print(f"  Config : {config_path}")
    print(f"  Seed   : {cfg['seed']}")
    print(f"  Grid   : {cfg['environment']['grid_size']}")
    print(f"  POIs   : {cfg['environment']['n_pois']}")
    print(f"  World states (K) : {cfg['environment']['n_world_states']}")
    print(f"  Familiar agents  : {cfg['agents']['n_familiar']}")
    print(f"  Novel agents     : {cfg['agents']['n_novel']}")
    print(f"  False-belief ρ   : {cfg['beliefs']['false_belief_rho']}")
    print(f"  Train / Val / Test-per-cond : "
          f"{cfg['dataset']['n_train']} / "
          f"{cfg['dataset']['n_val']} / "
          f"{cfg['dataset']['n_test_per_condition']}")
    print(f"  Output : {output_dir}")
    print("=" * 60)

    t0 = time.time()
    splits, metadata = generate_dataset(cfg)
    elapsed = time.time() - t0

    print(f"\nGeneration complete in {elapsed:.1f}s")
    print("\nSplit sizes:")
    for name, eps in splits.items():
        print(f"  {name:10s}  {len(eps):>6} episodes")

    print(f"\nSaving to {output_dir}…")
    save_dataset(splits, metadata, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
