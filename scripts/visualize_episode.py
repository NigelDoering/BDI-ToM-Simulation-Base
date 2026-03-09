"""
visualize_episode.py
====================
Render one or more episodes from a generated dataset as animated GIFs.

Usage
-----
  # Render episode 0 from train split (default):
  python scripts/visualize_episode.py

  # Render a specific episode from a specific split:
  python scripts/visualize_episode.py --split test_c4 --episode_idx 5

  # Render N random episodes from a split:
  python scripts/visualize_episode.py --split test_c2 --n_random 10

  # Render all four test conditions, one episode each:
  python scripts/visualize_episode.py --all_conditions

CLI arguments
-------------
  --datasets_dir   Directory containing split JSONs + metadata.json
  --output_dir     Where to save GIFs (default: outputs/)
  --split          Which split file to draw from (default: train)
  --episode_idx    Index of episode within the split to render
  --n_random       Render N randomly selected episodes from the split
  --all_conditions Render one random episode from each test condition (C1–C4)
  --fps            GIF frames per second (default: from config)
  --dpi            Output DPI (default: from config)
  --no_beliefs     Disable belief ring overlay
  --no_desires     Disable desire-scaled marker sizes
  --seed           Random seed for --n_random / --all_conditions selection
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from visualization.trajectory_viz import render_episode_from_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render episode trajectories as animated GIFs."
    )
    parser.add_argument(
        "--datasets_dir",
        default=os.path.join(os.path.dirname(__file__), "..", "datasets"),
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(os.path.dirname(__file__), "..", "outputs"),
    )
    parser.add_argument("--split",          default="train")
    parser.add_argument("--episode_idx",    type=int, default=0)
    parser.add_argument("--n_random",       type=int, default=None)
    parser.add_argument("--all_conditions", action="store_true")
    parser.add_argument("--fps",            type=int, default=4)
    parser.add_argument("--dpi",            type=int, default=80)
    parser.add_argument("--no_beliefs",     action="store_true")
    parser.add_argument("--no_desires",     action="store_true")
    parser.add_argument("--seed",           type=int, default=42)
    return parser.parse_args()


def load_split(datasets_dir: str, split_name: str) -> list:
    path = os.path.join(datasets_dir, f"{split_name}.json")
    with open(path) as f:
        return json.load(f)


def load_metadata(datasets_dir: str) -> dict:
    path = os.path.join(datasets_dir, "metadata.json")
    with open(path) as f:
        return json.load(f)


def render(
    episode: dict,
    metadata: dict,
    output_dir: str,
    fps: int,
    dpi: int,
    show_beliefs: bool,
    show_desires: bool,
    label: str = "",
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    fname = (
        f"ep{episode['episode_id']}"
        f"_agent{episode['agent_id']}"
        f"_c{episode['condition']}"
        f"_{'veridical' if episode['is_veridical'] else 'false'}"
        f"{'_' + label if label else ''}"
        ".gif"
    )
    out_path = os.path.join(output_dir, fname)
    render_episode_from_json(
        episode=episode,
        metadata=metadata,
        output_path=out_path,
        fps=fps,
        dpi=dpi,
        show_beliefs=show_beliefs,
        show_desires=show_desires,
    )
    print(f"  Saved: {out_path}")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    datasets_dir = os.path.abspath(args.datasets_dir)
    output_dir = os.path.abspath(args.output_dir)

    metadata = load_metadata(datasets_dir)

    show_beliefs = not args.no_beliefs
    show_desires = not args.no_desires

    common = dict(
        metadata=metadata,
        output_dir=output_dir,
        fps=args.fps,
        dpi=args.dpi,
        show_beliefs=show_beliefs,
        show_desires=show_desires,
    )

    if args.all_conditions:
        # One random episode per test condition
        for cond in range(1, 5):
            split_name = f"test_c{cond}"
            episodes = load_split(datasets_dir, split_name)
            ep = random.choice(episodes)
            print(f"Rendering condition {cond} episode {ep['episode_id']}…")
            render(ep, label=f"cond{cond}", **common)

    elif args.n_random is not None:
        episodes = load_split(datasets_dir, args.split)
        chosen = random.sample(episodes, min(args.n_random, len(episodes)))
        for ep in chosen:
            print(f"Rendering episode {ep['episode_id']} from {args.split}…")
            render(ep, **common)

    else:
        episodes = load_split(datasets_dir, args.split)
        ep = episodes[args.episode_idx]
        print(f"Rendering episode {ep['episode_id']} from {args.split}…")
        render(ep, **common)

    print("Done.")


if __name__ == "__main__":
    main()
