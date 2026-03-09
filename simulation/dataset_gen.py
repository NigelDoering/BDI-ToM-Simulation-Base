"""
Dataset Generator
=================
Generates the full BDI-ToM dataset with train / val / test splits covering
all four compositionality test conditions.

Split definitions
-----------------
  train     — familiar agents, mixed veridical + false beliefs
  val       — familiar agents, mixed beliefs (held-out episodes)
  test_c1   — familiar agents, veridical beliefs   (baseline)
  test_c2   — familiar agents, false beliefs       (belief generalisation)
  test_c3   — novel agents,   veridical beliefs    (desire generalisation)
  test_c4   — novel agents,   false beliefs        (compositional gen.)

Output layout
-------------
  datasets/
  ├── metadata.json          environment config, world states, agent profiles
  ├── train.json
  ├── val.json
  ├── test_c1.json
  ├── test_c2.json
  ├── test_c3.json
  └── test_c4.json

Each split file is a JSON array of episode dicts (see episode.Episode.to_dict).
metadata.json is a single object containing everything needed to reconstruct
the environment and agent pools without re-running generation.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from .agent import BDIAgent, create_agents
from .environment import GridEnvironment
from .episode import Episode, run_episode


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_dataset(cfg: dict) -> Dict[str, List[dict]]:
    """
    Generate the complete dataset according to `cfg`.

    Parameters
    ----------
    cfg : dict
        Parsed YAML config (typically loaded from config/default.yaml).

    Returns
    -------
    splits : dict mapping split name → list of episode dicts
    """
    seed = cfg.get("seed", 42)
    rng = np.random.default_rng(seed)

    env_cfg = cfg["environment"]
    agent_cfg = cfg["agents"]
    belief_cfg = cfg["beliefs"]
    data_cfg = cfg["dataset"]

    # ------------------------------------------------------------------
    # 1. Build environment
    # ------------------------------------------------------------------
    env = GridEnvironment(
        grid_size=tuple(env_cfg["grid_size"]),
        n_pois=env_cfg["n_pois"],
        n_world_states=env_cfg["n_world_states"],
        min_open_pois=env_cfg["min_open_pois"],
        rng=rng,
    )

    # ------------------------------------------------------------------
    # 2. Create agent pools
    # ------------------------------------------------------------------
    n_familiar = agent_cfg["n_familiar"]
    n_novel = agent_cfg["n_novel"]
    alpha = agent_cfg["dirichlet_alpha"]

    familiar_agents = create_agents(
        n=n_familiar,
        n_pois=env.n_pois,
        alpha=alpha,
        rng=rng,
        is_novel=False,
        start_id=0,
    )
    novel_agents = create_agents(
        n=n_novel,
        n_pois=env.n_pois,
        alpha=alpha,
        rng=rng,
        is_novel=True,
        start_id=n_familiar,
    )

    rho = belief_cfg["false_belief_rho"]
    false_frac = belief_cfg["false_belief_fraction"]

    n_train = data_cfg["n_train"]
    n_val = data_cfg["n_val"]
    n_test = data_cfg["n_test_per_condition"]
    min_steps = env_cfg.get("min_start_goal_steps", 0)

    # ------------------------------------------------------------------
    # 3. Generate episodes per split
    # ------------------------------------------------------------------
    episode_id = 0

    def _make_episodes(
        split: str,
        condition: int,
        agents: List[BDIAgent],
        n: int,
        false_fraction: float,
    ) -> List[dict]:
        nonlocal episode_id

        # Determine veridical/false counts
        n_false = int(round(n * false_fraction))
        n_veridical = n - n_false

        # Build a shuffled list of (veridical flag, world_state_idx)
        veridical_flags = [True] * n_veridical + [False] * n_false
        rng.shuffle(veridical_flags)

        episodes: List[dict] = []
        for veridical in tqdm(veridical_flags, desc=f"  {split}", leave=False):
            ws_idx = int(rng.integers(0, env.n_world_states))
            ws = env.world_states[ws_idx].copy()
            agent = agents[int(rng.integers(0, len(agents)))]

            ep = run_episode(
                episode_id=episode_id,
                agent=agent,
                env=env,
                world_state=ws,
                world_state_idx=ws_idx,
                veridical=veridical,
                rho=rho,
                split=split,
                condition=condition,
                rng=rng,
                min_start_goal_steps=min_steps,
            )
            if ep is not None:
                episodes.append(ep.to_dict())
                episode_id += 1

        return episodes

    def _make_test_condition(
        split: str,
        condition: int,
        agents: List[BDIAgent],
        n: int,
        veridical: bool,
    ) -> List[dict]:
        """Test splits have a single fixed belief type (veridical or false)."""
        nonlocal episode_id
        episodes: List[dict] = []

        for _ in tqdm(range(n), desc=f"  {split}", leave=False):
            ws_idx = int(rng.integers(0, env.n_world_states))
            ws = env.world_states[ws_idx].copy()
            agent = agents[int(rng.integers(0, len(agents)))]

            ep = run_episode(
                episode_id=episode_id,
                agent=agent,
                env=env,
                world_state=ws,
                world_state_idx=ws_idx,
                veridical=veridical,
                rho=rho,
                split=split,
                condition=condition,
                rng=rng,
                min_start_goal_steps=min_steps,
            )
            if ep is not None:
                episodes.append(ep.to_dict())
                episode_id += 1

        return episodes

    print("Generating dataset splits…")

    splits: Dict[str, List[dict]] = {}

    # Train: familiar agents, mixed beliefs (conditions 1+2 combined)
    splits["train"] = _make_episodes(
        split="train",
        condition=0,        # mixed; individual episodes store is_veridical flag
        agents=familiar_agents,
        n=n_train,
        false_fraction=false_frac,
    )

    # Validation: same distribution as train
    splits["val"] = _make_episodes(
        split="val",
        condition=0,
        agents=familiar_agents,
        n=n_val,
        false_fraction=false_frac,
    )

    # Test condition 1: familiar, veridical
    splits["test_c1"] = _make_test_condition(
        split="test_c1",
        condition=1,
        agents=familiar_agents,
        n=n_test,
        veridical=True,
    )

    # Test condition 2: familiar, false
    splits["test_c2"] = _make_test_condition(
        split="test_c2",
        condition=2,
        agents=familiar_agents,
        n=n_test,
        veridical=False,
    )

    # Test condition 3: novel, veridical
    splits["test_c3"] = _make_test_condition(
        split="test_c3",
        condition=3,
        agents=novel_agents,
        n=n_test,
        veridical=True,
    )

    # Test condition 4: novel, false (compositional generalisation — critical)
    splits["test_c4"] = _make_test_condition(
        split="test_c4",
        condition=4,
        agents=novel_agents,
        n=n_test,
        veridical=False,
    )

    # ------------------------------------------------------------------
    # 4. Build metadata
    # ------------------------------------------------------------------
    metadata = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": seed,
        "config": cfg,
        "environment": env.to_dict(),
        "agents": {
            "familiar": [a.to_dict() for a in familiar_agents],
            "novel": [a.to_dict() for a in novel_agents],
        },
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "conditions": {
            "1": "familiar agents, veridical beliefs (baseline)",
            "2": "familiar agents, false beliefs (belief generalisation)",
            "3": "novel agents, veridical beliefs (desire generalisation)",
            "4": "novel agents, false beliefs (compositional generalisation — critical test)",
        },
    }

    return splits, metadata


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_dataset(
    splits: Dict[str, List[dict]],
    metadata: dict,
    output_dir: str,
) -> None:
    """
    Write each split as a JSON array and metadata as a separate JSON object.

    Parameters
    ----------
    splits : dict of split_name → list of episode dicts
    metadata : dict
    output_dir : str
        Directory to write into (will be created if absent).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Metadata
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Wrote {meta_path}")

    # Splits
    for split_name, episodes in splits.items():
        path = os.path.join(output_dir, f"{split_name}.json")
        with open(path, "w") as f:
            json.dump(episodes, f)
        print(f"  Wrote {path}  ({len(episodes)} episodes)")
