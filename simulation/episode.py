"""
Episode
=======
Dataclass representing a single simulation episode and the `run_episode`
function that produces one.

Episode fields align exactly with the tuple (τ, W, Ba, Da, g*) from the
problem statement, extended with metadata needed for dataset splits and
analysis.

Condition mapping
-----------------
  1  Familiar agent, veridical belief   (dH = 0)
  2  Familiar agent, false belief       (dH > 0)
  3  Novel agent,    veridical belief   (dH = 0)
  4  Novel agent,    false belief       (dH > 0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .agent import BDIAgent
from .environment import GridEnvironment
from .pathfinding import bfs


@dataclass
class Episode:
    """
    A complete simulation episode.

    Attributes
    ----------
    episode_id : int
    agent_id : int
    condition : int
        1–4 per the compositionality test conditions.
    split : str
        'train', 'val', 'test_c1', 'test_c2', 'test_c3', or 'test_c4'.
    world_state : np.ndarray, shape (N,), dtype int8
        Ground-truth POI availability vector W.
    world_state_idx : int
        Index into env.world_states (for lookup / grouping in analysis).
    belief : np.ndarray, shape (N,), dtype int8
        Agent's (potentially false) belief vector Ba.
    desires : np.ndarray, shape (N,), dtype float64
        Agent's stable preference distribution Da (sums to 1).
    goal_idx : int
        Index of the selected goal POI g*.
    goal_position : (int, int)
        Grid coordinates of the goal POI.
    start_position : (int, int)
        Agent's starting grid cell.
    trajectory : list of (int, int)
        Full sequence of grid cells from start to goal (inclusive).
    trajectory_length : int
        Number of steps (= len(trajectory) - 1).
    hamming_distance : int
        dH(Ba, W): number of differing bits between belief and world state.
    is_veridical : bool
        True iff hamming_distance == 0.
    is_novel_agent : bool
        True iff the agent was drawn from the novel pool.
    """

    episode_id: int
    agent_id: int
    condition: int
    split: str

    world_state: np.ndarray
    world_state_idx: int
    belief: np.ndarray
    desires: np.ndarray

    goal_idx: int
    goal_position: Tuple[int, int]
    start_position: Tuple[int, int]
    trajectory: List[Tuple[int, int]]
    trajectory_length: int

    hamming_distance: int
    is_veridical: bool
    is_novel_agent: bool

    def to_dict(self) -> dict:
        """
        Return a JSON-serialisable representation.

        Trajectory and positions are stored as lists of [row, col] pairs.
        All numpy arrays are converted to Python lists so standard `json`
        can serialise without a custom encoder.
        """
        return {
            "episode_id": self.episode_id,
            "agent_id": self.agent_id,
            "condition": self.condition,
            "split": self.split,
            # --- world ---
            "world_state": self.world_state.tolist(),
            "world_state_idx": self.world_state_idx,
            # --- mental state ---
            "belief": self.belief.tolist(),
            "desires": self.desires.tolist(),
            # --- goal & trajectory ---
            "goal_idx": self.goal_idx,
            "goal_position": list(self.goal_position),
            "start_position": list(self.start_position),
            "trajectory": [list(p) for p in self.trajectory],
            "trajectory_length": self.trajectory_length,
            # --- diagnostics ---
            "hamming_distance": self.hamming_distance,
            "is_veridical": self.is_veridical,
            "is_novel_agent": self.is_novel_agent,
        }


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    episode_id: int,
    agent: BDIAgent,
    env: GridEnvironment,
    world_state: np.ndarray,
    world_state_idx: int,
    veridical: bool,
    rho: float,
    split: str,
    condition: int,
    rng: np.random.Generator,
    min_start_goal_steps: int = 0,
) -> Optional[Episode]:
    """
    Simulate one episode.

    Parameters
    ----------
    min_start_goal_steps : int
        Minimum Manhattan distance required between start and goal.
        Prevents trivially short trajectories that could be solved by simple
        direction extrapolation rather than mental-state reasoning.

    Returns
    -------
    Episode on success, None if BFS fails (should never happen on open grid).
    """
    # 1. Belief sampling
    if veridical:
        belief = agent.sample_veridical_belief(world_state)
    else:
        belief = agent.sample_false_belief(world_state, rho, rng)

    hamming = int(np.sum(belief != world_state))

    # 2. Goal selection (Equation 5)
    goal_idx = agent.select_goal(belief, rng)
    goal_pos = env.pois[goal_idx].position

    # 3. Start position — enforcing minimum distance from goal
    start_pos = env.sample_start(goal=goal_pos, min_steps=min_start_goal_steps)

    # 4. Trajectory via BFS
    traj = bfs(env.grid_size, start_pos, goal_pos)
    if traj is None:
        return None  # should never happen

    return Episode(
        episode_id=episode_id,
        agent_id=agent.agent_id,
        condition=condition,
        split=split,
        world_state=world_state,
        world_state_idx=world_state_idx,
        belief=belief,
        desires=agent.desires,
        goal_idx=goal_idx,
        goal_position=goal_pos,
        start_position=start_pos,
        trajectory=traj,
        trajectory_length=len(traj) - 1,
        hamming_distance=hamming,
        is_veridical=(hamming == 0),
        is_novel_agent=agent.is_novel,
    )
