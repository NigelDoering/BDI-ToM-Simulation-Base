"""
GridEnvironment
===============
Defines the 2-D grid world for the BDI Theory of Mind simulation.

Layout
------
- Grid of shape (rows, cols) with four-connected movement.
- N points of interest (POIs) placed at fixed, non-overlapping cells.
- K world state vectors W ∈ {0,1}^N indicate which POIs are open each episode.

Design notes
------------
- POIs are placed away from the grid border (margin=1) to ensure start
  positions can always be found and BFS paths are never degenerate.
- World states are sampled once at environment creation and reused across all
  episodes so that (world_state, agent) combinations are reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Set, Tuple

import numpy as np


@dataclass(frozen=True)
class POI:
    """A single point of interest at a fixed grid position."""
    idx: int
    position: Tuple[int, int]   # (row, col)


class GridEnvironment:
    """
    Parameters
    ----------
    grid_size : (int, int)
        (rows, cols) of the grid.
    n_pois : int
        Number of POIs to place.
    n_world_states : int
        K distinct world state configurations to generate.
    min_open_pois : int
        Minimum number of open POIs required per world state.
    rng : np.random.Generator
        Seeded random number generator for reproducibility.
    """

    def __init__(
        self,
        grid_size: Tuple[int, int],
        n_pois: int,
        n_world_states: int,
        min_open_pois: int,
        rng: np.random.Generator,
    ) -> None:
        self.grid_size = grid_size
        self.n_pois = n_pois
        self.n_world_states = n_world_states
        self.min_open_pois = min_open_pois
        self.rng = rng

        self.pois: List[POI] = self._place_pois()
        self._poi_position_set: Set[Tuple[int, int]] = {p.position for p in self.pois}
        self.world_states: List[np.ndarray] = self._generate_world_states()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _place_pois(self) -> List[POI]:
        """Place N POIs at random, non-overlapping interior grid cells."""
        rows, cols = self.grid_size
        occupied: Set[Tuple[int, int]] = set()
        pois: List[POI] = []

        while len(pois) < self.n_pois:
            r = int(self.rng.integers(1, rows - 1))
            c = int(self.rng.integers(1, cols - 1))
            if (r, c) not in occupied:
                occupied.add((r, c))
                pois.append(POI(idx=len(pois), position=(r, c)))

        return pois

    def _generate_world_states(self) -> List[np.ndarray]:
        """
        Generate K distinct world states W ∈ {0,1}^N.

        Each state is sampled uniformly at random with the constraint that
        at least `min_open_pois` POIs are open.  Rejection sampling is used;
        with N=100 and min_open=10 this converges quickly.
        """
        seen: Set[Tuple[int, ...]] = set()
        world_states: List[np.ndarray] = []

        while len(world_states) < self.n_world_states:
            w = self.rng.integers(0, 2, size=self.n_pois).astype(np.int8)
            if w.sum() < self.min_open_pois:
                continue
            key = tuple(w.tolist())
            if key not in seen:
                seen.add(key)
                world_states.append(w)

        return world_states

    # ------------------------------------------------------------------
    # Episode-level helpers
    # ------------------------------------------------------------------

    def sample_world_state(self) -> np.ndarray:
        """Return a random world state from the pre-generated set."""
        idx = int(self.rng.integers(0, self.n_world_states))
        return self.world_states[idx].copy()

    def sample_start(
        self,
        goal: Tuple[int, int] | None = None,
        min_steps: int = 0,
        exclude: Set[Tuple[int, int]] | None = None,
        max_retries: int = 10000,
    ) -> Tuple[int, int]:
        """
        Sample a random start position not occupied by any POI.

        On an open (wall-free) grid, BFS distance equals Manhattan distance,
        so the minimum-distance constraint is enforced with a cheap L1 check
        rather than a full BFS call.

        Parameters
        ----------
        goal : (int, int), optional
            Goal cell.  If provided alongside `min_steps`, start positions
            with Manhattan distance < min_steps from the goal are rejected.
        min_steps : int
            Minimum required Manhattan distance from `goal`.  Ignored if
            `goal` is None or min_steps == 0.
        exclude : set of (int, int), optional
            Additional positions to exclude beyond POIs and the goal cell.
        max_retries : int
            Safety cap on rejection-sampling iterations.  If exceeded (e.g.
            min_steps is larger than the grid), the constraint is relaxed and
            the nearest valid position is returned with a warning.
        """
        rows, cols = self.grid_size
        blocked = self._poi_position_set | (exclude or set())
        if goal is not None:
            blocked = blocked | {goal}

        enforce_dist = (goal is not None) and (min_steps > 0)
        gr, gc = goal if goal is not None else (0, 0)

        for _ in range(max_retries):
            r = int(self.rng.integers(0, rows))
            c = int(self.rng.integers(0, cols))
            if (r, c) in blocked:
                continue
            if enforce_dist and (abs(r - gr) + abs(c - gc)) < min_steps:
                continue
            return (r, c)

        # Fallback: relax distance constraint and return any non-blocked cell
        import warnings
        warnings.warn(
            f"sample_start: could not find a start position with "
            f"Manhattan distance ≥ {min_steps} from goal {goal} after "
            f"{max_retries} tries. Relaxing constraint.",
            RuntimeWarning,
            stacklevel=2,
        )
        while True:
            r = int(self.rng.integers(0, rows))
            c = int(self.rng.integers(0, cols))
            if (r, c) not in blocked:
                return (r, c)

    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Return the four-connected neighbours of `pos` that lie inside the grid."""
        r, c = pos
        rows, cols = self.grid_size
        neighbours = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                neighbours.append((nr, nc))
        return neighbours

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a JSON-serialisable representation of the environment."""
        return {
            "grid_size": list(self.grid_size),
            "n_pois": self.n_pois,
            "n_world_states": self.n_world_states,
            "min_open_pois": self.min_open_pois,
            "pois": [
                {"idx": p.idx, "position": list(p.position)}
                for p in self.pois
            ],
            "world_states": [w.tolist() for w in self.world_states],
        }
