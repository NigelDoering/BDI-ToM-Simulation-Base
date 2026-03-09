"""
BDI Agent
=========
Implements the Belief-Desire-Intention agent model described in the problem
statement.

Key design decisions
--------------------
Desires  — sampled once from a Dirichlet(α·1_N) prior at agent creation and
           held fixed for the agent's lifetime.  Different α values yield
           different prior shapes:
             α < 1  → sparse, peaked preferences
             α = 1  → uniform prior (all preference profiles equally likely)
             α > 1  → diffuse, near-uniform preferences

Beliefs  — sampled per episode by independently flipping the true world-state
           bits with probability ρ.  Guaranteed to have at least one believed-
           open POI so goal selection is always well-defined.

           Veridical belief: Ba = W  (Hamming distance 0)
           False belief:     Ba ≠ W  (Hamming distance > 0, enforced)

Goal     — selected by renormalising desires over believed-open POIs only
           (Equation 5 in the problem statement).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class BDIAgent:
    """
    Attributes
    ----------
    agent_id : int
    desires : np.ndarray, shape (N,)
        Normalised preference vector (sums to 1) sampled from Dirichlet prior.
    is_novel : bool
        True if this agent is withheld from training (test conditions 3 & 4).
    """

    agent_id: int
    desires: np.ndarray
    is_novel: bool = False

    # ------------------------------------------------------------------
    # Belief sampling
    # ------------------------------------------------------------------

    def sample_false_belief(
        self,
        world_state: np.ndarray,
        rho: float,
        rng: np.random.Generator,
        max_retries: int = 100,
    ) -> np.ndarray:
        """
        Sample a belief vector by flipping each bit of `world_state` with
        probability ρ.  Retries until:
          1. At least one POI is believed open (goal selection requires this).
          2. The belief differs from world_state (Hamming > 0).

        E[dH(Ba, W)] = ρ · N.
        """
        n = len(world_state)
        for _ in range(max_retries):
            flip = rng.random(n) < rho
            belief = world_state.copy()
            belief[flip] = 1 - belief[flip]
            if belief.sum() > 0 and not np.array_equal(belief, world_state):
                return belief

        # Fallback: flip exactly one random bit to guarantee Hamming > 0
        belief = world_state.copy()
        idx = int(rng.integers(n))
        belief[idx] = 1 - belief[idx]
        if belief.sum() == 0:
            # Ensure at least one open belief
            belief[idx] = 1
        return belief

    def sample_veridical_belief(self, world_state: np.ndarray) -> np.ndarray:
        """Return a copy of the world state (dH = 0)."""
        return world_state.copy()

    # ------------------------------------------------------------------
    # Goal selection  (Equation 5)
    # ------------------------------------------------------------------

    def select_goal(
        self,
        belief: np.ndarray,
        rng: np.random.Generator,
    ) -> int:
        """
        Sample goal index by renormalising desires over believed-open POIs.

        Returns the index of the selected POI (0 … N-1).
        """
        masked = self.desires * belief          # zero out believed-closed POIs
        total = masked.sum()

        if total == 0:
            # Degenerate: agent believes no POIs are open.
            # Fall back to raw desire distribution (shouldn't occur due to
            # belief sampling constraint, but kept as a safety net).
            probs = self.desires / self.desires.sum()
        else:
            probs = masked / total

        # Numerical safety: re-normalise to avoid floating-point drift
        probs = probs / probs.sum()
        return int(rng.choice(len(probs), p=probs))

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "desires": self.desires.tolist(),
            "is_novel": self.is_novel,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_agents(
    n: int,
    n_pois: int,
    alpha: float,
    rng: np.random.Generator,
    is_novel: bool = False,
    start_id: int = 0,
) -> List[BDIAgent]:
    """
    Create a pool of `n` BDI agents with Dirichlet(α·1_N)-sampled desires.

    Parameters
    ----------
    n        : number of agents to create
    n_pois   : dimensionality of the desire vector
    alpha    : Dirichlet concentration parameter
    rng      : seeded RNG for reproducibility
    is_novel : whether these agents are novel (withheld from training)
    start_id : first agent_id (allows familiar and novel pools to have unique IDs)
    """
    agents: List[BDIAgent] = []
    for i in range(n):
        desires = rng.dirichlet(np.ones(n_pois) * alpha)
        agents.append(
            BDIAgent(agent_id=start_id + i, desires=desires, is_novel=is_novel)
        )
    return agents
