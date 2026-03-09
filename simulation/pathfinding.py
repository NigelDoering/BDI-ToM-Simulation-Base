"""
Pathfinding
===========
BFS shortest-path on the 2-D four-connected grid.

The grid has no internal obstacles (walls); every cell is reachable from
every other cell, so BFS always succeeds.  The implementation is written
for clarity over raw speed — for N=100 POIs on a 50×50 grid the paths are
short and generation time is dominated by sampling, not BFS.
"""

from __future__ import annotations

from collections import deque
from typing import List, Optional, Tuple


def bfs(
    grid_size: Tuple[int, int],
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> Optional[List[Tuple[int, int]]]:
    """
    Return the lexicographically-first shortest path from `start` to `goal`.

    Parameters
    ----------
    grid_size : (rows, cols)
    start, goal : (row, col)

    Returns
    -------
    list of (row, col) including both endpoints, or None if unreachable
    (should never happen on an open grid).
    """
    if start == goal:
        return [start]

    rows, cols = grid_size
    # Store (position, path) pairs.  For memory efficiency we store only
    # predecessor pointers on large grids, but paths here are ≤ ~100 steps.
    queue: deque = deque()
    queue.append((start, [start]))
    visited = {start}

    while queue:
        (r, c), path = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < rows
                and 0 <= nc < cols
                and (nr, nc) not in visited
            ):
                new_path = path + [(nr, nc)]
                if (nr, nc) == goal:
                    return new_path
                visited.add((nr, nc))
                queue.append(((nr, nc), new_path))

    return None  # unreachable (should not occur on open grid)


def path_length(
    grid_size: Tuple[int, int],
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> int:
    """Return the number of steps (edges) in the shortest path, or -1 if unreachable."""
    path = bfs(grid_size, start, goal)
    return -1 if path is None else len(path) - 1
