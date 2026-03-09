"""
Trajectory Visualiser
=====================
Renders an animated GIF of a single BDI agent episode on the 2-D grid.

Visual encoding
---------------
Grid background  — light grey cells with subtle grid lines
POIs             — coloured diamonds:
                     ground-truth open   → green  (solid)
                     ground-truth closed → red    (solid)
                   Ring around diamond encodes agent *belief*:
                     believed open       → thick coloured ring
                     believed closed     → no ring / faint ring
                   Diamond *size* scales with agent desire weight (larger = higher desire)
                   when show_desires=True.
Agent            — filled blue circle; leaves a fading blue trail
Start marker     — blue upward triangle (▲)
Goal marker      — gold star (★), bordered in black for visibility
Step counter     — title bar: "Step t / T  |  Agent {id}  |  Condition {k}"
Legend           — compact legend in upper-right corner

Colour conventions
------------------
  Open POI (truth)   : #2ca02c  (green)
  Closed POI (truth) : #d62728  (red)
  Believed open ring : #1f77b4  (blue)
  Believed closed    : no ring
  Agent              : #1f77b4  (blue)
  Goal               : #ffbf00  (amber/gold)
"""

from __future__ import annotations

import io
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for scripts/servers
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Colour constants
# ---------------------------------------------------------------------------
_GREEN  = "#2ca02c"
_RED    = "#d62728"
_BLUE   = "#1f77b4"
_AMBER  = "#ffbf00"
_GREY   = "#cccccc"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_trajectory_gif(
    episode: Dict[str, Any],
    poi_positions: List[Tuple[int, int]],
    grid_size: Tuple[int, int],
    output_path: str,
    fps: int = 4,
    dpi: int = 80,
    show_beliefs: bool = True,
    show_desires: bool = True,
) -> str:
    """
    Create an animated GIF of `episode`'s trajectory and save to `output_path`.

    Parameters
    ----------
    episode : dict
        A single episode dict as produced by Episode.to_dict().
    poi_positions : list of (row, col)
        Grid coordinates for each POI (index matches episode arrays).
    grid_size : (rows, cols)
    output_path : str
        Destination file path (should end in '.gif').
    fps : int
    dpi : int
    show_beliefs : bool
        Overlay belief rings on POI markers.
    show_desires : bool
        Scale POI marker size by desire weight.

    Returns
    -------
    str : the output_path written.
    """
    traj = [tuple(p) for p in episode["trajectory"]]
    world_state = episode["world_state"]
    belief = episode["belief"]
    desires = episode["desires"]
    goal_pos = tuple(episode["goal_position"])
    start_pos = tuple(episode["start_position"])
    goal_idx = episode["goal_idx"]
    agent_id = episode["agent_id"]
    condition = episode["condition"]
    hamming = episode["hamming_distance"]

    n_pois = len(poi_positions)
    rows, cols = grid_size
    T = len(traj)

    # --- Desire sizes: map to marker area in points² ---
    desires_arr = np.array(desires, dtype=float)
    if show_desires and desires_arr.max() > 0:
        # Scale so minimum = 80pt², maximum = 350pt²
        d_norm = (desires_arr - desires_arr.min()) / (desires_arr.max() - desires_arr.min() + 1e-12)
        poi_sizes = 80 + d_norm * 270   # (N,)
    else:
        poi_sizes = np.full(n_pois, 150.0)

    # --- Figure size: keep cells square, cap at (10,10) ---
    cell_px = 10              # desired pixels per cell (at dpi=80 → 0.125in)
    fig_w = min(cols * cell_px / dpi, 10.0)
    fig_h = min(rows * cell_px / dpi, 10.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    frames: List[Image.Image] = []

    # ---------------------------------------------------------------------------
    # Render each frame
    # ---------------------------------------------------------------------------
    for t, (r, c) in enumerate(traj):
        ax.cla()

        # --- Grid background ---
        ax.set_facecolor("#f5f5f5")
        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(-0.5, rows - 0.5)
        ax.set_aspect("equal")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Light grid lines
        for i in range(rows + 1):
            ax.axhline(i - 0.5, color=_GREY, linewidth=0.3, zorder=0)
        for j in range(cols + 1):
            ax.axvline(j - 0.5, color=_GREY, linewidth=0.3, zorder=0)

        # --- POIs ---
        for i, (pr, pc) in enumerate(poi_positions):
            is_open = bool(world_state[i])
            believed_open = bool(belief[i]) if show_beliefs else is_open
            poi_color = _GREEN if is_open else _RED
            size = poi_sizes[i]

            # Belief ring: circle behind diamond
            if show_beliefs and believed_open:
                ax.scatter(
                    pc, pr,
                    s=size * 3.0,
                    marker="o",
                    facecolors="none",
                    edgecolors=_BLUE,
                    linewidths=1.5,
                    zorder=2,
                )

            # POI diamond marker
            ax.scatter(
                pc, pr,
                s=size,
                marker="D",
                color=poi_color,
                edgecolors="black",
                linewidths=0.5,
                zorder=3,
                alpha=0.85,
            )

            # Goal star overlay on top of the goal POI (every frame)
            if i == goal_idx:
                ax.scatter(
                    pc, pr,
                    s=size * 1.8,
                    marker="*",
                    color=_AMBER,
                    edgecolors="black",
                    linewidths=0.5,
                    zorder=4,
                )

        # --- Trajectory trail ---
        if t > 0:
            trail_rs = [p[0] for p in traj[:t + 1]]
            trail_cs = [p[1] for p in traj[:t + 1]]
            # Segments fade from transparent (oldest) to opaque (newest)
            n_seg = len(trail_rs) - 1
            for s in range(n_seg):
                alpha = 0.2 + 0.8 * (s / max(n_seg - 1, 1))
                ax.plot(
                    [trail_cs[s], trail_cs[s + 1]],
                    [trail_rs[s], trail_rs[s + 1]],
                    color=_BLUE,
                    linewidth=1.5,
                    alpha=alpha,
                    zorder=5,
                )

        # --- Start marker ---
        ax.scatter(
            start_pos[1], start_pos[0],
            s=90,
            marker="^",
            color=_BLUE,
            edgecolors="black",
            linewidths=0.5,
            zorder=6,
            label="Start",
        )

        # --- Agent circle (current position) ---
        ax.scatter(
            c, r,
            s=120,
            marker="o",
            color=_BLUE,
            edgecolors="white",
            linewidths=1.0,
            zorder=7,
        )

        # --- Title ---
        belief_tag = "veridical" if hamming == 0 else f"false (dH={hamming})"
        ax.set_title(
            f"Step {t}/{T - 1}  |  Agent {agent_id}"
            + (f"  |  Cond {condition}" if condition > 0 else "")
            + f"  |  Belief: {belief_tag}",
            fontsize=7,
            pad=3,
        )

        # --- Legend (first frame only is enough; Pillow keeps it) ---
        legend_elements = [
            mpatches.Patch(facecolor=_GREEN, edgecolor="black", label="Open POI (truth)"),
            mpatches.Patch(facecolor=_RED,   edgecolor="black", label="Closed POI (truth)"),
            mpatches.Patch(facecolor="none", edgecolor=_BLUE,   label="Believed open (ring)"),
            mpatches.Patch(facecolor=_AMBER, edgecolor="black", label="Goal"),
        ]
        ax.legend(
            handles=legend_elements,
            loc="upper right",
            fontsize=5,
            framealpha=0.8,
            markerscale=0.7,
        )

        # --- Capture frame ---
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()

    plt.close(fig)

    # ---------------------------------------------------------------------------
    # Assemble GIF
    # ---------------------------------------------------------------------------
    if not frames:
        raise ValueError("No frames generated — trajectory may be empty.")

    duration_ms = max(1, int(1000 / fps))
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    return output_path


# ---------------------------------------------------------------------------
# Convenience: render from metadata + episode dict loaded from JSON
# ---------------------------------------------------------------------------

def render_episode_from_json(
    episode: Dict[str, Any],
    metadata: Dict[str, Any],
    output_path: str,
    fps: int = 4,
    dpi: int = 80,
    show_beliefs: bool = True,
    show_desires: bool = True,
) -> str:
    """
    High-level wrapper that extracts POI positions and grid size from a
    metadata dict (as written by dataset_gen.save_dataset) and calls
    create_trajectory_gif.

    Parameters
    ----------
    episode  : episode dict loaded from a split JSON file
    metadata : dict loaded from metadata.json
    output_path : destination GIF path
    """
    env_meta = metadata["environment"]
    grid_size = tuple(env_meta["grid_size"])
    poi_positions = [tuple(p["position"]) for p in env_meta["pois"]]

    return create_trajectory_gif(
        episode=episode,
        poi_positions=poi_positions,
        grid_size=grid_size,
        output_path=output_path,
        fps=fps,
        dpi=dpi,
        show_beliefs=show_beliefs,
        show_desires=show_desires,
    )
