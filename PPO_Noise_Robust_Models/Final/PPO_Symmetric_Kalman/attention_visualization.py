"""
Attention-weight visualization for thesis figures.

Two modes
---------
1. plot_searchlight  — "radar sweep" overlay: lines from ownship to every
                       neighbour, where alpha and linewidth scale with weight.
2. plot_top_neighbors — horizontal bar chart of the Top-N most-attended neighbours.
3. plot_attention_combined — convenience wrapper: both plots side-by-side.

Usage example (from evaluate.py)
---------------------------------
    from attention_visualization import plot_attention_combined

    plot_attention_combined(
        ownship_pos   = (own_x_km, own_y_km),
        neigh_pos     = {"KL002": (x, y), ...},
        attn_weights  = {"KL002": 0.42, ...},   # raw, unnormalised
        ownship_id    = "KL001",
        step          = 30,
        save_path     = "figures/attention_ep001_step0030.png",
    )
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.transforms as transforms
from matplotlib.markers import MarkerStyle
from typing import Optional

# ── Style ─────────────────────────────────────────────────────────────────────
try:
    plt.style.use("seaborn-v0_8-paper")
except OSError:
    try:
        plt.style.use("seaborn-paper")
    except OSError:
        pass   # matplotlib version without that style — defaults are fine

matplotlib.rcParams.update({
    "font.family":   "serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ── Palette choices ──────────────────────────────────────────────────────────
_CMAP          = cm.viridis         # more clear to the eye
_OWNSHIP_COLOUR = "red"             # deep red — clearly distinct
_INTRUDER_COLOUR = "#b2182b"        # deep red used as fallback edge colour

# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalise(weights: np.ndarray) -> np.ndarray:
    """Map an array of weights to [0, 1] by dividing by its maximum."""
    w = np.asarray(weights, dtype=float)
    wmax = w.max()
    return w / wmax if wmax > 1e-12 else np.zeros_like(w)

def _get_rotated_marker(vx: float, vy: float, base_marker: str = '^') -> MarkerStyle:
    """Return a MarkerStyle rotated to point in the direction of (vx, vy)."""
    if abs(vx) < 1e-4 and abs(vy) < 1e-4:
        # Stationary, just return base marker
        return MarkerStyle(base_marker)
    
    # We want '^' (which points up, i.e., vy>0, vx=0) to point in (vx, vy)
    angle_deg = np.degrees(np.arctan2(vy, vx)) - 90
    
    # Matplotlib MarkerStyle expects an Affine2D transform
    t = transforms.Affine2D().rotate_deg(angle_deg)
    return MarkerStyle(base_marker, transform=t)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Searchlight overlay
# ─────────────────────────────────────────────────────────────────────────────

def plot_searchlight(
    ownship_pos: tuple,
    neigh_pos: dict,           # { neighbor_id : (x_km, y_km) }
    attn_weights: dict,        # { neighbor_id : weight  (raw, unnormalised) }
    ownship_vel: Optional[tuple] = None, # (vx, vy)
    neigh_vel: Optional[dict] = None,    # { neighbor_id : (vx, vy) }
    ownship_id: str = "Ownship",
    step: int = 0,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Draw the "searchlight" overlay.

    A line is drawn from *ownship_pos* to each neighbour.  Both the alpha
    (transparency) and the linewidth of the line are scaled by that
    neighbour's attention weight.  A small label printed near each intruder
    shows the raw weight as ``α = 0.42``.

    Parameters
    ----------
    ownship_pos  : 2-tuple (x_km, y_km) – ownship position.
    neigh_pos    : dict mapping neighbor_id → (x_km, y_km).
    attn_weights : dict mapping neighbor_id → raw attention weight.
    ownship_id   : label shown next to the ownship star marker.
    step         : current simulation step (title only).
    ax           : optional existing Axes; a new figure is created if None.
    save_path    : if given, the figure is saved there at 200 dpi.

    Returns
    -------
    matplotlib.figure.Figure
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.get_figure()

    own_x, own_y = ownship_pos
    neigh_ids = list(neigh_pos.keys())

    if not neigh_ids:
        ax.scatter(own_x, own_y, s=200, marker="*",
                   color=_OWNSHIP_COLOUR, edgecolors="black", linewidths=0.8,
                   zorder=6, label="Ownship")
        
        ax.set_title("Attention weights visualized",
                     fontsize=12, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.75, edgecolor="grey")
        return fig

    raw   = np.array([attn_weights.get(nid, 0.0) for nid in neigh_ids], dtype=float)
    normw = _normalise(raw)

    scalar_map = cm.ScalarMappable(
        cmap=_CMAP,
        norm=mcolors.Normalize(vmin=0.0, vmax=1.0),
    )

    # ── lines (drawn first so markers sit on top) ────────────────────────────
    for nid, nw in zip(neigh_ids, normw):
        nx, ny = neigh_pos[nid]
        colour = scalar_map.to_rgba(nw)
        lw     = 0.4 + 4.0 * nw        # linewidth ∈ [0.4, 4.4]
        alpha  = 0.10 + 0.90 * nw      # alpha     ∈ [0.10, 1.00]
        ax.plot([own_x, nx], [own_y, ny],
                color=colour, lw=lw, alpha=float(alpha), zorder=2, solid_capstyle="round")

    # ── neighbour markers ────────────────────────────────────────────────────
    
    total_raw = raw.sum()
    pcts = (raw / total_raw * 100.0) if total_raw > 1e-12 else np.zeros_like(raw)
    
    # Get indices of the top 3 items
    top_3_indices = set(np.argsort(raw)[-3:])
    
    for i, (nid, rw, nw, pct) in enumerate(zip(neigh_ids, raw, normw, pcts)):
        nx, ny = neigh_pos[nid]
        colour = scalar_map.to_rgba(nw)
        
        # Draw pointing arrow if velocity is known, else circle
        if neigh_vel and nid in neigh_vel:
            vx, vy = neigh_vel[nid]
            m = _get_rotated_marker(vx, vy, base_marker='^')
            ax.scatter(nx, ny, s=100, color=colour, edgecolors="black",
                       linewidths=0.5, zorder=4, marker=m)
        else:
            ax.scatter(nx, ny, s=55, color=colour, edgecolors="black",
                       linewidths=0.5, zorder=4)
            
        if i in top_3_indices and pct > 0.1: # Only label top 3
            label = f"{pct:.1f}%"
            txt = ax.text(
                nx + 0.02, ny + 0.02, label,
                fontsize=8, va="bottom", ha="left",
                color="black", zorder=5, fontweight="bold"
            )
            txt.set_path_effects([
                pe.withStroke(linewidth=2.0, foreground="white")
            ])

    # ── ownship marker ───────────────────────────────────────────────────────
    
    if ownship_vel:
        vx, vy = ownship_vel
        m = _get_rotated_marker(vx, vy, base_marker='^')
        ax.scatter(own_x, own_y, s=300, marker=m,
                   color=_OWNSHIP_COLOUR, edgecolors="black", linewidths=0.8,
                   zorder=6, label="Ownship")
    else:
        ax.scatter(own_x, own_y, s=220, marker="*",
                   color=_OWNSHIP_COLOUR, edgecolors="black", linewidths=0.8,
                   zorder=6, label="Ownship")

    # ── colourbar ────────────────────────────────────────────────────────────
    scalar_map.set_array([])
    cbar = fig.colorbar(scalar_map, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Attention weight (normalised)", fontsize=9)

    ax.set_xlabel("Longitude (km)", fontsize=10)
    ax.set_ylabel("Latitude (km)", fontsize=10)
    ax.set_title("Attention weights visualized",
                 fontsize=12, fontweight="bold")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", framealpha=0.75, edgecolor="grey")

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[viz] Saved searchlight figure → {save_path}")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Top-N neighbour importance bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_top_neighbors(
    attn_weights: dict,        # { neighbor_id : weight }
    ownship_id: str = "Ownship",
    top_n: int = 5,
    step: int = 0,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Horizontal bar chart of the Top-N most-attended neighbours.

    Each bar's length represents that neighbour's *share* of the total
    attention (in percent), and is coloured by the plasma colourmap.

    Parameters
    ----------
    attn_weights : dict mapping neighbor_id → raw attention weight.
    ownship_id   : label for the figure title.
    top_n        : number of bars to show (default 5).
    step         : current simulation step (title only).
    ax           : optional existing Axes.
    save_path    : if given, saved there at 200 dpi.

    Returns
    -------
    matplotlib.figure.Figure
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(5, 3.2))
    else:
        fig = ax.get_figure()

    if not attn_weights:
        ax.text(0.5, 0.5, "No attention data available",
                transform=ax.transAxes, ha="center", va="center", fontsize=10)
        return fig

    # Sort and trim to top_n
    sorted_items = sorted(attn_weights.items(), key=lambda kv: kv[1], reverse=True)
    top = sorted_items[:top_n]
    labels = [item[0] for item in top]
    values = np.array([item[1] for item in top], dtype=float)

    total       = values.sum()
    percentages = (values / total * 100.0) if total > 1e-12 else np.zeros_like(values)
    normw       = _normalise(values)
    colours     = [_CMAP(nw) for nw in normw]

    y_pos = np.arange(len(labels))
    bars  = ax.barh(y_pos, percentages, color=colours,
                    edgecolor="black", linewidth=0.5, height=0.55)

    # Percentage labels at the right end of each bar
    x_max = percentages.max() if percentages.max() > 0 else 1.0
    for bar, pct in zip(bars, percentages):
        ax.text(
            bar.get_width() + x_max * 0.015,
            bar.get_y() + bar.get_height() / 2.0,
            f"{pct:.1f}%",
            va="center", ha="left", fontsize=8,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Share of total attention (%)", fontsize=10)
    ax.set_title(
        f"Top-{top_n} neighbours  —  {ownship_id}  |  step {step}",
        fontsize=11, fontweight="bold",
    )
    ax.set_xlim(0, x_max * 1.25)
    ax.invert_yaxis()               # highest weight at the top
    ax.grid(axis="x", alpha=0.30, linestyle="--")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[viz] Saved bar chart figure → {save_path}")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Combined figure (searchlight left, bar chart right)
# ─────────────────────────────────────────────────────────────────────────────

def plot_attention_combined(
    ownship_pos: tuple,
    neigh_pos: dict,
    attn_weights: dict,
    ownship_id: str = "Ownship",
    top_n: int = 5,
    step: int = 0,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Side-by-side combined figure: searchlight (left) + importance bar chart (right).

    Parameters
    ----------
    ownship_pos  : 2-tuple (x_km, y_km).
    neigh_pos    : dict mapping neighbor_id → (x_km, y_km).
    attn_weights : dict mapping neighbor_id → raw attention weight.
    ownship_id   : label shown in both panels and the suptitle.
    top_n        : how many bars to show in the right panel.
    step         : simulation step shown in titles.
    save_path    : if given, figure is saved there at 200 dpi.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, (ax_l, ax_r) = plt.subplots(
        1, 2,
        figsize=(12, 5),
        gridspec_kw={"width_ratios": [1.1, 0.9]},
    )
    fig.suptitle(
        f"Attention mechanism  —  {ownship_id}  |  Step {step}",
        fontsize=13, fontweight="bold", y=1.01,
    )

    plot_searchlight(
        ownship_pos, neigh_pos, attn_weights,
        ownship_id=ownship_id, step=step, ax=ax_l,
    )
    plot_top_neighbors(
        attn_weights,
        ownship_id=ownship_id, top_n=top_n, step=step, ax=ax_r,
    )

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[viz] Saved combined attention figure → {save_path}")

    return fig
