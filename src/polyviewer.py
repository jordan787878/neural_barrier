# ---- minimal PolyCollection viewer ----

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize

def _poly_viewer_setup(x_range, cmap_name="cool", show_plot: bool = True):
    (x_min, x_max), (y_min, y_max) = np.asarray(x_range, float)
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    # ax.set_title("Pending U(B) ≥ 0 (boxes)")

    pc = PolyCollection(
        verts=[],
        cmap=plt.get_cmap(cmap_name),
        norm=Normalize(vmin=0.0, vmax=1.0),
        edgecolors="k", linewidths=0, antialiased=False, alpha=0.95
    )
    ax.add_collection(pc)

    # Seed an array so the colorbar has a mappable with data (some mpl versions require this).
    pc.set_array(np.array([0.0], dtype=float))

    cbar = fig.colorbar(pc, ax=ax)
    cbar.set_label("Upper bound of G[V](B)")

    # NEW: a small text label above the axes showing the current global GV bound
    gv_txt = ax.text(
        0.01, 1.02, "", transform=ax.transAxes, ha="left", va="bottom",
        fontsize=18, color="black"
    )

    if show_plot:
        plt.show(block=False)  # live window only when requested
    return fig, ax, pc, cbar, gv_txt


def _boxes_to_verts(boxes_arr: np.ndarray):
    if boxes_arr.size == 0: return []
    x0 = boxes_arr[:, 0, 0]; x1 = boxes_arr[:, 0, 1]
    y0 = boxes_arr[:, 1, 0]; y1 = boxes_arr[:, 1, 1]
    verts = np.stack([
        np.stack([x0, y0], axis=1),
        np.stack([x1, y0], axis=1),
        np.stack([x1, y1], axis=1),
        np.stack([x0, y1], axis=1),
    ], axis=1)  # (N,4,2)
    return [verts[i] for i in range(verts.shape[0])]


def _poly_viewer_update(fig, ax, pc: PolyCollection, cbar, boxes, ub, it,
                        gv_bound=None, fixed_vmax: float | None = None, do_pause: bool = True,
                        gv_txt=None):
    """Update verts + colors in one shot from the full ub vector, and show global GV bound."""
    boxes_arr = np.asarray(boxes, float)
    pc.set_verts([] if boxes_arr.size == 0 else _boxes_to_verts(boxes_arr))

    vals = np.asarray(ub, float).ravel()
    if vals.size == 0:
        vals = np.array([0.0], float)

    pc.norm.vmin = 0.0
    pc.norm.vmax = float(fixed_vmax) if fixed_vmax is not None else float(max(vals.max(), 1e-12))
    pc.set_array(vals)
    cbar.update_normal(pc)

    # Show per-iteration stats in the x-label
    ax.set_xlabel(f"iter={it}, to verify={len(boxes)}, Upper Bound G[V](B) ∈ [{vals.min():.3f},{vals.max():.3f}]")

    # NEW: show the current global GV bound [L, U] from this iteration
    if gv_txt is not None and gv_bound is not None:
        L, U = gv_bound
        gv_txt.set_text(f"G[V](B) ∈ [{L:.4g}, {U:.4g}]",)

    fig.canvas.draw()
    if do_pause:
        plt.pause(0.01)  # only pump GUI when showing a window
