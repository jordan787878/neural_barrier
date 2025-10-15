# verify_crown_direct.py
# -----------------------------------------------------------------------------
# Direct LiRPA (CROWN) verifier of the generator:
#   Φ(x) = f(x)·∇V(x) + 0.5 [ γ11(x)^2 H11(x) + γ22(x)^2 H22(x) ]
# for a trained 1-hidden-layer sigmoid net V.
#
# Public API (unchanged):
#   verify_crown_differential(const, net, refine_N=2, max_depth=20, device="cpu",
#                             visualize=True, gif_path=None, gif_fps=1.0, live_monitor=False)
#
# Assumptions:
# - const provides torch-callables:
#       f_of_x(x) -> (f1, f2)          # each (N,1)
#       gamma_sq_of_x(x) -> (g11_sq, g22_sq)  # each (N,1)
# - set-ops utilities: rect_diff_2d, refine_list_2d_box
# - Only plain CROWN (no β-CROWN).
# -----------------------------------------------------------------------------

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from src.setoperations import rect_diff_2d, refine_list_2d_box
from src.polyviewer import _poly_viewer_setup, _poly_viewer_update
import matplotlib.pyplot as plt
from matplotlib import animation


# ───────────────────────────── Core: Φ(x) module ──────────────────────────────
class _Phi(nn.Module):
    """
    One-hidden-layer sigmoid net:
        V(x) = W1 · σ(W0 (x/100) + b0) + b1
    Forward returns Φ(x) = f·∇V + 0.5 (γ11² H11 + γ22² H22).
    ∇V and Hessian diagonals are computed in closed form via h = σ(z).
    """
    def __init__(self, net: nn.Module, const):
        super().__init__()
        assert hasattr(const, "f_of_x") and hasattr(const, "gamma_sq_of_x"), \
            "const must provide f_of_x(x) and gamma_sq_of_x(x) torch-callables."

        self.const = const
        lin0: nn.Linear = net.Layers[0]
        lin1: nn.Linear = net.Layers[1]

        # Cache weights/biases as buffers (constants in the graph)
        W0 = lin0.weight.detach().clone() / 100.0  # (m,D), includes input normalization (absorbs x/100 into W0)”
        b0 = lin0.bias.detach().clone()            # (m,)
        W1 = lin1.weight.detach().clone()          # (1,m)
        b1 = lin1.bias.detach().clone()            # (1,)  (not used by Φ’s derivatives)

        self.register_buffer("W0", W0)
        self.register_buffer("b0", b0)
        self.register_buffer("W1", W1)
        self.register_buffer("b1", b1)
        self.register_buffer("s1", W0[:, 0])       # s_k1
        self.register_buffer("s2", W0[:, 1])       # s_k2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # z, h, σ', σ''
        z = F.linear(x, self.W0, self.b0)              # (N,m)
        h = torch.sigmoid(z)                           # (N,m)
        d = h * (1.0 - h)                              # σ'(z)
        q = (1.0 - 2.0 * h) * d                        # σ''(z)

        # ∇V and Hessian diagonals (batchwise sums over neurons)
        dVdx1 = torch.sum(self.W1 * (d * self.s1), dim=1, keepdim=True)        # (N,1)
        dVdx2 = torch.sum(self.W1 * (d * self.s2), dim=1, keepdim=True)        # (N,1)
        H11   = torch.sum(self.W1 * (q * (self.s1 ** 2)), dim=1, keepdim=True) # (N,1)
        H22   = torch.sum(self.W1 * (q * (self.s2 ** 2)), dim=1, keepdim=True) # (N,1)

        # Dynamics (torch-callable from const)
        f1, f2 = self.const.f_of_x(x)                    # each (N,1)
        g11_sq, g22_sq = self.const.gamma_sq_of_x(x)     # each (N,1)

        drift = f1 * dVdx1 + f2 * dVdx2
        diff  = 0.5 * (g11_sq * H11 + g22_sq * H22)
        return drift + diff                               # (N,1)


# ───────────────────────────── Box-level CROWN ────────────────────────────────
@torch.no_grad()
def _bound_phi_on_box(const, net, x_range, device="cpu") -> Tuple[float, float]:
    """
    Direct LiRPA: build Φ(x) end-to-end and bound it on the box via CROWN.
    Returns (L, U).
    """
    D = int(net.Layers[0].in_features)
    mod = _Phi(net, const).to(device).eval()
    bm = BoundedModule(mod, torch.zeros(1, D, device=device), device=device)

    xr = np.asarray(x_range, dtype=np.float32)              # (D,2)
    xL = torch.from_numpy(xr[:, 0][None, :]).to(device)     # (1,D)
    xU = torch.from_numpy(xr[:, 1][None, :]).to(device)     # (1,D)
    x0 = (xL + xU) / 2

    ptb = PerturbationLpNorm(norm=np.inf, eps=None, x_L=xL, x_U=xU)
    bx  = BoundedTensor(x0, ptb)

    lb, ub = bm.compute_bounds(x=(bx,), method="CROWN")     # (1,1)
    L, U = float(lb.item()), float(ub.item())
    return (min(L, U), max(L, U))


# ───────────────────────────── Outer verification loop ────────────────────────
def _unit_verify_crown_direct(const, net, boxes, device="cpu", verbose=False):
    Ls, Us = [], []
    for box in boxes:
        L, U = _bound_phi_on_box(const, net, box, device=device)
        Ls.append(L); Us.append(U)
    Ls = np.asarray(Ls, dtype=np.float64); Us = np.asarray(Us, dtype=np.float64)
    if verbose:
        print("[info] per-box U:", Us)
    return np.array([Ls.min() if Ls.size else 0.0,
                     Us.max() if Us.size else 0.0], dtype=np.float64), Us


# ───────────────────────────── API ─────────────────────────────
def verify_crown_differential_direct(const, net, refine_N: int = 2, max_depth: int = 20,
    visualize: bool = True,
    gif_path: str | None = None, gif_fps: float = 1.0,
    show_plot: bool = True,   # <-- NEW: set False to disable any window (GIF-only)
    device: str = "cpu",
):
    assert hasattr(const, "f_of_x") and hasattr(const, "gamma_sq_of_x"), \
        "verify_crown_direct requires torch-callable const.f_of_x and const.gamma_sq_of_x."

    # Initial cover: X_RANGE \ (X_GOAL ∪ X_UNSAFE)
    boxes = []
    for rect in rect_diff_2d(const.X_RANGE, const.X_GOAL_RANGE):
        boxes += rect_diff_2d(rect, const.X_UNSAFE_RANGE)

    condition = 0.0
    GV_final = None

    # We need a figure if we either visualize live OR save a GIF
    need_figure = visualize or (gif_path is not None)
    if need_figure:
        # If you only want a GIF (no window), pass show_plot=False
        fig, ax, pc, cbar, gv_txt = _poly_viewer_setup(const.X_RANGE, cmap_name="cool", show_plot=show_plot)

    writer_ctx = None
    if gif_path is not None and need_figure:
        writer = animation.PillowWriter(fps=gif_fps)
        writer_ctx = writer.saving(fig, gif_path, dpi=110)
        writer_ctx.__enter__()

    try:
        for it in range(max_depth):
            GV_bound, per_box_ub = _unit_verify_crown_direct(const, net, boxes, device=device)
            GV_final = GV_bound

            ub = np.asarray(per_box_ub, dtype=float).reshape(-1)
            if ub.size == 0:
                to_refine, to_refine_ub = [], np.empty((0,), dtype=float)
            else:
                idx = np.nonzero(ub >= condition)[0]
                to_refine    = [boxes[i] for i in idx] if idx.size else []
                to_refine_ub = ub[idx] if idx.size else np.empty((0,), dtype=float)
                print("[info] regions to be verified: ", idx.size)

            if need_figure:
                _poly_viewer_update(
                    fig, ax, pc, cbar,
                    to_refine, to_refine_ub, it,
                    gv_bound=GV_bound, gv_txt=gv_txt,
                    do_pause=show_plot  # pause only if displaying
                )

                if gif_path is not None:
                    fig.canvas.draw()
                    writer.grab_frame()

            if not to_refine:
                print("[complete] GV bound:", GV_bound)
                return GV_final

            boxes = refine_list_2d_box(to_refine, refine_N)

    finally:
        if writer_ctx is not None:
            writer_ctx.__exit__(None, None, None)
            print(f"[info] Saved GIF to: {gif_path}")

        # Only block/show if we were displaying a window
        if need_figure and show_plot:
            ax.set_xlabel(ax.get_xlabel() + "  |  done ✓")
            fig.canvas.draw()
            try: plt.show()
            except Exception: pass

    print("[info] GV bound:", GV_bound)
    return GV_final
