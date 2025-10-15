# verify_crown.py
# -----------------------------------------------------------------------------
# Certify negativity of the generator G[phi] for a trained 1-hidden-layer
# sigmoid network phi(x) = W1 * sigma(W0 * (x/100) + b0) + b1 on a 2D domain.
# Hidden activations are bounded with CROWN-IBP (auto_LiRPA). Derivatives and
# generator terms are bounded via tight interval algebra, including exact cubic
# extremizers for sigma'' over [l, u].
#
# Public API mirrors your IBP verifier (no 'method' arg):
#   verify_crown_differential(const, net, refine_N=2, max_depth=100, device="cpu")
#
# Requirements:
#   pip install torch auto-lirpa
# -----------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple

from src.setoperations import rect_diff_2d, refine_list_2d_box
from src.polyviewer import _poly_viewer_setup, _poly_viewer_update
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
import matplotlib.pyplot as plt
from matplotlib import animation


# ---------------------------- Helpers / Wrappers ------------------------------

class HiddenWrapper(nn.Module):
    """
    First affine + sigmoid of IBPNet, including x/100 normalization.
    Input:  x in R^d  ->  h = sigmoid(W0 * (x/100) + b0) in R^m
    """
    def __init__(self, net: nn.Module):
        super().__init__()
        self.inp = net.Layers[0]          # Linear(in->m)
        self.act = nn.Sigmoid()           # Sigmoid
        self._norm = 100.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.inp(x / self._norm))


@torch.no_grad()
def crown_hidden_bounds(net: nn.Module, x_range: np.ndarray, device: str = "cpu"):
    hidden_mod = HiddenWrapper(net).to(device).eval()

    X_DIM = net.Layers[0].in_features
    dummy = torch.zeros(1, X_DIM, device=device)
    bm = BoundedModule(hidden_mod, dummy, device=device)

    # ---- FIX: build xL/xU via NumPy, then from_numpy (fast, no warning) ----
    xr = np.asarray(x_range, dtype=np.float32)         # shape (d, 2)
    xL_np = xr[:, 0][None, :]                          # (1, d)
    xU_np = xr[:, 1][None, :]                          # (1, d)

    xL = torch.from_numpy(xL_np).to(device)            # (1, d)
    xU = torch.from_numpy(xU_np).to(device)            # (1, d)
    x0 = (xL + xU) / 2

    ptb = PerturbationLpNorm(norm=np.inf, eps=None, x_L=xL, x_U=xU)
    bx  = BoundedTensor(x0, ptb)

    lb, ub = bm.compute_bounds(x=(bx,), method="CROWN")
    lb_np = lb.squeeze(0).detach().cpu().numpy()
    ub_np = ub.squeeze(0).detach().cpu().numpy()
    return ub_np, lb_np


def _sum_weighted_interval(weights: np.ndarray, intervals: np.ndarray) -> Tuple[float, float]:
    """
    Tight sum_i w_i * [l_i, u_i] -> [L, U].
    For each i, if w_i >= 0 choose l_i for the lower sum and u_i for the upper; swap if w_i < 0.
    weights:   (m,)
    intervals: (m, 2) with [:,0]=low, [:,1]=high
    Returns (L, U) scalars.
    """
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    Ls = np.where(w >= 0, intervals[:, 0], intervals[:, 1]) * w
    Us = np.where(w >= 0, intervals[:, 1], intervals[:, 0]) * w
    return float(Ls.sum()), float(Us.sum())


def _pairwise_scalar_interval(a_low: float, a_high: float, b_low: float, b_high: float) -> Tuple[float, float]:
    """
    Tight interval of scalar product [a_low,a_high] * [b_low,b_high].
    """
    p = np.array([a_low * b_low, a_low * b_high, a_high * b_low, a_high * b_high], dtype=np.float64)
    return float(p.min()), float(p.max())


# Tight cubic extremizer for sigma''(z) over h in [l, u]
# sigma'(z) = h(1-h);   sigma''(z) = (1 - 2h) * h(1-h) = h - 3h^2 + 2h^3
_hcrit_lo = 0.5 - 0.5/np.sqrt(3.0)  # ~0.211324865
_hcrit_hi = 0.5 + 0.5/np.sqrt(3.0)  # ~0.788675135
def _sigma2_cubic(h):                # (1-2h)*h*(1-h)
    return (1.0 - 2.0*h) * h * (1.0 - h)

def _tight_sigma2_bounds(lb: np.ndarray, ub: np.ndarray):
    """
    Compute tight per-neuron bounds for sigma'' over h in [lb, ub] using exact
    cubic extremizers. Vectorized over neurons.
    Returns arrays (N,) for (low, high).
    """
    # Candidates per-neuron: endpoints and critical points that fall inside [lb, ub]
    # Stack candidates to a (N,M) matrix, evaluate cubic, take rowwise min/max.
    cands = [lb, ub]
    mask_lo = (lb <= _hcrit_lo) & (_hcrit_lo <= ub)
    mask_hi = (lb <= _hcrit_hi) & (_hcrit_hi <= ub)

    # Build candidate matrix with broadcasting-safe fills
    cols = [lb, ub]
    if np.any(mask_lo):
        hlo = np.where(mask_lo, _hcrit_lo, lb)  # values for rows where present; dummy otherwise
        cols.append(hlo)
    if np.any(mask_hi):
        hhi = np.where(mask_hi, _hcrit_hi, ub)
        cols.append(hhi)

    C = np.vstack(cols).T  # shape (N, M)
    vals = _sigma2_cubic(C)
    return np.min(vals, axis=1), np.max(vals, axis=1)


# ---------------------------- Core bound computation --------------------------

def _core_operation_crown_vectorize(const, net, x_range, device="cpu"):
    """
    Vectorized generator bound using:
      - CROWN-IBP for hidden activation intervals [lb, ub]
      - tight sigma' bounds (concavity) and tight sigma'' bounds (cubic extremizers)
      - interval algebra for drift/diffusion products
    Returns
    -------
    GV_bound : np.ndarray shape (2,), [low, high] bound of G[phi] on x_range.
    """
    # --- 1) Hidden activation bounds via CROWN-IBP over the box ---
    ub, lb = crown_hidden_bounds(net, x_range, device=device)  # (m,), (m,)
    lb = np.minimum(lb, ub)  # guard ordering
    ub = np.maximum(lb, ub)

    # --- 2) Weights/constants ---
    W0 = (net.Layers[0].weight.detach().cpu().numpy() / 100.0)  # (m, d)
    w0_x1 = W0[:, 0]
    w0_x2 = W0[:, 1]
    W1 = net.Layers[1].weight.detach().cpu().numpy()[0]         # (m,)

    f1_low, f1_high = const.get_f1_bound(x_range)
    f2_low, f2_high = const.get_f2_bound(x_range)
    g11_low, g11_high = const.get_g11square_bound(x_range)
    g22_low, g22_high = const.get_g22square_bound(x_range)

    # --- 3) Derivative bounds from activation interval [lb, ub] ---
    # sigma'(z) = h(1-h): exact tight bound via concavity on [0,1]
    lb_term = lb * (1 - lb)
    ub_term = ub * (1 - ub)
    mid_mask = (lb <= 0.5) & (ub >= 0.5)
    ub_dhi = np.where(mid_mask, 0.25, np.maximum(lb_term, ub_term))
    lb_dhi = np.minimum(lb_term, ub_term)

    # stack for sign-aware first-derivative bounds
    dhi_base     = np.stack([lb_dhi, ub_dhi], axis=1)  # (m,2)
    dhi_base_rev = dhi_base[:, [1, 0]]

    sign1 = (w0_x1 >= 0)[:, None]
    sign2 = (w0_x2 >= 0)[:, None]
    dhi_dx1 = np.where(sign1, dhi_base, dhi_base_rev) * w0_x1[:, None]
    dhi_dx2 = np.where(sign2, dhi_base, dhi_base_rev) * w0_x2[:, None]

    # sigma''(z): tight per-neuron bound via cubic extremizers over h in [lb,ub]
    # then scale by (w/100)^2 (nonnegative) -> no bound swapping needed
    sig2_low, sig2_high = _tight_sigma2_bounds(lb, ub)
    dhi2 = np.stack([sig2_low, sig2_high], axis=1)     # (m,2)

    # --- 4) Compose the generator bounds (tighter aggregation) ---
    # First-derivative parts per neuron: [dphi/dx1]_k and [dphi/dx2]_k intervals
    # already include the correct sign of w0_xj via dhi_dx{1,2}.
    # Aggregate across neurons WITH W1 weights before multiplying by f1, f2.
    S_dx1_low, S_dx1_high = _sum_weighted_interval(W1, dhi_dx1)   # scalars
    S_dx2_low, S_dx2_high = _sum_weighted_interval(W1, dhi_dx2)   # scalars

    # Drift contribution = f1 * S_dx1 + f2 * S_dx2 (both are scalar×interval)
    drift1_low, drift1_high = _pairwise_scalar_interval(f1_low, f1_high, S_dx1_low, S_dx1_high)
    drift2_low, drift2_high = _pairwise_scalar_interval(f2_low, f2_high, S_dx2_low, S_dx2_high)
    drift_low  = drift1_low  + drift2_low
    drift_high = drift1_high + drift2_high

    # Second-derivative intervals per neuron (nonnegative scaling by (w0/100)^2 already baked into scale1/2)
    # Note: dhi2 = [sigma''] interval; multiply by w0_xj^2 >= 0 so no interval swap per neuron
    H1 = np.stack([ (w0_x1**2) * dhi2[:, 0], (w0_x1**2) * dhi2[:, 1] ], axis=1)  # (m,2)
    H2 = np.stack([ (w0_x2**2) * dhi2[:, 0], (w0_x2**2) * dhi2[:, 1] ], axis=1)  # (m,2)

    # Aggregate across neurons with W1 weights BEFORE multiplying by g^2 scalars
    S_H1_low, S_H1_high = _sum_weighted_interval(W1, H1)  # scalars
    S_H2_low, S_H2_high = _sum_weighted_interval(W1, H2)  # scalars

    # Diffusion contribution = 0.5 * ( g11^2 * S_H1 + g22^2 * S_H2 )
    diff1_low, diff1_high = _pairwise_scalar_interval(g11_low, g11_high, S_H1_low, S_H1_high)
    diff2_low, diff2_high = _pairwise_scalar_interval(g22_low, g22_high, S_H2_low, S_H2_high)
    diff_low  = 0.5 * (diff1_low  + diff2_low)
    diff_high = 0.5 * (diff1_high + diff2_high)

    GV_bound = np.array([drift_low + diff_low, drift_high + diff_high], dtype=np.float64)
    GV_bound[0] = min(GV_bound[0], GV_bound[1])  # guard ordering
    return GV_bound


def _unit_verify_crown_differential(const, net, boxes, device="cpu", verbose=False):
    lbs, ubs = [], []
    for x_range in boxes:
        L, U = _core_operation_crown_vectorize(const, net, x_range, device=device)
        lbs.append(L); ubs.append(U)
    lbs = np.asarray(lbs); ubs = np.asarray(ubs)
    if(verbose):
        print("[info] upper bounds of each region: ", ubs)
    return np.array([lbs.min(), ubs.max()]), ubs


# ---------------------------- Public API --------------------------------------
def verify_crown_differential(
    const, net, refine_N: int = 2, max_depth: int = 4,
    device: str = "cpu",
    visualize: bool = True,
    gif_path: str | None = None, gif_fps: float = 1.0,
    show_plot: bool = True,   # <-- NEW: set False to disable any window (GIF-only)
):
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
            GV_bound, per_box_ub = _unit_verify_crown_differential(const, net, boxes, device=device)
            GV_final = GV_bound

            ub = np.asarray(per_box_ub, dtype=float).reshape(-1)
            if ub.size == 0:
                to_refine, to_refine_ub = [], np.empty((0,), dtype=float)
            else:
                idx = np.nonzero(ub >= condition)[0]
                to_refine    = [boxes[i] for i in idx] if idx.size else []
                to_refine_ub = ub[idx] if idx.size else np.empty((0,), dtype=float)

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
