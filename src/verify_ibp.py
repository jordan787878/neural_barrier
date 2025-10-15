import numpy as np
from src.setoperations import rect_diff_2d, refine_list_2d_box
from src.polyviewer import _poly_viewer_setup, _poly_viewer_update
import matplotlib.pyplot as plt
from matplotlib import animation


def _sum_weighted_interval(weights: np.ndarray, intervals: np.ndarray):
    """
    Tight sum_i w_i * [l_i, u_i] -> (L, U).
    weights:   (m,)
    intervals: (m, 2) with [:,0]=low, [:,1]=high
    """
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    iv = np.asarray(intervals, dtype=np.float64)
    Ls = np.where(w >= 0, iv[:, 0], iv[:, 1]) * w
    Us = np.where(w >= 0, iv[:, 1], iv[:, 0]) * w
    return float(Ls.sum()), float(Us.sum())


def _pairwise_scalar_interval(a_low, a_high, b_low, b_high):
    """Tight interval product [a_low,a_high] * [b_low,b_high]."""
    p = np.array([a_low*b_low, a_low*b_high, a_high*b_low, a_high*b_high], dtype=np.float64)
    return float(p.min()), float(p.max())


def _core_operation_ibp_vectorize(const, net, x_range):
    """
    IBP hidden bounds + simple interval algebra, but with *tight aggregation*:
      sum across neurons with W1 first, then multiply by [f], [g^2].
    Returns np.array([low, high]).
    """
    # --- hidden activation intervals (IBP) ---
    ub_hidden, lb_hidden = net.bound_propagation_hidden_layer(x_range)  # returns arrays per neuron
    ub_hidden = np.asarray(ub_hidden, dtype=np.float64).reshape(-1)
    lb_hidden = np.asarray(lb_hidden, dtype=np.float64).reshape(-1)
    # ensure ordering
    lb = np.minimum(lb_hidden, ub_hidden)
    ub = np.maximum(lb_hidden, ub_hidden)

    # --- weights / constants ---
    W0 = (net.Layers[0].weight.detach().cpu().numpy() / 100.0)  # (m, 2)
    w1 = net.Layers[1].weight.detach().cpu().numpy()[0]         # (m,)
    s1 = W0[:, 0]; s2 = W0[:, 1]

    f1_low, f1_high = const.get_f1_bound(x_range)
    f2_low, f2_high = const.get_f2_bound(x_range)
    g11_low, g11_high = const.get_g11square_bound(x_range)
    g22_low, g22_high = const.get_g22square_bound(x_range)

    # --- sigma'(z) bounds via concavity of d(h)=h(1-h) on [0,1] ---
    d_lb = lb * (1 - lb)
    d_ub = ub * (1 - ub)
    crosses_half = (lb <= 0.5) & (ub >= 0.5)
    ub_dhi = np.where(crosses_half, 0.25, np.maximum(d_lb, d_ub))
    lb_dhi = np.minimum(d_lb, d_ub)
    dhi_base     = np.stack([lb_dhi, ub_dhi], axis=1)   # (m,2)
    dhi_base_rev = dhi_base[:, [1, 0]]

    # per-neuron intervals for ∂V/∂x_j (sign-aware by s_j)
    I_dx1 = np.where((s1 >= 0)[:, None], dhi_base, dhi_base_rev) * s1[:, None]
    I_dx2 = np.where((s2 >= 0)[:, None], dhi_base, dhi_base_rev) * s2[:, None]

    # --- sigma''(z)=(1-2h)h(1-h) via *IBP four-corner* (not the cubic tightener) ---
    one_minus_2h = np.stack([1 - 2*ub, 1 - 2*lb], axis=1)  # interval for (1-2h)
    # interval multiply: [lb_dhi,ub_dhi] * [1-2ub, 1-2lb]
    # four corner product per neuron
    p11 = lb_dhi * (1 - 2*ub)
    p12 = lb_dhi * (1 - 2*lb)
    p21 = ub_dhi * (1 - 2*ub)
    p22 = ub_dhi * (1 - 2*lb)
    sig2_low = np.minimum.reduce([p11, p12, p21, p22])
    sig2_high = np.maximum.reduce([p11, p12, p21, p22])
    J_sig2 = np.stack([sig2_low, sig2_high], axis=1)  # (m,2)

    # scale sigma'' by s_j^2 (nonnegative -> no swap)
    H1 = np.stack([ (s1**2) * J_sig2[:, 0], (s1**2) * J_sig2[:, 1] ], axis=1)  # (m,2)
    H2 = np.stack([ (s2**2) * J_sig2[:, 0], (s2**2) * J_sig2[:, 1] ], axis=1)  # (m,2)

    # --- tight aggregation: sum across neurons *then* multiply by dynamics ---
    # sum over k with W1 signs first
    S_dx1_low, S_dx1_high = _sum_weighted_interval(w1, I_dx1)
    S_dx2_low, S_dx2_high = _sum_weighted_interval(w1, I_dx2)
    S_H1_low,  S_H1_high  = _sum_weighted_interval(w1, H1)
    S_H2_low,  S_H2_high  = _sum_weighted_interval(w1, H2)

    # drift = f1*S_dx1 + f2*S_dx2
    d1_low, d1_high = _pairwise_scalar_interval(f1_low, f1_high, S_dx1_low, S_dx1_high)
    d2_low, d2_high = _pairwise_scalar_interval(f2_low, f2_high, S_dx2_low, S_dx2_high)
    drift_low, drift_high = d1_low + d2_low, d1_high + d2_high

    # diffusion = 0.5 * ( g11^2 * S_H1 + g22^2 * S_H2 )
    m1_low, m1_high = _pairwise_scalar_interval(g11_low, g11_high, S_H1_low, S_H1_high)
    m2_low, m2_high = _pairwise_scalar_interval(g22_low, g22_high, S_H2_low, S_H2_high)
    diff_low, diff_high = 0.5*(m1_low + m2_low), 0.5*(m1_high + m2_high)

    out = np.array([drift_low + diff_low, drift_high + diff_high], dtype=np.float64)
    out[0] = min(out[0], out[1])  # guard ordering
    return out


def _unit_verify_ibp_differential(const, net, boxes, verbose=False):
    lbs, ubs = [], []
    for x_range in boxes:
        L, U = _core_operation_ibp_vectorize(const, net, x_range)
        lbs.append(L); ubs.append(U)
    lbs = np.asarray(lbs); ubs = np.asarray(ubs)
    if verbose:
        print("[info] upper bounds per region:", ubs)
    return np.array([lbs.min(), ubs.max()]), ubs


def verify_ibp_differential(
    const, net, refine_N: int = 2, max_depth: int = 3,
    visualize: bool = True,
    gif_path: str | None = None, gif_fps: float = 1.0,
    show_plot: bool = True,              # False => GIF-only, no window
):
    """
    IBP verifier with tight sum-then-multiply aggregation and optional live/GIF viz.
    API mirrors verify_crown_differential.
    """
    # Initial cover: X_RANGE \ (X_GOAL ∪ X_UNSAFE)
    boxes = []
    for rect in rect_diff_2d(const.X_RANGE, const.X_GOAL_RANGE):
        boxes += rect_diff_2d(rect, const.X_UNSAFE_RANGE)

    condition = 0.0
    GV_final = None

    need_figure = visualize or (gif_path is not None)
    if need_figure:
        fig, ax, pc, cbar, gv_txt = _poly_viewer_setup(const.X_RANGE, cmap_name="cool", show_plot=show_plot)

    writer_ctx = None
    if gif_path is not None and need_figure:
        writer = animation.PillowWriter(fps=gif_fps)
        writer_ctx = writer.saving(fig, gif_path, dpi=110)
        writer_ctx.__enter__()

    try:
        for it in range(max_depth):
            GV_bound, per_box_ub = _unit_verify_ibp_differential(const, net, boxes, verbose=False)
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
                    do_pause=show_plot
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

        if need_figure and show_plot:
            ax.set_xlabel(ax.get_xlabel() + "  |  done ✓")
            fig.canvas.draw()
            try: plt.show()
            except Exception: pass

    print("[info] GV bound:", GV_final)
    return GV_final
