import numpy as np
from src.setoperations import *
"""
[Log]
load model from: output/net_ibp.pth
[info] number of regions:  4
[info] number of regions:  16
[info] number of regions:  64
[info] number of regions:  240
[info] number of regions:  892
[info] number of regions:  2932
[info] number of regions:  9696
[info] number of regions:  28664
[info] number of regions:  74500
[info] number of regions:  158400
[info] number of regions:  165776
[info] number of regions:  38548
[info] number of regions:  728
[complete] GV bound:  [-0.22078211 -0.02622605]
"""


def verify_ibp_differential(const, net):
    condition = 0.0
    # --- Init Region to verifiy ---
    X_DIFFER_RANGE_REFINE = []
    _X_DIFFER_RANGE_REFINE = rect_diff_2d(const.X_RANGE, const.X_GOAL_RANGE)
    for _rect in _X_DIFFER_RANGE_REFINE:
        X_DIFFER_RANGE_REFINE = X_DIFFER_RANGE_REFINE + rect_diff_2d(_rect, const.X_UNSAFE_RANGE)

    for iter in range(100): # 100 the "depth" of refinement process
        GV_bound, upper_bounds_per_partition = _unit_verify_ibp_differential(const, net, X_DIFFER_RANGE_REFINE)
        X_DIFFER_RANGE_REFINE_to_verify = [] # initialize partitions that still need to be verify
        for i in range(len(X_DIFFER_RANGE_REFINE)):
            if(upper_bounds_per_partition[i] >= condition):
                X_DIFFER_RANGE_REFINE_to_verify.append(X_DIFFER_RANGE_REFINE[i])
            # else:
            #     print("[info] verify partition: ", upper_bounds_per_partition[i])
        if(len(X_DIFFER_RANGE_REFINE_to_verify) == 0):
            print("[complete] GV bound: ", GV_bound)
            break
        X_DIFFER_RANGE_REFINE = refine_list_2d_box(X_DIFFER_RANGE_REFINE_to_verify, 2)


def _unit_verify_ibp_differential(const, net, X_DIFFER_RANGE_REFINE):
    print("[info] number of regions: ", len(X_DIFFER_RANGE_REFINE))
    GV_lbs = []
    GV_ubs = []
    for it in range(0, len(X_DIFFER_RANGE_REFINE)):
        x_range = X_DIFFER_RANGE_REFINE[it]
        # --- core operation ---
        # GV_bound_old = _core_operation(const, net, x_range)
        GV_bound = _core_operation_vectorize(const, net, x_range)
        # np.testing.assert_allclose(GV_bound_old, GV_bound)
        # ----------------------
        GV_lbs.append(GV_bound[0])
        GV_ubs.append(GV_bound[1])
    GV_lbs = np.array(GV_lbs)
    GV_ubs = np.array(GV_ubs)
    bound = np.array([GV_lbs.min(), GV_ubs.max()])
    return bound, GV_ubs


def _core_operation(const, net, x_range):
    """
    The vanilla bound propagation of GV based on test_intervalboundpropagate.py
    """
    # --- Bound Propagation --- 
    W2 = net.Layers[1].weight
    GV_bound = np.array([0.0, 0.0])
    for i in range(W2.shape[1]):
        ub, lb = net.bound_propagation_hidden_layer(x_range)
        ub = np.array(ub)[i]
        lb = np.array(lb)[i]
        w_vec = net.Layers[0].weight[i, :] / 100.0
        if(0.5 <= ub and 0.5 >= lb):
            candadate = [lb, 0.5, ub]
        else:
            candadate = [lb, ub]
        if len(candadate) == 3:
            ub_dhi_dx = 0.5*(1-0.5)
        else:
            ub_dhi_dx = max(candadate[0]*(1-candadate[0]), candadate[-1]*(1-candadate[-1]))
        lb_dhi_dx = min(candadate[0]*(1-candadate[0]), candadate[-1]*(1-candadate[-1]))

        b1 = np.array([lb_dhi_dx, ub_dhi_dx])
        b2 = np.array([1 - 2*ub, 1 - 2*lb])
        dhi_dxdx_bound = const.get_bounds_from_product_of_two_bounds(b1, b2)

        f1_bound = const.get_f1_bound(x_range)
        f2_bound = const.get_f2_bound(x_range)
        g11sq_bound = const.get_g11square_bound(x_range)
        g22sq_bound = const.get_g22square_bound(x_range)

        wi_x1 = w_vec[0].detach().numpy()
        wi_x2 = w_vec[1].detach().numpy()
        if(wi_x1 >=0 ):
            dhi_dx1_bound = np.array([lb_dhi_dx, ub_dhi_dx])*wi_x1
        else:
            dhi_dx1_bound = np.array([ub_dhi_dx, lb_dhi_dx])*wi_x1
        if(wi_x2 >=0 ):
            dhi_dx2_bound = np.array([lb_dhi_dx, ub_dhi_dx])*wi_x2
        else:
            dhi_dx2_bound = np.array([ub_dhi_dx, lb_dhi_dx])*wi_x2

        bound_1 = const.get_bounds_from_product_of_two_bounds(f1_bound, dhi_dx1_bound) + \
                const.get_bounds_from_product_of_two_bounds(f2_bound, dhi_dx2_bound)

        bound_2 = 0.5*(wi_x1**2 * const.get_bounds_from_product_of_two_bounds(g11sq_bound, dhi_dxdx_bound) +\
                    wi_x2**2 * const.get_bounds_from_product_of_two_bounds(g22sq_bound, dhi_dxdx_bound))

        Gh_i_bound = bound_1 + bound_2
        W2_i = W2[0,i].detach().numpy()
        if(W2_i >= 0):
            GV_bound += W2_i * Gh_i_bound
        else:
            GV_bound += W2_i * np.array([Gh_i_bound[1], Gh_i_bound[0]])
    return GV_bound


def _core_operation_vectorize(const, net, x_range):
    """
    Optimized version by avoiding for loop
    """
    ub_arr, lb_arr = net.bound_propagation_hidden_layer(x_range)  # each length‑128
    ub = np.asarray(ub_arr)
    lb = np.asarray(lb_arr)

    # network weights as NumPy arrays
    W0 = (net.Layers[0].weight.detach().numpy() / 100.0)  # shape (128, 2)
    w0_x1 = W0[:, 0]
    w0_x2 = W0[:, 1]

    W2 = net.Layers[1].weight.detach().numpy()[0]        # shape (128,)

    # your constant bounds (all scalars)
    f1_low, f1_high = const.get_f1_bound(x_range)
    f2_low, f2_high = const.get_f2_bound(x_range)
    g11_low, g11_high = const.get_g11square_bound(x_range)
    g22_low, g22_high = const.get_g22square_bound(x_range)

    # 2) sigmoid‑derivative bounds for each neuron
    lb_term = lb * (1 - lb)
    ub_term = ub * (1 - ub)
    mid_mask = (lb <= 0.5) & (ub >= 0.5)

    ub_dhi = np.where(mid_mask, 0.25, np.maximum(lb_term, ub_term))
    lb_dhi = np.minimum(lb_term, ub_term)

    # stack into shape (128,2) = [lb_dhi, ub_dhi]
    dhi_base    = np.stack([lb_dhi, ub_dhi], axis=1)
    dhi_base_rev= dhi_base[:, [1,0]]  # flipped bounds

    # 3) first‐derivative bounds multiplied by W0
    sign1 = (w0_x1 >= 0)[:, None]
    dhi_dx1 = np.where(sign1, dhi_base, dhi_base_rev) * w0_x1[:, None]

    sign2 = (w0_x2 >= 0)[:, None]
    dhi_dx2 = np.where(sign2, dhi_base, dhi_base_rev) * w0_x2[:, None]

    # 4) bound_1 = f1⋅dhi_dx1  +  f2⋅dhi_dx2   (vectorized min/max of four products)
    def pairwise_bounds(a_low, a_high, b_bounds):
        """
        Given scalars a_low, a_high and array b_bounds shape (N,2),
        returns (low, high) arrays of shape (N,) for the product bounds.
        """
        b0, b1 = b_bounds[:,0], b_bounds[:,1]
        p1 = a_low  * b0
        p2 = a_low  * b1
        p3 = a_high * b0
        p4 = a_high * b1
        low  = np.minimum.reduce([p1, p2, p3, p4])
        high = np.maximum.reduce([p1, p2, p3, p4])
        return low, high

    b1_low, b1_high = pairwise_bounds(f1_low, f1_high, dhi_dx1)
    b2_low, b2_high = pairwise_bounds(f2_low, f2_high, dhi_dx2)

    bound1_low  = b1_low  + b2_low
    bound1_high = b1_high + b2_high

    # 5) second‐derivative bounds: dhi'' = (1−2u)⋅h'' ranges
    c0 = 1 - 2*ub
    c1 = 1 - 2*lb
    p1 = lb_dhi * c0
    p2 = lb_dhi * c1
    p3 = ub_dhi * c0
    p4 = ub_dhi * c1
    dhi2_low  = np.minimum.reduce([p1,p2,p3,p4])
    dhi2_high = np.maximum.reduce([p1,p2,p3,p4])
    dhi2 = np.stack([dhi2_low, dhi2_high], axis=1)

    # 6) bound_2 = 0.5*( w0_x1^2⋅(g11⋅dhi2)  +  w0_x2^2⋅(g22⋅dhi2) )
    h1_low, h1_high = pairwise_bounds(g11_low, g11_high, dhi2)
    h2_low, h2_high = pairwise_bounds(g22_low, g22_high, dhi2)

    bound2_low  = 0.5*( w0_x1**2 * h1_low  +  w0_x2**2 * h2_low )
    bound2_high = 0.5*( w0_x1**2 * h1_high +  w0_x2**2 * h2_high )

    # 7) per‐neuron Gh bounds
    Gh_low  = bound1_low  + bound2_low
    Gh_high = bound1_high + bound2_high
    Gh      = np.stack([Gh_low, Gh_high], axis=1)
    Gh_rev  = Gh[:, [1,0]]

    # 8) final weighted sum over neurons
    signW2 = (W2 >= 0)[:, None]
    GV_bound = np.sum(
        np.where(signW2,
                W2[:, None] * Gh,
                W2[:, None] * Gh_rev),
        axis=0
    )
    return GV_bound