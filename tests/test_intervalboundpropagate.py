import torch
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import copy

from src.models import IBPNet, load_trained_model
from src.constants import Const_GBM
from src.setoperations import rect_diff_2d, refine_list_2d_box


def test_intervalboundpropagate():
    # torch.manual_seed(0)
    # np.random.seed(0)
    const = Const_GBM()
    net = IBPNet(const, neurons=128)
    batch_size = 10000
    refine_N = 4
    print("\n")

    # --- Initialize (Refined) Ranges: np.array --> List[np.array] ---
    X_INIT_RANGE_REFINE = [const.X_INIT_RANGE]
    X_UNSAFE_RANGE_REFINE = [const.X_UNSAFE_RANGE]

    X_DIFFER_RANGE_REFINE = []
    _X_DIFFER_RANGE_REFINE = rect_diff_2d(const.X_RANGE, const.X_GOAL_RANGE)
    for _rect in _X_DIFFER_RANGE_REFINE:
        X_DIFFER_RANGE_REFINE = X_DIFFER_RANGE_REFINE + rect_diff_2d(_rect, const.X_UNSAFE_RANGE)

    X_OUTSIDEGOAL_RANGE_REFINE = copy.deepcopy(X_DIFFER_RANGE_REFINE)

    # --- Init ---
    x = const.sample_x(const.X_INIT_RANGE, batch_size)
    V = net(x).detach().numpy()
    ub, lb = _get_bounds_over_list_of_rects(X_INIT_RANGE_REFINE, net)
    np.testing.assert_array_less(lb, V.min(), err_msg="IBP invalid: lb is not smaller than V(x init)")
    np.testing.assert_array_less(V.max(), ub, err_msg="IBP invalid: lb is not smaller than V(x init)")
    print("[info] INIT: lower bound {:.4f}, min V {:.4f}, max V {:.4f}, upper bound {:.4f}".format(
        lb, V.min(), V.max(), ub))
    # testing refinements
    X_INIT_RANGE_REFINE = refine_list_2d_box(X_INIT_RANGE_REFINE, refine_N)
    ub, lb = _get_bounds_over_list_of_rects(X_INIT_RANGE_REFINE, net)
    np.testing.assert_array_less(lb, V.min(), err_msg="IBP invalid: lb is not smaller than V(x init)")
    np.testing.assert_array_less(V.max(), ub, err_msg="IBP invalid: lb is not smaller than V(x init)")
    print("[info] INIT (refine): lower bound {:.4f}, min V {:.4f}, max V {:.4f}, upper bound {:.4f}".format(
        lb, V.min(), V.max(), ub))
    
    # --- Unsafe ---
    x = const.sample_x(const.X_UNSAFE_RANGE, batch_size)
    V = net(x).detach().numpy()
    ub, lb = _get_bounds_over_list_of_rects(X_UNSAFE_RANGE_REFINE, net)
    np.testing.assert_array_less(lb, V.min(), err_msg="IBP invalid: lb is not smaller than V(x unsafe)")
    np.testing.assert_array_less(V.max(), ub, err_msg="IBP invalid: lb is not smaller than V(x unsafe)")
    print("[info] UNSAFE: lower bound {:.4f}, min V {:.4f}, max V {:.4f}, upper bound {:.4f}".format(
        lb, V.min(), V.max(), ub))
    # test refinements
    X_UNSAFE_RANGE_REFINE = refine_list_2d_box(X_UNSAFE_RANGE_REFINE, refine_N)
    ub, lb = _get_bounds_over_list_of_rects(X_UNSAFE_RANGE_REFINE, net)
    np.testing.assert_array_less(lb, V.min(), err_msg="IBP invalid: lb is not smaller than V(x unsafe)")
    np.testing.assert_array_less(V.max(), ub, err_msg="IBP invalid: lb is not smaller than V(x unsafe)")
    print("[info] UNSAFE (refined) lower bound {:.4f}, min V {:.4f}, max V {:.4f}, upper bound {:.4f}".format(
        lb, V.min(), V.max(), ub))
    
    # --- Goal ---
    ## condition 1, V(x) x \in X \setminus X_G > Beta_S
    x_full = const.sample_x(const.X_RANGE, batch_size)
    inside_goal = const.filter_sample_insidebound(x_full, const.X_GOAL_RANGE)
    outside_goal = ~inside_goal
    if outside_goal.any():
        x = x_full[outside_goal]
        V = net(x).detach().numpy()
        ubs = []
        lbs = []
        for rect in X_OUTSIDEGOAL_RANGE_REFINE:
            ub, lb = net.bound_propagation_last_layer(rect)
            ub = np.array(ub)[0]
            lb = np.array(lb)[0]
            x_rect = const.sample_x(rect, batch_size)
            V_rect = net(x_rect).detach().numpy()
            np.testing.assert_array_less(lb, V_rect.min(), err_msg="IBP invalid: lb is not smaller than V(x unsafe)")
            np.testing.assert_array_less(V_rect.max(), ub, err_msg="IBP invalid: lb is not smaller than V(x unsafe)")
            # print("[info] GOAL cond1: lower bound {:.4f}, min V {:.4f}, max V {:.4f}, upper bound {:.4f}".format(
            # lb, V_rect.min(), V_rect.max(), ub))
            ubs.append(ub)
            lbs.append(lb)
        ubs = np.array(ubs)
        lbs = np.array(lbs)
        ub = ubs.max(); lb = lbs.min()
        np.testing.assert_array_less(lb, V.min(), err_msg="IBP invalid: lb is not smaller than V(x unsafe)")
        np.testing.assert_array_less(V.max(), ub, err_msg="IBP invalid: lb is not smaller than V(x unsafe)")
        print("[info] GOAL cond1: lower bound {:.4f}, min V {:.4f}, max V {:.4f}, upper bound {:.4f}".format(
            lb, V.min(), V.max(), ub))
        # testing refinement
        X_OUTSIDEGOAL_RANGE_REFINE = refine_list_2d_box(X_OUTSIDEGOAL_RANGE_REFINE, refine_N)
        ub, lb = _get_bounds_over_list_of_rects(X_OUTSIDEGOAL_RANGE_REFINE, net)
        np.testing.assert_array_less(lb, V.min(), err_msg="IBP invalid: lb is not smaller than V(x unsafe)")
        np.testing.assert_array_less(V.max(), ub, err_msg="IBP invalid: lb is not smaller than V(x unsafe)")
        print("[info] GOAL cond1 (refined): lower bound {:.4f}, min V {:.4f}, max V {:.4f}, upper bound {:.4f}".format(
            lb, V.min(), V.max(), ub))
    ## condition 2, `exists' V(x) x \in X_G <= Beta_S: not really need bound propagation
    # x = const.sample_x(const.X_GOAL_RANGE, batch_size)
    # V = net(x).detach().numpy()
    # ub, lb = net.bound_propagation_last_layer(const.X_GOAL_RANGE)
    # ub = np.array(ub)[0]
    # lb = np.array(lb)[0]
    # np.testing.assert_array_less(lb, V.min(), err_msg="IBP invalid: lb is not smaller than V(x unsafe)")
    # np.testing.assert_array_less(V.max(), ub, err_msg="IBP invalid: lb is not smaller than V(x unsafe)")
    # print("[info] GOAL cond2: lower bound {:.4f}, min V {:.4f}, max V {:.4f}, upper bound {:.4f}".format(
    #     lb, V.min(), V.max(), ub))

    # --- Visual ---
    _visual_regions(const, 
                X_INIT_RANGE_REFINE, 
                X_UNSAFE_RANGE_REFINE,
                X_OUTSIDEGOAL_RANGE_REFINE,
                X_DIFFER_RANGE_REFINE, show_plots=False)
    

def test_differential(batch_size=1):
    """
    G[V(x,theta)] = sum_i ( w_i * G[z(x)_i] ) + G[b], 
    where z(x)_i is the input to the last layer, w_i and b are the last layer weights and bias
    if only one hidden layer, then z(x)_i = Act(sum_{j=1}^d w_{ji} x_j + b_i)
    denote y(x)_i = sum_{j=1}^d w_{ji} x_j + b_i, then z(x)_i = Act( y(x)_i ).
    * G[b] = 0: the bias of the last layer has NULL effect

    * G[z] = f1 z_x1 + f2 z_x2 + 0.5*(g_11^2 * z_x1x1 + g_22^2 * z_x2x2)
        * zi_x1 = w1i * sigmoid(y(x)_i) * (1 - sigmoid())
        * zi_x2 = w2i * sigmoid(y(x)_i) * (1 - sigmoid())
        partial[ sigmoid(w_1i * x1 + w_2i * x2 + b_i) ]/ partial x1 = ?
    """
    const = Const_GBM()
    net = IBPNet(const, neurons=128)
    x_full = const.sample_x(const.X_RANGE, 1)
    with torch.no_grad():
        # ----- fetch parameters -----
        W1, b1 = net.Layers[0].weight, net.Layers[0].bias  # shapes [neurons, X_DIM], [neurons]
        W2, b2 = net.Layers[1].weight, net.Layers[1].bias  # shapes [1, neurons],  [1]
        # ----- forward pass step‑by‑step -----
        x_norm  = x_full / 100.0                           # input normalisation
        z1      = torch.matmul(x_norm, W1.t()) + b1   # hidden layer pre‑activation
        z1_test = torch.matmul(x_norm, W1[0,:].t()) + b1[0]
        # print(x_norm.shape, W1.shape, b1.shape, W2.shape, b2.shape)
        # print(z1[:,0]); print(z1_test)
        np.testing.assert_allclose(z1[:,0], z1_test)

    x_full = const.sample_x(const.X_RANGE, 1, requires_grad=True)
    # -------------------------------------------------------------
    # Forward pass: scalar output h_i(x)
    # -------------------------------------------------------------
    i = 7              
    z_i = net.Layers[0](x_full / 100.0)[0, i]   # pre‑activation
    h_i = torch.sigmoid(z_i)                    # scalar
    # -------------------------------------------------------------
    # 1) FIRST derivatives  ∂h_i / ∂x_j
    # -------------------------------------------------------------
    dh_dx = torch.autograd.grad(
        h_i, x_full, create_graph=True
    )[0].squeeze()           # shape (d,)
    # closed‑form: (w_{ji}/100) · σ(z) · (1‑σ(z))
    sigma = h_i.detach()                    # same as torch.sigmoid(z_i)
    w_vec = net.Layers[0].weight[i, :] / 100.0
    dh_dx_manual = w_vec * sigma * (1 - sigma)  # shape (d,)
    # print(dh_dx, dh_dx_manual)
    np.testing.assert_allclose(
        dh_dx.detach().cpu().numpy(),
        dh_dx_manual.detach().cpu().numpy(),
        rtol=1e-6, atol=1e-8
    )
    print("✓ first‑order derivatives match")
    # -------------------------------------------------------------
    # 2) SECOND derivatives  ∂²h_i / ∂x_j²   (j = 1,2)
    # -------------------------------------------------------------
    d2h_dx2 = torch.autograd.functional.hessian(
        lambda x: torch.sigmoid(net.Layers[0](x / 100.0)[0, i]),
        x_full
    ).squeeze().diagonal()[:2]               # tensor([∂²/∂x₁², ∂²/∂x₂²])
    # closed‑form: (w_{ji}/100)² · σ(z)·(1‑σ(z))·(1‑2σ(z))
    factor = sigma * (1 - sigma) * (1 - 2 * sigma)
    # print(w_vec.shape)
    d2_manual = factor * (w_vec ** 2)
    # print(d2h_dx2, d2_manual)
    np.testing.assert_allclose(
        d2h_dx2.detach().cpu().numpy(),
        d2_manual.detach().cpu().numpy(),
        rtol=1e-6, atol=1e-8
    )
    print("✓ second‑order derivatives match")

    GV_list = []
    GV_manual_list = []
    for points in range(batch_size):
        x_full = const.sample_x(const.X_RANGE, 1, requires_grad=True)  # shape [1, d] (d ≥ 2)
        # Auto: GV
        GV = _diff_operator(const, net, x_full)

        # GV manual
        GV_manual = torch.zeros(1)
        W2, _ = net.Layers[1].weight, net.Layers[1].bias  # shapes [1, neurons],  [1]
        A = torch.tensor(const.A_MATRIX, dtype=torch.float32, requires_grad=False)
        sigma = torch.tensor(const.SIGMA, dtype=torch.float32, requires_grad=False)
        x1 = x_full[:,0]
        x2 = x_full[:,1]
        f1 = A[0,0]*x1 + A[0,1]*x2
        f2 = A[1,0]*x1 + A[1,1]*x2
        g11 = sigma*x1
        g22 = sigma*x2
        for i in range(net.Layers[0].bias.shape[0]):
            z_i = net.Layers[0](x_full / 100.0)[0, i]   # pre‑activation
            h_i = torch.sigmoid(z_i)                    # scalar
            # closed‑form: (w_{ji}/100) · σ(z) · (1‑σ(z))
            h_i = h_i.detach()                    # same as torch.sigmoid(z_i)
            w_vec = net.Layers[0].weight[i, :] / 100.0
            dh_dx_manual = w_vec * h_i * (1 - h_i)  # shape (d,)
            factor = h_i * (1 - h_i) * (1 - 2 * h_i)
            d2_manual = factor * (w_vec ** 2)
            GV_manual += W2[0,i] * (f1*dh_dx_manual[0] +\
                                    f2*dh_dx_manual[1] +\
                                    0.5*(g11**2 * d2_manual[0] +\
                                         g22**2 * d2_manual[1]))
        GV_list.append(GV.item())
        GV_manual_list.append(GV_manual.item())
    np.testing.assert_allclose(np.array(GV_list), np.array(GV_manual_list),rtol=1e-2, atol=1e-3)
    

def test_intervalboundpropagate_differential(batch_size=10000, N_refine=10):
    torch.manual_seed(0)
    np.random.seed(0)
    delta = 1e-6
    const = Const_GBM()

    net = IBPNet(const, neurons=128)
    # net = load_trained_model(net, "output/net_ibp.pth")

    # --- Define x range ---
    X_DIFFER_RANGE_REFINE = []
    _X_DIFFER_RANGE_REFINE = rect_diff_2d(const.X_RANGE, const.X_GOAL_RANGE)
    for _rect in _X_DIFFER_RANGE_REFINE:
        X_DIFFER_RANGE_REFINE = X_DIFFER_RANGE_REFINE + rect_diff_2d(_rect, const.X_UNSAFE_RANGE)
    X_DIFFER_RANGE_REFINE = refine_list_2d_box(X_DIFFER_RANGE_REFINE, N_refine)
    
    GV_mins = []
    GV_maxs = []
    GV_lbs = []
    GV_ubs = []
    print("[info] number of regions: ", len(X_DIFFER_RANGE_REFINE))
    for it in range(0, len(X_DIFFER_RANGE_REFINE)):
        x_range = X_DIFFER_RANGE_REFINE[it]
        # print("[info] check {:2d}-th region".format(it))

        x_full = const.sample_x(x_range, batch_size, requires_grad=True)
        GV = _diff_operator(const, net, x_full).detach().numpy()

        # --- Bound Propagation --- 
        W2 = net.Layers[1].weight
        GV_bound = np.array([0.0, 0.0])
        # a batch_size output of the i-th nueron of the hidden layer
        for i in range(128):
            z_i = net.Layers[0](x_full / 100.0)[:, i]
            h_i = torch.sigmoid(z_i).detach().numpy()                    # scalar
            # compute bound h_i
            ub, lb = net.bound_propagation_hidden_layer(x_range)
            ub = np.array(ub)[i]
            lb = np.array(lb)[i]
            np.testing.assert_array_less(lb, h_i.min()+delta)
            np.testing.assert_array_less(h_i.max(), ub+delta)
            
            # bound dhi_dx (the compoenet with out constant factor mulitplied by weight)
            w_vec = net.Layers[0].weight[i, :] / 100.0
            dhi_dx = h_i * (1 - h_i) 
            if(0.5 <= ub and 0.5 >= lb):
                candadate = [lb, 0.5, ub]
            else:
                candadate = [lb, ub]
            if len(candadate) == 3:
                ub_dhi_dx = 0.5*(1-0.5)
            else:
                ub_dhi_dx = max(candadate[0]*(1-candadate[0]), candadate[-1]*(1-candadate[-1]))
            lb_dhi_dx = min(candadate[0]*(1-candadate[0]), candadate[-1]*(1-candadate[-1]))
            # ub_dhi_dx *= w_vec
            # lb_dhi_dx *= w_vec
            np.testing.assert_array_less(lb_dhi_dx, dhi_dx.min()+delta)
            np.testing.assert_array_less(dhi_dx.max(), ub_dhi_dx+delta)
            # print(lb_dhi_dx, dhi_dx.min(), dhi_dx.max(), ub_dhi_dx)

            # bound dhi_dxdx (the compoenet with out constant factor mulitplied by weight**2)
            dhi_dxdx = (h_i * (1 - h_i)) * (1 - 2*h_i)
            b1 = np.array([lb_dhi_dx, ub_dhi_dx])
            b2 = np.array([1 - 2*ub, 1 - 2*lb])
            dhi_dxdx_bound = const.get_bounds_from_product_of_two_bounds(b1, b2)
            # ub_dhi_dx *= w_vec**2
            # lb_dhi_dx *= w_vec**2
            np.testing.assert_array_less(dhi_dxdx_bound[0], dhi_dxdx.min()+delta)
            np.testing.assert_array_less(dhi_dxdx.max(), dhi_dxdx_bound[1]+delta)
            # print(dhi_dxdx_bound[0], dhi_dxdx.min(), dhi_dxdx.max(), dhi_dxdx_bound[1])

            # write a function to compute f1, f2, g11^2, g22^2 bounds
            f1_bound = const.get_f1_bound(x_range)
            f2_bound = const.get_f2_bound(x_range)
            g11sq_bound = const.get_g11square_bound(x_range)
            g22sq_bound = const.get_g22square_bound(x_range)

            # write a function to compute G[z(x)_i] bounds,
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
        np.testing.assert_array_less(GV_bound[0], GV.min()+delta)
        np.testing.assert_array_less(GV.max(), GV_bound[1]+delta)
        GV_mins.append(GV.min())
        GV_maxs.append(GV.max())
        GV_lbs.append(GV_bound[0])
        GV_ubs.append(GV_bound[1])
    # --- Print summary ---
    GV_mins = np.array(GV_mins)
    GV_maxs = np.array(GV_maxs)
    GV_lbs = np.array(GV_lbs)
    GV_ubs = np.array(GV_ubs)
    print("[info] DIFFERENTIAL: (refined with {:1d}-order) lower bound {:.4f}, min GV {:.4f}, max GV {:.4f}, upper bound {:.4f}".format(
        N_refine, GV_lbs.min(), GV_mins.min(), GV_maxs.max(), GV_ubs.max()))


def _diff_operator(const, net, x):
    """
    infinitesimal generator of the GBM system, G[V(x)].
    """
    V = net(x)
    x1 = x[:,0].view(-1,1)
    x2 = x[:,1].view(-1,1)
    V_x = torch.autograd.grad(V, x, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    V_x1 = V_x[:,0].view(-1,1)
    V_x2 = V_x[:,1].view(-1,1)
    
    # Compute the second derivative (Hessian) of p with respect to x
    hessian = []
    for i in range(V_x.size(1)):
        grad2 = torch.autograd.grad(V_x[:, i], x, grad_outputs=torch.ones_like(V_x[:, i]), create_graph=True)[0]
        hessian.append(grad2)
    V_xx = torch.stack(hessian, dim=-1)
    V_x1x1 = V_xx[:, 0, 0].view(-1,1)
    V_x2x2 = V_xx[:, 1, 1].view(-1,1)

    A = torch.tensor(const.A_MATRIX, dtype=torch.float32, requires_grad=False)
    f1 = torch.reshape(A[0,0]*x1 + A[0,1]*x2, (-1,1))
    f2 = torch.reshape(A[1,0]*x1 + A[1,1]*x2, (-1,1))
    sigma = torch.tensor(const.SIGMA, dtype=torch.float32, requires_grad=False)
    g11 = sigma*x1
    g22 = sigma*x2
    GV = f1*V_x1 + f2*V_x2 #+ 0.5*(g11*g11*V_x1x1 + g22*g22*V_x2x2)
    return GV


def _get_bounds_over_list_of_rects(list_of_rects, net):
    ubs = []
    lbs = []
    for rect in list_of_rects:
        ub, lb = net.bound_propagation_last_layer(rect)
        ub = np.array(ub)[0]
        lb = np.array(lb)[0]
        ubs.append(ub)
        lbs.append(lb)
    ubs = np.array(ubs)
    lbs = np.array(lbs)
    ub = ubs.max(); lb = lbs.min()
    return ub, lb


def _draw_2dboxes(
    ax, 
    list_2d_boxes, 
    color='red', 
    facecolor='none', 
    linewidth=1, 
    autoscale=True, 
    **kwargs
):
    for box in list_2d_boxes:
        x0, x1 = box[0]
        y0, y1 = box[1]

        rect = patches.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            edgecolor=color,
            facecolor=facecolor,
            linewidth=linewidth,
            **kwargs
        )
        ax.add_patch(rect)

    if autoscale:
        ax.relim()
        ax.autoscale_view()

    return ax


def _visual_regions(const, X_INIT, X_UNSAFE, X_OUTSIDEGOAL, X_DIFF, show_plots=False):
    if(show_plots):
        fig, ax = plt.subplots(figsize=(6,6))
        # print(X_DIFF)
        _draw_2dboxes(ax, X_INIT, color='blue', linewidth=1)
        _draw_2dboxes(ax, X_UNSAFE, color='red', linewidth=1)
        _draw_2dboxes(ax, X_OUTSIDEGOAL, color='green', linewidth=1)
        _draw_2dboxes(ax, X_DIFF, color='cyan', linewidth=2)
        ax.set_aspect('equal', 'box')
        plt.xlim(const.X_RANGE[0,:])
        plt.ylim(const.X_RANGE[1,:])
        plt.show()