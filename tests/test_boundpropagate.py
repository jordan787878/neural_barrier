import os
import numpy as np
import torch
import pytest

# Your code
pytest.importorskip("auto_LiRPA")  # verify_crown depends on it
from src.constants import Const_GBM
from src.models import IBPNet
from src.verify_crown import _core_operation_crown_vectorize
from src.verify_ibp import _core_operation_ibp_vectorize
from src.verify_crown_direct import _bound_phi_on_box
from src.torch_dyn_wrappers import attach_torch_dynamics


def _load_ibpnet_from_ckpt(const, ckpt_path: str, device: str = "cpu") -> IBPNet:
    """
    Load a pretrained IBPNet from a checkpoint at `ckpt_path`.
    - Accepts either a pure state_dict or {"model_state_dict": ...}.
    - Infers hidden width from 'Layers.0.weight' shape.
    """
    if not os.path.exists(ckpt_path):
        pytest.skip(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict):
        # Could be a raw state_dict already
        state = ckpt
    else:
        raise RuntimeError("Unrecognized checkpoint format.")

    # Infer hidden width from state dict
    try:
        hidden = int(state["Layers.0.weight"].shape[0])
    except KeyError as e:
        raise KeyError(f"Missing key in checkpoint: {e}. "
                       f"Expecting 'Layers.0.weight' etc. Did you save the right model?")

    # Build model with the same hidden size and load weights
    net = IBPNet(const, neurons=hidden)
    net.load_state_dict(state, strict=True)
    net.to(device).eval()
    return net


def _true_generator_on_grid_autograd(const, net, x_range, nx=100, ny=100, device="cpu"):
    """
    Ground truth G[V] using your dynamics:
      f(x) = A x,
      g11 = σ x1, g22 = σ x2 (diagonal diffusion).
    """
    net = net.to(device).eval()

    (x0, x1), (y0, y1) = np.asarray(x_range, float)
    xs = torch.linspace(float(x0), float(x1), nx, device=device)
    ys = torch.linspace(float(y0), float(y1), ny, device=device)
    X, Y = torch.meshgrid(xs, ys, indexing="xy")             # (nx, ny)
    pts = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1).requires_grad_(True)  # (N,2)

    V = net(pts)                                             # (N,1)
    ones = torch.ones_like(V)
    grad = torch.autograd.grad(V, pts, grad_outputs=ones, create_graph=True)[0]    # (N,2)

    # Diagonal of Hessian
    d2x1 = torch.autograd.grad(grad[:, 0], pts, grad_outputs=torch.ones_like(grad[:, 0]),
                               create_graph=True)[0][:, 0]
    d2x2 = torch.autograd.grad(grad[:, 1], pts, grad_outputs=torch.ones_like(grad[:, 1]),
                               create_graph=True)[0][:, 1]

    x1v, x2v = pts[:, 0], pts[:, 1]
    A = torch.tensor(const.A_MATRIX, dtype=pts.dtype, device=device)
    sigma = torch.tensor(const.SIGMA, dtype=pts.dtype, device=device)

    f1 = A[0, 0] * x1v + A[0, 1] * x2v
    f2 = A[1, 0] * x1v + A[1, 1] * x2v
    g11_sq = (sigma * x1v) ** 2
    g22_sq = (sigma * x2v) ** 2

    G = f1 * grad[:, 0] + f2 * grad[:, 1] + 0.5 * (g11_sq * d2x1 + g22_sq * d2x2)  # (N,)
    return G.reshape(nx, ny).detach().cpu().numpy()


def test_small_region_bounds_enclose_ground_truth_pretrained():
    """
    Load your pretrained IBPNet (output/net_ibp.pth), then verify the CROWN bound
    conservatively encloses the true min/max of G[V] on a small 2D box.

    Current result:
        [box] x∈[-8.0, -6.0], y∈[5.0, 7.0]
        [bound] verifier: [-2.46909, 2.84701]  |  truth: [0.16016, 0.21914]
    """
    torch.set_num_threads(1)
    device = "cpu"

    const = Const_GBM()
    attach_torch_dynamics(const)
    ckpt_path = "output/net_ibp.pth"
    net = _load_ibpnet_from_ckpt(const, ckpt_path, device=device)
    
    N_trial = 100
    low = -100.0
    high = 100.0
    rng = np.random.default_rng()

    for ii in range(N_trial):
        # X axis
        x1, x2 = rng.uniform(low, high, size=2)
        x_lo, x_hi = (x1, x2) if x1 <= x2 else (x2, x1)
        # Y axis
        y1, y2 = rng.uniform(low, high, size=2)
        y_lo, y_hi = (y1, y2) if y1 <= y2 else (y2, y1)
        x_range = np.array([[x_lo, x_hi],
                    [y_lo, y_hi]], dtype=float)
        print(x_range)
    
        # Ground truth via autograd on a dense grid
        G = _true_generator_on_grid_autograd(const, net, x_range, nx=81, ny=81, device=device)
        gt_min, gt_max = float(np.min(G)), float(np.max(G))
        tol = 1e-8
        print("\n")
        print(f"[pretrained] hidden={net.Layers[0].out_features}")
        print(f"[box] x∈[{x_range[0,0]}, {x_range[0,1]}], y∈[{x_range[1,0]}, {x_range[1,1]}]")
        print("\n")
        
        # --- Using IBP ---
        Lb, Ub = _core_operation_ibp_vectorize(const, net, x_range)
        Lb, Ub = float(Lb), float(Ub)
        print("IBP")
        assert Lb <= Ub, "Bound ordering violated (L > U)."
        assert (Lb - tol) <= gt_min, f"Lower bound not conservative: L={Lb:g} > min_true={gt_min:g}"
        assert gt_max <= (Ub + tol), f"Upper bound not conservative: max_true={gt_max:g} > U={Ub:g}"
        print(f"[bound] verifier: [{Lb:.6g}, {Ub:.6g}]  |  truth: [{gt_min:.6g}, {gt_max:.6g}]")

        # --- Using CROWN ---
        Lb, Ub = _core_operation_crown_vectorize(const, net, x_range, device=device)
        Lb, Ub = float(Lb), float(Ub)
        print("CROWN")
        assert Lb <= Ub, "Bound ordering violated (L > U)."
        assert (Lb - tol) <= gt_min, f"Lower bound not conservative: L={Lb:g} > min_true={gt_min:g}"
        assert gt_max <= (Ub + tol), f"Upper bound not conservative: max_true={gt_max:g} > U={Ub:g}"
        print(f"[bound] verifier: [{Lb:.6g}, {Ub:.6g}]  |  truth: [{gt_min:.6g}, {gt_max:.6g}]")

        # --- Direct CROWN-on-Φ ---
        Ld, Ud = _bound_phi_on_box(const, net, x_range, device=device)
        Ld, Ud = float(Ld), float(Ud)
        print("CROWN (direct)")
        assert Ld <= Ud, "Bound ordering violated (L > U)."
        assert (Ld - tol) <= gt_min, f"Lower bound not conservative: L={Ld:g} > min_true={gt_min:g}"
        assert gt_max <= (Ud + tol), f"Upper bound not conservative: max_true={gt_max:g} > U={Ud:g}"
        print(f"[bound] verifier (direct): [{Ld:.6g}, {Ud:.6g}]  |  truth: [{gt_min:.6g}, {gt_max:.6g}]")

