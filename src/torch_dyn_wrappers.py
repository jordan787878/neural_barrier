# src/torch_dyn_wrappers.py
# ---------------------------------------------------------------------
# Torch-callable adapters for dynamics:
#   f_of_x(x) -> (f1, f2)          with shapes (N,1)
#   gamma_sq_of_x(x) -> (g11_sq, g22_sq) with shapes (N,1)
#
# Works with your Const_GBM (linear drift f(x)=A x, diagonal diffusion g=σ·diag(x)).
# Keeps everything differentiable for end-to-end CROWN-on-Φ.
# ---------------------------------------------------------------------

from __future__ import annotations
import torch
import torch.nn as nn

class TorchDynamicsAdapter(nn.Module):
    """
    Wraps a `const` that exposes A_MATRIX and SIGMA so it becomes torch-callable.
    Intended for 2D state (N,2). Dtypes/devices follow the input x.
    """
    def __init__(self, const):
        super().__init__()
        # store as buffers to keep them inside the verification graph
        A = torch.tensor(const.A_MATRIX, dtype=torch.float32)
        sigma = torch.tensor(const.SIGMA, dtype=torch.float32)
        self.register_buffer("A", A)          # (2,2)
        self.register_buffer("sigma", sigma)  # scalar
        self._const = const                   # optional: keep original around

    def f_of_x(self, x: torch.Tensor):
        """
        Returns (f1, f2), each (N,1). Uses f(x)=A x, where A is (2,2).
        """
        A = self.A.to(device=x.device, dtype=x.dtype)
        f = x @ A.t()           # (N,2)
        return f[:, :1], f[:, 1:2]

    def gamma_sq_of_x(self, x: torch.Tensor):
        """
        Returns (g11_sq, g22_sq), each (N,1).
        For GBM with g=σ·diag(x), the squared entries are (σ x_j)^2.
        """
        s2 = (self.sigma.to(device=x.device, dtype=x.dtype)) ** 2
        xsq = x * x             # (N,2)
        g11_sq = s2 * xsq[:, :1]
        g22_sq = s2 * xsq[:, 1:2]
        return g11_sq, g22_sq


def attach_torch_dynamics(const) -> TorchDynamicsAdapter:
    """
    Convenience: build an adapter AND attach its callables onto `const`
    so existing code that checks hasattr(const, 'f_of_x') just works.
    Returns the adapter (which you can also pass directly).
    """
    adapter = TorchDynamicsAdapter(const)

    # Bind methods onto `const` (thin proxy to adapter) without breaking pickling.
    def _f_of_x(x: torch.Tensor):
        return adapter.f_of_x(x)
    def _gamma_sq_of_x(x: torch.Tensor):
        return adapter.gamma_sq_of_x(x)

    setattr(const, "f_of_x", _f_of_x)
    setattr(const, "gamma_sq_of_x", _gamma_sq_of_x)
    return adapter
