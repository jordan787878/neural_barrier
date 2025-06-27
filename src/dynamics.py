import numpy as np
from .constants import Const_GBM
const = Const_GBM()


def dyn_f(x):
    """
    f(x) of SDE
    x: (N_batch, x_dim)
    return: (N_batch, x_dim), A*x_row, for each x_row in x
    """
    x = np.asarray(x, dtype=np.float32)
    return x.dot(const.A_MATRIX.T)


def dyn_g(x):
    """
    g(x) of SDE
    x: (N_batch, x_dim)
    return: (N_batch, x_dim, x_dim), where each (i, :, :) is a (x_dim by x_dim) matrix
            formulated by 0.2 * diag(x)
    """
    x = np.asarray(x, dtype=np.float32)        # ensure float32
    I = np.eye(const.X_DIM, dtype=np.float32)  # (2, 2) identity
    # x[..., :, None] has shape (N_batch, 2, 1)
    # I[None, :, :] has shape (1, 2, 2)
    # broadcasting → (N_batch, 2, 2)
    return const.SIGMA * x[:, :, None] * I[None, :, :]





