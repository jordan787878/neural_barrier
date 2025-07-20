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


def propagate_traj():
    low, high = const.X_INIT_RANGE.T
    batch_size = 3
    x = np.random.uniform(low, high, size=(batch_size, low.shape[0])).astype(np.float32)
    sim_time = 10.0
    dt = 1e-3
    x1_data = x[:,0].reshape(-1,1)
    x2_data = x[:,1].reshape(-1,1)
    for k in range(int(sim_time/dt)):
        # standard brownian noise
        dw = np.random.randn(batch_size, const.X_DIM).astype(np.float32)
        dw = dw*np.sqrt(dt)
        # sde
        dx = dyn_f(x)*dt + np.matmul(dyn_g(x), dw[..., None])[..., 0]
        # update state
        x = x + dx
        x1_data = np.concatenate([x1_data, x[:,0].reshape(-1,1)], axis=1)
        x2_data = np.concatenate([x2_data, x[:,1].reshape(-1,1)], axis=1)
    np.testing.assert_allclose(x, 0.0, rtol=1e-3, atol=1e-4, 
                               err_msg="test_dynamics does not approach zeros")
    return x1_data, x2_data





