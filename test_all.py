import src.dynamics as dynamics
import numpy as np
import matplotlib.pyplot as plt
from src.constants import Const_GBM
const = Const_GBM()


def test_dynamics(show_plot=False):
    """
    test if x_eq goes to (0, 0)
    """
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
        dx = dynamics.dyn_f(x)*dt + np.matmul(dynamics.dyn_g(x), dw[..., None])[..., 0]
        # update state
        x = x + dx
        x1_data = np.concatenate([x1_data, x[:,0].reshape(-1,1)], axis=1)
        x2_data = np.concatenate([x2_data, x[:,1].reshape(-1,1)], axis=1)
    np.testing.assert_allclose(x, 0.0, rtol=1e-3, atol=1e-4, 
                               err_msg="test_dynamics does not approach zeros")
    if(show_plot):
        for i in range(batch_size):
            plt.plot(x1_data[i,:], x2_data[i,:])
        plt.xlim(const.X_RANGE[0,:])
        plt.ylim(const.X_RANGE[1,:])
        plt.show()
    return x1_data, x2_data


def main():
    test_dynamics(show_plot=True)


if __name__ == "__main__":
    main() 