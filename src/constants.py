import numpy as np
import torch
from shapely.geometry import Polygon


class Const_GBM():
    """
    define constants for bivariate geometric brownian motion
    """
    _eps = np.float32(0.9)
    _zeta = np.float32(0.1)
    _alpha_ra = np.float32(1.0)
    _beta_s = np.float32(0.9)
    _kappa = np.float32(4.0) #4.0
    _beta_ra = _kappa/(1.0-_eps)
    _x_range = np.array([[-100.0, 100.0], 
                         [-100.0, 100.0]],
                         dtype=np.float32)
    _x_init_range = np.array([[45.0, 55.0],
                              [-55.0, -45.0]],
                         dtype=np.float32)
    _x_goal_range = np.array([[-25.0, 25.0],
                              [-25.0, 25.0]],
                         dtype=np.float32)
    _x_unsafe_range = np.array([[-100.0, -80.0],
                              [-100.0, 100.0]],
                         dtype=np.float32)
    _x_dim = _x_range.shape[0]

    _A_matrix = np.array([[-1.5, 1.0], 
                          [-1.0, -1.5]], dtype=np.float32)
    
    _sigma = np.float32(0.2)

    @property
    def EPS(self):
        return self._eps
    
    @property
    def ZETA(self):
        return self._zeta
    
    @property
    def ALPHA_RA(self):
        return self._alpha_ra
    
    @property
    def BETA_RA(self):
        return self._beta_ra
    
    @property
    def BETA_S(self):
        return self._beta_s
    
    @property
    def X_RANGE(self):
        return self._x_range
    
    @property
    def X_INIT_RANGE(self):
        return self._x_init_range
    
    @property
    def X_GOAL_RANGE(self):
        return self._x_goal_range
    
    @property
    def X_UNSAFE_RANGE(self):
        return self._x_unsafe_range
    
    @property
    def X_DIM(self):
        return self._x_dim
    
    @property
    def A_MATRIX(self):
        return self._A_matrix
    
    @property
    def SIGMA(self):
        return self._sigma
    

    def sample_x(self, bound, batch_size, requires_grad=False):
        low, high = bound.T
        x = np.random.uniform(low, high, size=(batch_size, low.shape[0])).astype(np.float32)
        x_tensor = torch.tensor(x, dtype=torch.float32, requires_grad=requires_grad)
        return x_tensor
    

    def generate_grid(self, N=50):
        lows, highs = self.X_RANGE.T
        # build each axis vector
        xs = np.linspace(lows[0], highs[0], N, dtype=self.X_RANGE.dtype)
        ys = np.linspace(lows[1], highs[1], N, dtype=self.X_RANGE.dtype)
        # make the 2D mesh
        X, Y = np.meshgrid(xs, ys, indexing='ij')
        grid_points = np.vstack([X.ravel(), Y.ravel()]).T
        grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32, requires_grad=False)
        return X, Y, grid_points_tensor
    

    def filter_sample_insidebound(self, x, bound):
        bound   = torch.as_tensor(bound, dtype=x.dtype)
        lows, highs = bound[:,0], bound[:,1]                        # each (2,)
        inside = (x >= lows) & (x <= highs)                  # shape (M,2) bool
        inside = inside.all(dim=1)                       # shape (M,)  bool
        return inside
    

    def get_f1_bound(self, bound):
        x1_bound = bound[0,:]
        x2_bound = bound[1,:]
        A00_bound = np.array([self.A_MATRIX[0,0], self.A_MATRIX[0,0]])
        A01_bound = np.array([self.A_MATRIX[0,1], self.A_MATRIX[0,1]])
        bounds = self.get_bounds_from_product_of_two_bounds(
            A00_bound, x1_bound
        ) + self.get_bounds_from_product_of_two_bounds(
            A01_bound, x2_bound
        )
        return bounds
    

    def get_f2_bound(self, bound):
        x1_bound = bound[0,:]
        x2_bound = bound[1,:]
        A10_bound = np.array([self.A_MATRIX[1,0], self.A_MATRIX[1,0]])
        A11_bound = np.array([self.A_MATRIX[1,1], self.A_MATRIX[1,1]])
        bounds = self.get_bounds_from_product_of_two_bounds(
            A10_bound, x1_bound
        ) + self.get_bounds_from_product_of_two_bounds(
            A11_bound, x2_bound
        )
        return bounds
    

    def get_g11square_bound(self, bound):
        # g11 = sigma*x1, sigma is a constant
        x1_bound = bound[0,:]
        x_lo = x1_bound[0]
        x_hi = x1_bound[1]
        if x_lo >= 0:                       # case 1
            f_lo, f_hi = x_lo**2, x_hi**2
        elif x_hi <= 0:                     # case 2
            f_lo, f_hi = x_hi**2, x_lo**2
        elif np.abs(x_hi) < np.abs(x_lo):
            f_hi = x_lo**2     
            f_lo = 0.0
        else:
            f_hi = x_hi**2     
            f_lo = 0.0
        return np.array([f_lo, f_hi])*self.SIGMA**2
    

    def get_g22square_bound(self, bound):
        # g22 = sigma*x2, sigma is a constant
        x2_bound = bound[1,:]
        x_lo = x2_bound[0]
        x_hi = x2_bound[1]
        if x_lo >= 0:                       # case 1
            f_lo, f_hi = x_lo**2, x_hi**2
        elif x_hi <= 0:                     # case 2
            f_lo, f_hi = x_hi**2, x_lo**2
        elif np.abs(x_hi) < np.abs(x_lo):
            f_hi = x_lo**2     
            f_lo = 0.0
        else:
            f_hi = x_hi**2     
            f_lo = 0.0
        return np.array([f_lo, f_hi])*self.SIGMA**2
        
    

    def get_bounds_from_product_of_two_bounds(self, b1, b2):
        points = np.array([b1[0]*b2[0], b1[0]*b2[1], b1[1]*b2[0], b1[1]*b2[1]])
        lb = points.min()
        ub = points.max()
        return np.array([lb, ub])
    