# Install

* create a python virtual environment by: python3.12 -m venv .venv
* activate virtual environment: source .venv/bin/activate
* install required packages by: pip install -r requirements.txt 

# Acknowledgement
* We reproduce the result of the paper "Neural Continuous-Time Supermartingale Certificates, Grigory Neustroev, Mirco Giacobbe, and Anna Lukina". (when we say paper, we refer to this paper)

# Documentation
## System 1: Bivariate Geomatric Brownian Motion (GBM), 2D
* $dx = (\mu x + \pi(x)) dt + \sigma(x) dw$, $\mu = [-0.5, 1; -1, -0.5], \sigma = 0.2 \cdot diag(x)$
* policy $\pi(x) = -x$
* Note that there is an typo in the sign in the paper. We can verify by the fact that: If the policy is stabilizing, it should look like this.
* In the code we combine $\mu x - x = Ax$, where $A=[-1.5, 1; -1, -1.5]$
* Specifications: $X = [-100, 100]^2, X_0 = [45, 55] \times [-55, -45], X_{*}=[-25, 25]^2, X_{unsafe}=[-100, -80] \times [-100, 100]$
* Train Barrier: $\epsilon = \delta = 0.9$, $\alpha_{RA}=1$, $\beta_{RA}=\kappa \alpha_{RA}/(1-\epsilon) = \kappa/0.1$, and $\beta_S=0.9$. In the experiment $\kappa=4$ suc that $\kappa > 1, \beta_{RA} > \alpha_{RA}=1$. In brief the reach-avoid probability is $\epsilon$ (90%).
* Loss = $L_{unsafe} + L_{0} + L_{\star} + L_{G}$, where
* $L_{unsafe} = \beta_{RA} - V(t,x)$, for all $x \in X_{unsafe}$ and all $t$.
* $L_0 = V(t=0,x) - \alpha_{RA}$, for all $x \in X_0$
* $L_{\star} = \beta_S - V(t,x)$, for all $x \in X_{\star}$ and all $t$.
* $L_G = G[V(t,x)] + \zeta$, for all $x \in l^-_{\beta_{RA}}$ \ $int(X_{\star})$ and all $t$, in the paper. $\zeta=1$. Note that the first barrier is trained by setting $\zeta=0.1$ in order to make the differential constraints easier to satisfy.
* Note that we can drop the time variable if the system is time-homogeneous, thus making $V(t,x) = V(x)$ only.
* Note also that each loss formulated as: "we count how points violate the conditions $L_i ≤ 0$ and sum them up".
* Python 3.12.10
* NOTE: the 5th condition regarding $\beta_S$ seems inconsistent with the loss $L_{\star}$ in paper. First, a more rigorous math should say: an empty set $\{x, V(x) ≤ \beta_S\} \subset X_{*}$, which implies the loss $L_{\star}$ to be modified into two part: (1) for all x such that $V(x) ≤ \alpha_{RA}$ and $x$ is outside $X_{\star}$, $V(x) > \beta_S$, and (2) there exists x inside $X_{\star}$ such that $0 ≤ V(x) ≤ \beta_S$. See the loss implemented in main.py.

# How to run
* run: python test_all.py to verify dynamics (similar to Fig. 4 in the paper)
* run: python main.py to see the trained barrier.
  * you can look into the function check_barrier to see how to get the values of the barrier V(x) given any input x (needs to be a torch tensor of shape N by 2, N is any batch_size), as well as the value of the differential operator on V, i.e., G[V(x)].