import torch
import numpy as np
from src.constants import Const_GBM
from src.models import BaseNet, load_trained_model
import matplotlib.pyplot as plt
const = Const_GBM()


def diff_operator(const, net, x):
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
    GV = f1*V_x1 + f2*V_x2 + 0.5*(g11*g11*V_x1x1 + g22*g22*V_x2x2)
    return GV
    

def init_weights_xavier(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.0)


def train_barrier(const, net, iterations=20000):
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.95)
    batch_size = 500

    # --- deterministic grid ---
    _, _, grid_points_tensor = const.generate_grid(N=100)

    inside_init = filter_sample_insidebound(grid_points_tensor, const.X_INIT_RANGE)
    if inside_init.any():
        grid_points_init = grid_points_tensor[inside_init]

    inside_unsafe = filter_sample_insidebound(grid_points_tensor, const.X_UNSAFE_RANGE)
    if inside_unsafe.any():
        grid_points_unsafe = grid_points_tensor[inside_unsafe]

    inside_goal = filter_sample_insidebound(grid_points_tensor, const.X_GOAL_RANGE)
    if inside_goal.any():
        grid_points_goal = grid_points_tensor[inside_goal]             

    # --- training ---
    best = {"best_iter":None, "best_model": None, "best_loss": np.inf}
    for iter in range(iterations):
        loss = torch.tensor(0.0)
        optimizer.zero_grad()
        vio_dict = {"init": None,
                    "goal": None,
                    "unsafe": None,
                    "differential": None}

        # --- loss init ---
        x = const.sample_x(const.X_INIT_RANGE, batch_size)
        x = torch.cat((x, grid_points_init), dim=0)
        V = net(x)
        violate = (const.ALPHA_RA < V)
        violate_percent = 100.0*(violate.sum()/V.shape[0])
        vio_dict["init"] = violate_percent.item()
        loss_init = torch.relu(V - const.ALPHA_RA).mean()
        loss = loss + loss_init

        # --- loss unsafe ---
        x = const.sample_x(const.X_UNSAFE_RANGE, batch_size)
        x = torch.cat((x, grid_points_unsafe), dim=0)
        V = net(x)
        violate = (V < const.BETA_RA)
        violate_percent = 100.0*(violate.sum()/V.shape[0])
        vio_dict["unsafe"] = violate_percent.item()
        loss_unsafe =  torch.relu(const.BETA_RA - V).mean()
        loss = loss + loss_unsafe

        # --- loss goal ---
        # x = const.sample_x(const.X_GOAL_RANGE, batch_size)
        # V = net(x)
        # violate = (V < const.BETA_S)
        # violate_percent = 100.0*(violate.sum()/V.shape[0])
        # vio_dict["goal"] = violate_percent.item()
        # loss_goal = torch.relu(const.BETA_S - V).mean()
        # loss = loss + loss_goal
        """
        change the loss_goal via:
        the set Phi = {x such that V(x) <= 0.9} is 
        (1) Phi is non-empty and
        (2) Phi is a proper subset inside const.X_GOAL_RANGE
        """
        # --- loss goal ---
        # 1) sample a batch from the full domain
        x_full = const.sample_x(const.X_RANGE, batch_size)
        x_full = torch.cat((x_full, grid_points_tensor), dim=0)
        inside_goal = filter_sample_insidebound(x_full, const.X_GOAL_RANGE)
        outside_goal = ~inside_goal
        if outside_goal.any():
            x = x_full[outside_goal]
            V = net(x)
            violate = (V <= const.BETA_S)
            violate_percent = 100.0*(violate.sum()/V.shape[0])
            vio_dict["goal"] = violate_percent.item()
            loss_goal = torch.relu(const.BETA_S - V).mean()
            loss = loss + loss_goal
        if inside_goal.any():
            x = x_full[inside_goal]
            V = net(x)
            V_min = V.min()
            vio_dict["V(x_goal) min"] = V_min.item()
            # hinge‐loss to keep 0 ≤ V_min ≤ 0.9
            loss_lower = torch.relu(0.0   - V_min)   # penalizes V_min < 0
            loss_upper = torch.relu(V_min - 0.9)     # penalizes V_min > 0.9
            loss_goal = loss_lower + loss_upper
            loss = loss + loss_goal

        # --- loss Differential ---
        x = const.sample_x(const.X_RANGE, batch_size, requires_grad=True)
        x = torch.cat((x, grid_points_tensor), dim=0)
        inside_goal = filter_sample_insidebound(x, const.X_GOAL_RANGE)
        if inside_goal.any():
            x = x[~inside_goal]
        V = net(x)
        mask = (V.view(-1) <= const.BETA_RA)
        if(mask.any()):
            x = x[mask]
        else:
            raise("cannot find samples for differential constraint")
        GV = diff_operator(const, net, x)
        violate = (-const.ZETA < GV)
        violate_percent = 100.0*(violate.sum()/GV.shape[0])
        vio_dict["differential"] = violate_percent.item()
        vio_dict["GV max"] = GV.max()
        loss_differential = torch.relu(GV + const.ZETA).mean()
        loss = loss + loss_differential
        # if(violate_percent > 0.0):
        #     loss = loss + torch.mean(GV + const.ZETA)**2

        # --- loss Nonegative
        # x = const.sample_x(const.X_RANGE, batch_size)
        # x = torch.cat((x, grid_points_tensor), dim=0)
        # V = net(x)
        # violate = (V < 0.0)
        # violate_percent = 100.0*(violate.sum()/V.shape[0])
        # vio_dict["non-neg"] = violate_percent.item()
        # loss_nonneg = torch.relu(0.0 - V).mean()
        # loss = loss + loss_nonneg
        
        # --- optimize ---
        if(loss.item() < 0.95*best["best_loss"]):
            best["best_iter"] = iter
            best["best_loss"] = loss.item()
            best["best_model"] = net
            print("[Best] iter {:3d}, loss: {:.4f}".format(iter, loss.item()))
            for k, v in vio_dict.items():
                if v is None:
                    print(f".  {k}: None")
                else:
                    print(f".  {k}: {v:.4f}")
        if(loss.item() <= 0.0):
            print("[Training Complete] no violation found")
            return net
        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()
    return best["best_model"]


def filter_sample_insidebound(x, bound):
    bound   = torch.as_tensor(bound, dtype=x.dtype)
    lows, highs = bound[:,0], bound[:,1]                        # each (2,)
    inside = (x >= lows) & (x <= highs)                  # shape (M,2) bool
    inside = inside.all(dim=1)                       # shape (M,)  bool
    return inside


def visual_barrier(const, net):
    # --- build grid and evaluate V ---
    X, Y, grid_pts = const.generate_grid()               # X,Y: (M,M), grid_pts: (M*M,2)
    V = net(grid_pts).detach().cpu().numpy().reshape(X.shape)

    fig, ax = plt.subplots(figsize=(6,5))

    # 1) filled background contour for V
    cf = ax.contourf(
        X, Y, V,
        levels=50,                  # many intermediate levels
        cmap='viridis',
        alpha=0.6
    )
    cbar = fig.colorbar(cf, ax=ax, label=r'$V(x)$')
    # 2) explicit contour lines at 0.9, 1.0, and 40.0
    levels = [0.1, 0.9, 1.0, 10.0, 40.0]
    cs = ax.contour(
        X, Y, V,
        levels=levels,
        colors=['white','green','blue', 'yellow','red'],  # pick distinct colors
        linewidths=1.5,
    )
    ax.clabel(cs, fmt='%1.1f', fontsize=10)

    # 3) overlay your rectangles in 2D
    visual_specification(ax, const.X_INIT_RANGE,   color='blue')
    visual_specification(ax, const.X_UNSAFE_RANGE, color='red')
    visual_specification(ax, const.X_GOAL_RANGE,   color='green')

    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_aspect('equal', 'box')
    plt.tight_layout()
    plt.show()


def visual_barrier_3d(const, net):
    X, Y, grid_points_tensor = const.generate_grid()
    V = net(grid_points_tensor).detach().numpy().reshape(X.shape)
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    # 3D surface with colormap and transparency
    surf = ax.plot_surface(
        X, Y, V,
        rstride=1, cstride=1,
        cmap='viridis',       # choose any matplotlib colormap
        edgecolor='none',     # no grid lines
        alpha=0.6             # make it translucent
    )
    # add a colorbar
    cb = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    cb.set_label(r'$V(x)$')
    visual_specification(ax, const.X_INIT_RANGE, "black")
    visual_specification(ax, const.X_UNSAFE_RANGE, "red")
    visual_specification(ax, const.X_GOAL_RANGE, "green")
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.show()


def visual_specification(ax, bound, color, z_value=0.0):
    # --- overlay the init‐region rectangle ---
    # get [x_min, x_max], [y_min, y_max]
    x0, x1 = bound[0]
    y0, y1 = bound[1]
    # pick a constant Z (e.g. the floor of your surface)
    z0 = z_value
    # rectangle corners, closing the loop
    rect = np.array([
        [x0, y0],
        [x1, y0],
        [x1, y1],
        [x0, y1],
        [x0, y0],
    ], dtype=np.float32)
    # plot in 3D: xs, ys, and constant z0
    ax.plot(rect[:,0], rect[:,1], z0, color=color, linewidth=1.5, linestyle="--")


def check_barrier(const, net):
    batch_size = 100000
    x = const.sample_x(const.X_RANGE, batch_size)
    V = net(x).detach().numpy()
    print("[check non-negative] V(x) >= 0 for all x, Vmin={:.4f}".format(V.min()))

    x = const.sample_x(const.X_INIT_RANGE, batch_size)
    V = net(x).detach().numpy()
    print("[check init] V(x) <= 1.0 for all x, Vmax={:.4f}".format(V.max()))

    x = const.sample_x(const.X_UNSAFE_RANGE, batch_size)
    V = net(x).detach().numpy()
    print("[check unsafe] V(x) >= 10.0 for all x in X_unsafe, Vmin={:.4f}".format(V.min()))

    x = const.sample_x(const.X_GOAL_RANGE, batch_size)
    V = net(x).detach().numpy()
    print("[check goal] (1) exists V(x) <= 0.9, x in X_goal, Vmin={:.4f}".format(V.min()))
    x = const.sample_x(const.X_RANGE, batch_size)
    inside_goal = filter_sample_insidebound(x, const.X_GOAL_RANGE)
    outside_goal = ~inside_goal
    x = x[outside_goal]
    V = net(x).detach().numpy()
    print("[check goal] (2) V(x) >= 0.9 for all x outside X_goal, Vmin={:.4f}".format(V.min()))

    x = const.sample_x(const.X_RANGE, batch_size, requires_grad=True)
    inside_goal = filter_sample_insidebound(x, const.X_GOAL_RANGE)
    if inside_goal.any():
        x = x[~inside_goal]
    V = net(x)
    mask = (V.view(-1) <= const.BETA_RA)
    if(mask.any()):
        x = x[mask]
    GV = diff_operator(const, net, x).detach().numpy()
    print("[check differentail] V(x) < 0 for all x in the sub-beta_RA set and outside the target, GVmax={:.4f}".format(GV.max()))


def main():
    train_flag = False
    net_path = "output/net.pth"

    net = BaseNet(const, neurons=128)
    if(train_flag):
        torch.manual_seed(0)
        np.random.seed(0)
        net.apply(init_weights_xavier)
        # visual_barrier(const, net)
        net = train_barrier(const, net, iterations=50000)
        torch.save({
                    'model_state_dict': net.state_dict(),
                    # 'epoch': epoch, 
                    # 'loss_history': loss_history, 
                    # 'train_time': train_time,
                    }, net_path)
    else:
        net = load_trained_model(net, net_path)

    check_barrier(const, net)
    visual_barrier(const, net)
    visual_barrier_3d(const, net)


if __name__ == "__main__":
    main()    