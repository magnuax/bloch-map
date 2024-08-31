import numpy as np
import matplotlib.pyplot as plt
import utils as utils
plt.style.use("ggplot")
    
def plot_channel(T, N=50):
    """
    Plots the action of a quantum channel on the Bloch sphere.
    Args:
        T (function): Function representing a qubit channel. Maps 2x2 matrices to 2x2 matrices. 
        N (int, optional): Number of points in meshgrid. Defaults to 50.
    """
    
    theta = np.linspace(0, 2*np.pi, N)
    phi   = np.linspace(0, np.pi, N)    
    theta, phi = np.meshgrid(theta, phi)

    x = np.sin(phi)*np.cos(theta)
    y = np.sin(phi)*np.sin(theta)
    z = np.cos(phi)
    
    x_ = np.zeros((N, N))
    y_ = np.zeros((N, N))
    z_ = np.zeros((N, N))
    
    for i in range(N):
        for j in range(N):
        
            X_ = utils.bloch_to_state(np.array([x[i,j], y[i,j], z[i,j]]))

            x_[i, j], y_[i, j], z_[i, j] = utils.state_to_bloch(T(X_))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(x, y, z, color="k", alpha=0.1, edgecolor="k", cstride=2, rstride=2)
    ax.plot_surface(x_, y_, z_, color="r")
    
    ax.text(0, 0, 1.2, r"$|\uparrow\rangle$", fontsize=15, horizontalalignment="center")
    ax.text(0, 0, -1.3, r"$|\downarrow\rangle$", fontsize=15, horizontalalignment="center")

    ax.text(1.2, 0, 0, r"$|+\rangle$", fontsize=15, horizontalalignment="center")
    ax.text(-1.3, 0, 0, r"$|-\rangle$", fontsize=15, horizontalalignment="center")
    
    ax.text(0, 1.2, 0, r"$|\rightarrow\rangle$", fontsize=15, horizontalalignment="center")
    ax.text(0, -1.3, 0, r"$|\leftarrow\rangle$", fontsize=15, horizontalalignment="center")

    ax.plot([0, 0], [0, 0], [-1, 1], "k--", marker="o", zorder=0, markersize=5, alpha=0.5)
    ax.plot([0, 0], [-1, 1], [0, 0], "k--", marker="o", zorder=0, markersize=5, alpha=0.5)
    ax.plot([-1, 1], [0, 0], [0, 0], "k--", marker="o", zorder=0, markersize=5, alpha=0.5)
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    ax.set_xlim([-1.25, 1.25])
    ax.set_ylim([-1.25, 1.25])
    ax.set_zlim([-1.25, 1.25])

    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_zticks([-1, 0, 1])
    
    ax.set_aspect("equal", adjustable="box")        

    color = ax.patch.get_facecolor()
    fig.patch.set_facecolor(color)
    
    return fig, ax
    
    
if __name__ == "__main__":
        
    T = lambda X: utils.depolarizing_channel(X, p=0.1)
    fig, ax = plot_channel(T)
    plt.show()

    fig, ax = plot_channel(utils.example_map)
    plt.show()