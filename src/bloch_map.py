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
    ax1 = fig.add_subplot(111, projection="3d")
    #ax2 = fig.add_subplot(122, projection="3d")

    ax1.plot_surface(x, y, z, color="k", alpha=0.2)
    ax1.plot_surface(x_, y_, z_, color="r")
    
    axs = [ax1]
    for ax in axs:    
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
    
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        ax.set_zticks([-1, 0, 1])
        
        ax.set_aspect("equal", adjustable="box")        
    
    color = ax.patch.get_facecolor()
    fig.patch.set_facecolor(color)
    plt.show()
    
    
if __name__ == "__main__":
    
    T = lambda X: utils.depolarizing_channel(X, p=0.1)
    
    plot_channel(T)
    plot_channel(utils.example_map)
    