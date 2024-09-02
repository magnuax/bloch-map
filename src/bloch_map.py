import numpy as np
import plotly.graph_objs as go
import utils as utils

class BlochMap:
    def __init__(self):
        return
    
    def set_channel(self, channel_func, params):
        if hasattr(params, '__iter__'):
            self.T = lambda X: channel_func(X, *params)
        else:
            self.T = lambda X: channel_func(X, params)
    
    def get_sphere_mesh(self, N):
        """
        Generates NxN mesh of points on the unit sphere.
        """
        theta = np.linspace(0, 2*np.pi, N)
        phi   = np.linspace(0, np.pi, N)    
        theta, phi = np.meshgrid(theta, phi)

        x = np.sin(phi)*np.cos(theta)
        y = np.sin(phi)*np.sin(theta)
        z = np.cos(phi)
        
        return x,y,z
    
    def map_bloch_sphere(self, x, y, z):
        """
        Maps pure states on Bloch sphere according to self.T
        """
        Tx = np.zeros_like(x)
        Ty = np.zeros_like(y)
        Tz = np.zeros_like(z)

        N,M = x.shape
    
        for i in range(N):
            for j in range(M):
                X_new = utils.bloch_to_state(np.array([x[i,j], y[i,j], z[i,j]]))
                Tx[i, j], Ty[i, j], Tz[i, j] = utils.state_to_bloch(self.T(X_new))
        
        return Tx, Ty, Tz
    
    def add_annotations(self):
        annotations = [
            dict(x=0, y=0, z=1.1, text=r"|↑⟩", showarrow=False),
            dict(x=0, y=0, z=-1.1, text=r"|↓⟩", showarrow=False),
            dict(x=1.1, y=0, z=0, text=r"|+⟩", showarrow=False),
            dict(x=-1.1, y=0, z=0, text=r"|-⟩", showarrow=False),
            dict(x=0, y=1.1, z=0, text=r"|→⟩", showarrow=False),
            dict(x=0, y=-1.1, z=0, text=r"|←⟩", showarrow=False)]
        
        return annotations
    
    def add_axis_lines(self):
        axis_lines = []
        
        coords = np.array([[-1,1], [0,0], [0,0]])
        for i in range(3):
            axis_lines.append(go.Scatter3d(
                x=coords[0], y=coords[1], z=coords[2],
                mode='lines',
                line=dict(color='black', width=2, dash='dash'),
                showlegend=False))

            coords = np.roll(coords, 1, axis=0)
            
        return axis_lines
    
    def add_grid_lines(self):
        grid_lines = []
        
        num_lines = 48
        phi_lines = np.linspace(0, np.pi, num_lines)
        theta_lines = np.linspace(0, 2*np.pi, num_lines)
        
        for phi in phi_lines[::8]:
            x_phi = np.sin(phi) * np.cos(theta_lines)
            y_phi = np.sin(phi) * np.sin(theta_lines)
            z_phi = np.cos(phi) * np.ones_like(theta_lines)
            grid_lines.append(go.Scatter3d(
                x=x_phi, y=y_phi, z=z_phi, 
                mode='lines', 
                line=dict(color='rgba(0, 0, 0, 0.05)', width=1),
                showlegend=False))

        for theta in theta_lines[::8]:
            x_theta = np.sin(phi_lines) * np.cos(theta)
            y_theta = np.sin(phi_lines) * np.sin(theta)
            z_theta = np.cos(phi_lines)
            grid_lines.append(go.Scatter3d(
                x=x_theta, y=y_theta, z=z_theta, 
                mode='lines', 
                line=dict(color='rgba(0, 0, 0, 0.05)', width=1),
                showlegend=False))
            
        return grid_lines
    
    def plot_channel(self, channel_func, params, N=50):
        self.set_channel(channel_func, params)
        x, y, z = self.get_sphere_mesh(N)
        Tx, Ty, Tz = self.map_bloch_sphere(x, y, z)
        
        """Plot surfaces"""
        trace_bloch_sphere = go.Surface(
            x=x, y=y, z=z,
            opacity=0.1,
            surfacecolor=np.zeros_like(x),
            colorscale='Greys',
            showscale=False)
        
        trace_channel = go.Surface(
            x=Tx, y=Ty, z=Tz,
            opacity=0.9,
            colorscale='Reds',
            showscale=False)
        
        annotations = self.add_annotations()
        axis_lines = self.add_axis_lines()
        grid_lines = self.add_grid_lines()
    
        layout = go.Layout(
        scene=dict(
            xaxis=dict(range=[-1.25, 1.25], tickvals=[-1, 0, 1], title="X"),
            yaxis=dict(range=[-1.25, 1.25], tickvals=[-1, 0, 1], title="Y"),
            zaxis=dict(range=[-1.25, 1.25], tickvals=[-1, 0, 1], title="Z"),
            aspectratio=dict(x=1, y=1, z=1),
            annotations=annotations,
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        paper_bgcolor='rgba(255,255,255,0)',
        plot_bgcolor='rgba(255,255,255,0)',
        )

        fig = go.Figure(data=[trace_bloch_sphere, trace_channel] + axis_lines + grid_lines, layout=layout)
        return fig
    