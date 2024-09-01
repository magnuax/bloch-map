import numpy as np
import dash
from dash import dcc, html
import plotly.graph_objs as go
import utils as utils

def plot_channel(p, qx, qy, qz, N=50):
    """
    Plots the action of a depolarizing & dephasing channel on the Bloch sphere.
    Args:
        N (int, optional): Number of points in meshgrid. Defaults to 50.
    """
    
    T0 = lambda X: utils.depolarizing_channel(X, p=p)
    T1 = lambda X: utils.dephasing_x_channel(T0(X), p=qx)
    T2 = lambda X: utils.dephasing_y_channel(T1(X), p=qy)
    T = lambda X: utils.dephasing_z_channel(T2(X), p=qz)
    
    

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

    trace_bloch_sphere = go.Surface(
        x=x,
        y=y,
        z=z,
        opacity=0.1,
        surfacecolor=np.zeros_like(x),
        colorscale='Greys',
        showscale=False,
    )

    trace_channel = go.Surface(
        x=x_,
        y=y_,
        z=z_,
        opacity=0.9,
        colorscale='Reds',
        showscale=False,
    )

    annotations = [
        dict(x=0, y=0, z=1.1, text=r"|↑⟩", showarrow=False),
        dict(x=0, y=0, z=-1.1, text=r"|↓⟩", showarrow=False),
        dict(x=1.1, y=0, z=0, text=r"|+⟩", showarrow=False),
        dict(x=-1.1, y=0, z=0, text=r"|-⟩", showarrow=False),
        dict(x=0, y=1.1, z=0, text=r"|→⟩", showarrow=False),
        dict(x=0, y=-1.1, z=0, text=r"|←⟩", showarrow=False),
    ]

    # Add dashed lines for axes
    axis_lines = []

    axis_lines.append(go.Scatter3d(
        x=[-1, 1], y=[0, 0], z=[0, 0],
        mode='lines',
        line=dict(color='black', width=2, dash='dash'),
        showlegend=False
    ))

    axis_lines.append(go.Scatter3d(
        x=[0, 0], y=[-1, 1], z=[0, 0],
        mode='lines',
        line=dict(color='black', width=2, dash='dash'),
        showlegend=False
    ))

    axis_lines.append(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[-1, 1],
        mode='lines',
        line=dict(color='black', width=2, dash='dash'),
        showlegend=False
    ))

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

    fig = go.Figure(data=[trace_bloch_sphere, trace_channel] + axis_lines, layout=layout)
    
    return fig

if __name__ == "__main__":
    # Dash app

    # make one slider for each dephasing direction
    sliders_dephase = []
    for direction in ['x', 'y', 'z']:
        sliders_dephase.append(
            html.Label(f'Dephasing parameter q_{direction}:'))
        sliders_dephase.append(
            dcc.Slider(
                id=f'q-{direction}-slider',
                min=0,
                max=1,
                step=0.01,
                value=0.1,
                marks={i/10: str(i/10) for i in range(0, 11)},
                tooltip={"placement": "bottom", "always_visible": True}))

    # and one slider for depolarization
    slider_depolarize = [
        html.Label('Depolarizing parameter p:'),
        dcc.Slider(
            id='p-slider',
            min=0,
            max=1,
            step=0.01,
            value=0.1,
            marks={i/10: str(i/10) for i in range(0, 11)},
            tooltip={"placement": "bottom", "always_visible": True})]
    
    # Set layout
    app = dash.Dash(__name__)
        
    app.layout = html.Div([
        dcc.Graph(id='bloch-channel'),
        *slider_depolarize,
        *sliders_dephase,
        html.Div(id='slider-output-container')
    ],
        style={'width': '50%', 'margin': 'auto', 'textAlign': 'center'},)

    @app.callback(
        dash.dependencies.Output('bloch-channel', 'figure'),
        [dash.dependencies.Input('p-slider', 'value'), 
         dash.dependencies.Input('q-x-slider', 'value'), 
         dash.dependencies.Input('q-y-slider', 'value'), 
         dash.dependencies.Input('q-z-slider', 'value')]
    )
    def update_graph(p,qx,qy,qz):
        return plot_channel(p,qx,qy,qz)

    app.run_server(debug=True)