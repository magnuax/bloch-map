import numpy as np
import dash
from dash import dcc, html
import plotly.graph_objs as go
import utils as utils

def plot_channel(channel_func, params, N=50):
    """
    Plots the action of a depolarizing & dephasing channel on the Bloch sphere.
    Args:
        N (int, optional): Number of points in meshgrid. Defaults to 50.
    """
    if hasattr(params, '__iter__'):
        T = lambda X: channel_func(X, *params)
    else:
        T = lambda X: channel_func(X, params)

    x,y,z = utils.get_sphere_mesh(N)
    
    Tx = np.zeros((N, N))
    Ty = np.zeros((N, N))
    Tz = np.zeros((N, N))
    
    for i in range(N):
        for j in range(N):
            X_new = utils.bloch_to_state(np.array([x[i,j], y[i,j], z[i,j]]))
            Tx[i, j], Ty[i, j], Tz[i, j] = utils.state_to_bloch(T(X_new))

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
        x=Tx,
        y=Ty,
        z=Tz,
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
    
    # Latitude and Longitude lines
    grid_lines = []
    num_lines = 48
    phi_lines = np.linspace(0, np.pi, num_lines)
    theta_lines = np.linspace(0, 2*np.pi, num_lines)
    
    for phi in phi_lines[::8]:
        Txphi = np.sin(phi) * np.cos(theta_lines)
        Typhi = np.sin(phi) * np.sin(theta_lines)
        Tzphi = np.cos(phi) * np.ones_like(theta_lines)
        grid_lines.append(go.Scatter3d(
            x=Txphi, y=Typhi, z=Tzphi, 
            mode='lines', 
            line=dict(color='rgba(0, 0, 0, 0.05)', width=1),
            showlegend=False
        ))

    for theta in theta_lines[::8]:
        Txtheta = np.sin(phi_lines) * np.cos(theta)
        Tytheta = np.sin(phi_lines) * np.sin(theta)
        Tztheta = np.cos(phi_lines)
        grid_lines.append(go.Scatter3d(
            x=Txtheta, y=Tytheta, z=Tztheta, 
            mode='lines', 
            line=dict(color='rgba(0, 0, 0, 0.05)', width=1),
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

    fig = go.Figure(data=[trace_bloch_sphere, trace_channel] + axis_lines + grid_lines, layout=layout)
    
    return fig

channel_functions = {
    'depolarizing': utils.depolarizing_channel,
    'amplitude_damping': utils.amplitude_damping_channel,
    'phase_damping': utils.phase_damping_channel,
    'holevo_werner': utils.holevo_werner_channel,
    'resonant_amplitude_damping': utils.resonant_amplitude_damping_channel}


# Set layout
app = dash.Dash(__name__)
    
app.layout = html.Div([
    html.Div([dcc.Graph(id='bloch-channel')],
             style={'width': '80%', 'margin': 'auto', 'align': 'left', 'display': 'inline-block'}),
    html.Div([
    html.Br(),
    html.Label("Quantum channel"),
    html.Br(),
    html.Br(),
    html.Br(),
    dcc.Dropdown(
        id='channel-dropdown',
        options=[{'label':"Depolarizing",'value':'depolarizing'},
                {'label':"Amplitude Damping",'value':'amplitude_damping'},
                {'label':"Phase Damping",'value':'phase_damping'},
                 {'label':"Holevo-Werner",'value':'holevo_werner'},
                 {'label':"Resonant Amplitude Damping",'value':'resonant_amplitude_damping'}],
        value='depolarizing',
        clearable=False
        ),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Label("Probability [%]"),
    html.Br(),
    html.Br(),
    dcc.Slider(
        id='p-slider',
        min=0,
        max=100,
        step=0.1,
        value=50,
        marks={i: str(i) for i in range(0, 101, 10)},
        tooltip={"placement": "bottom", "always_visible": True}
    ),
    html.Br(),
    html.Br(),
    html.Div(id='p-output-container')
    ], style={'width': '50%', 'margin': 'auto', 'align': 'right', 'display': 'inline-block'})],
    style={'width': '100%', 'margin': 'auto', 'marginTop':'50px', 'textAlign': 'center','display': 'flex'})

@app.callback(
    dash.dependencies.Output('bloch-channel', 'figure'),
    [dash.dependencies.Input('channel-dropdown', 'value'),
        dash.dependencies.Input('p-slider', 'value')]
)
def update_graph(selected_channel, p):
    channel_func = channel_functions[selected_channel]
    return plot_channel(channel_func, p/100)

if __name__ == "__main__":
    app.run_server(debug=True)