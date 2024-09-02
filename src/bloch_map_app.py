import channels as ch
from bloch_map import BlochMap
import dash
from dash import dcc, html

channel_functions = {
    'depolarizing': ch.depolarizing,
    'amplitude_damping': ch.amplitude_damping,
    'phase_damping': ch.phase_damping,
    'holevo_werner': ch.holevo_werner,
    'resonant_amplitude_damping': ch.resonant_amplitude_damping}

app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([dcc.Graph(id='bloch-channel')],
            style={'width': '80%', 'margin': 'auto', 'align': 'left', 'display': 'inline-block'}),
    html.Div([
    html.Br(),
    html.Label("Quantum channel"),
    html.Br(), html.Br(), html.Br(),
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
    html.Br(), html.Br(), html.Br(), html.Br(), html.Br(),
    html.Label("Probability [%]"),
    html.Br(), html.Br(),
    dcc.Slider(
        id='p-slider',
        min=0,
        max=100,
        step=0.1,
        value=50,
        marks={i: str(i) for i in range(0, 101, 10)},
        tooltip={"placement": "bottom", "always_visible": True}
        ),
    html.Br(), html.Br(),
    html.Div(id='p-output-container')
    ], style={'width': '50%', 'margin': 'auto', 'align': 'right', 'display': 'inline-block'})],
    style={'width': '100%', 'margin': 'auto', 'marginTop':'50px', 'textAlign': 'center','display': 'flex'})


bloch_instance = BlochMap()

@app.callback(
    dash.dependencies.Output('bloch-channel', 'figure'),
    [dash.dependencies.Input('channel-dropdown', 'value'),
        dash.dependencies.Input('p-slider', 'value')]
    )

def update_graph(selected_channel, p):
    channel_func = channel_functions[selected_channel]
    figure = bloch_instance.plot_channel(channel_func, p/100)

    # Maintain view state
    figure.update_layout(uirevision='constant',)

    return figure
if __name__ == "__main__":
    app.run_server(debug=True)