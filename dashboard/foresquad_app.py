import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px

# from peaky_finders.d_utils import get_peak_data, get_forecasts
import constant as c
from get_map import get_iso_map
from d_utils import create_load_duration
import map as l

Year = '2016'

app = dash.Dash(
    external_stylesheets=[dbc.themes.LUX],
    suppress_callback_exceptions=True
)
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
    ])


index_page = html.Div([
        html.Br(),
        html.Br(),
        dbc.Row([
            dbc.Col(html.H1(children="Welcome to Peaky Finders"), width=5),
            dbc.Col(width=5),
        ], justify='center'),
        html.Br(),
        html.Br(),
        dbc.Row([
            dbc.Col(
                html.Div([
                    html.H4(
                        children=(
                            "To what extent do weather and weekday determine",
                            "total electricity demand on the grid? Click an"
                            "ISO button below to find out.")),
                    html.Div(l.BUTTON_LAYOUT)]), width=7
            ),
            dbc.Col(width=3),
        ], justify="center"),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        dbc.Row([
            dbc.Col(html.H4(children="ISO Territory Map"), width=4),
            dbc.Col(width=4)
        ], justify='center'),

])


"""Linear Regression LAYOUT"""
nyiso_layout = l.set_iso_layout(
    model='Linear Regression',
    full_name=c.NYISO_FULL_NAME,
    description=c.NYISO_DESCRIPTION,
    year=Year,
    mae=c.LR_MAE,
    model_description=c.LR_MODEL_DESCRIPTION,
    peak_data=peak_data,
    load_duration_curves=load_duration_curves,
)
@app.callback(dash.dependencies.Output('nyiso-content', 'children'),
              [dash.dependencies.Input('nyiso-button', 'value')])


@app.callback(dash.dependencies.Output('nyiso-graph', 'figure'),
             [dash.dependencies.Input('nyiso-dropdown', 'value')])
def plot_nyiso_load_(value):
    return l.plot_load_curve(
        value, model='Linear Regression', load= load, predictions=predictions
    )

@app.callback(dash.dependencies.Output("LR-scatter", "figure"), 
    [dash.dependencies.Input("LR-scatter-dropdown", "value")])
def nyiso_scatter_plot(value):
    return l.plot_scatter(value, model='Linear Reg', peak_data=peak_data)

