import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go


TEMPLATE = 'ggplot2'

BUTTON_LAYOUT = [
    dcc.Link(
        html.Button('HOME', id='home-button', className="mr-1"),
        href='/'),
    dcc.Link(
        html.Button('Dataset', id='dataset-button', className="mr-1"),
        href='/dataset'),
    dcc.Link(
        html.Button('Linear Regression Predictions', id='model-button', className="mr-1"),
        href='/model'),
    dcc.Link(
        html.Button('Decision Tree Predictions', id='model-button', className="mr-1"),
        href='/model'),
    dcc.Link(
        html.Button('XGBoost Predictions', id='model-button', className="mr-1"),
        href='/model'),
    dcc.Link(
        html.Button('ARIMA Predictions', id='model-button', className="mr-1"),
        href='/model'),

]

def set_iso_layout(
    model: str,
    year: str,
    mae: float,
    model_description: str,
    peak_data: dict,
    load_duration_curves: dict
):
    layout = html.Div([
        html.Div(id=f'{model}-content'),
        html.Br(),
        dbc.Row([
            dbc.Col(
                html.Div(BUTTON_LAYOUT), width=4),
            dbc.Col(width=7),
        ], justify='center'),
        html.Br(),
        html.Br(),
        dbc.Row([
            dbc.Col(
                html.H3('Model Performance'), width=9
            ),
            dbc.Col(width=2),
        ], justify='center'),
        dbc.Row([
            dbc.Col(
                html.Div(
                    children=f'Mean Absolute Error (MAE) for {year} : {mae}'
                ), width=9
            ),
            dbc.Col(width=2),
        ], justify='center'),
        html.Br(),
        dbc.Row([
            dbc.Col(
                    dcc.Dropdown(
                        id=f'{model}-dropdown',
                        options=[
                            {'label': 'Actual', 'value': 'Actual'},
                            {'label': 'Predicted', 'value': 'Predicted'}
                        ],
                        value=['Actual', 'Predicted'],
                        multi=True,
                    ), width=6
            ),
            dbc.Col(width=5),
        ], justify='center'),
        dcc.Graph(id=f'{model}-graph'),
        html.Br(),
        html.Br(),
        dbc.Row([
            dbc.Col(html.H3('Training Data'), width=9),
            dbc.Col(width=2)
        ], justify='center'),
        dbc.Row([
            dbc.Col(
                    html.Div(children=model_description), width=9
            ),
            dbc.Col(width=2)
        ], justify='center'),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    html.Div([
                        dcc.Graph(
                            figure=plot_histogram(model=model, peak_data=peak_data)
                        ),
                    ]), width=4),
                dbc.Col(
                    html.Div([
                        dcc.Graph(
                            figure=plot_load_duration(
                                model=model,
                                load_duration_curves=load_duration_curves
                            ),
                        ),]), width=4),
                dbc.Col(
                    html.Div([
                        dcc.Dropdown(
                            id=f'{model}-scatter-dropdown',
                            options=[
                                {'label': 'Day of Week', 'value': 'weekday'},
                                {'label': 'Season', 'value': 'season'}
                                ],
                            value='season',
                            multi=False,
                        ),
                        dcc.Graph(id=f'{model}-scatter')
                    ]
                ), width=4),
            ]
        ),
    ])
    return layout

def plot_load_curve(value, model: str, load: dict, predictions: dict):
    model = model.upper()
    fig = go.Figure()
    if 'Actual' in value:
        fig.add_trace(go.Scatter(
            x=load[model].index,
            y=load[model].values,
            name='Actual Load',
            line=dict(color='maroon', width=3)))
    if 'Predicted' in value:
        fig.add_trace(go.Scatter(
            x=predictions[model].index,
            y=predictions[model].values,
            name = 'Forecasted Load',
            line=dict(color='darkturquoise', width=3, dash='dash')))
    return fig.update_layout(
        title="System Load: Actual vs. Predicted",
        xaxis_title="Date",
        yaxis_title="Load (MW)",
        template=TEMPLATE
    )


'''def plot_histogram(model: str, peak_data: dict):
    model = model.upper()
    return px.histogram(
        peak_data[model],
        x=peak_data[model]['load_MW'],
        nbins=75,
        marginal="rug",
        title=f"Distribution of {model} Daily Peaks",
        color_discrete_sequence=['darkturquoise'] 
    ).update_layout(
        template=TEMPLATE,
        xaxis_title='Peak Load (MW)',
        yaxis_title='Number of Days'
    )
'''

'''def plot_scatter(value, model: str, peak_data: dict):
    fig = px.scatter(
        peak_data[model.upper()].dropna(),
        x="load_MW",
        y="temperature", 
        color=value
    )
    return fig.update_layout(
        template=TEMPLATE, title='Peak Load vs. Temperature'
    )'''

'''def plot_load_duration(iso: str, load_duration_curves: dict):
    return go.Figure().add_trace(
        go.Scatter(
            x=load_duration_curves[iso.upper()].reset_index().index,
            y=load_duration_curves[iso.upper()].values,
            mode = 'lines',
            fill='tozeroy',
            line=dict(color='maroon', width=3)
        )).update_layout(
            title="Peak Load Sorted by Day (Highest to Lowest)",
            xaxis_title="Number of Days",
            yaxis_title="Load (MW)",
            template=TEMPLATE)'''