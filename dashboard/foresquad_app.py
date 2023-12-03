import pandas as pd
import numpy as np
import base64
import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
import plotly.express as px
from matplotlib import pyplot as plt
import seaborn as sns
from io import BytesIO
from dash.dependencies import Input, Output
from IPython.display import Image
import plotly.graph_objs as go
from plotly.subplots import make_subplots
# from peaky_finders.d_utils import get_peak_data, get_forecasts
import constant as c
#from d_utils import create_load_duration
import layout as l
import dash_attempt_2 as d
from dash.exceptions import PreventUpdate

XGBresults = pd.read_csv('/Users/bariscavus/PycharmProjects/pythonProject/enerji/DI502_ElectricityPrediction/dashboard/XGB_results.csv')

#DTresults = pd.read_csv('/Users/bariscavus/PycharmProjects/pythonProject/enerji/DI502_ElectricityPrediction/dashboard/results/DT_results2.csv')
#LRresults = pd.read_csv('/Users/bariscavus/PycharmProjects/pythonProject/enerji/DI502_ElectricityPrediction/dashboard/results/LR_results.csv')

app = dash.Dash(
    external_stylesheets=[dbc.themes.LUX],
    suppress_callback_exceptions=True
)
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dcc.Tabs(id='tabs', value='/'),
    html.Div(id='page-content',style={'padding': '20px'}),
])



@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/':
        # Only display the content of the home page without the Home link
        return index_page
    else:
        # Wrap the content of each page with a div that contains a link to the Home tab
        return html.Div([
            dcc.Link('Home', href='/'),  # Add a link to the home page
            tabs_content.get(pathname, index_page),
        ])

# Update the callback for updating the tab from the URL
@app.callback(Output('tabs', 'value'),
              [Input('url', 'pathname')])
def update_tab_from_url(pathname):
    return pathname

# Update the callback for updating the URL from the tab
@app.callback(Output('url', 'pathname'),
              [Input('tabs', 'value')])
def update_url_from_tab(selected_tab):
    return selected_tab


index_page = html.Div([
    html.Div([
        html.Br(),
        html.Br(),
        dbc.Row([
            dbc.Col([
                html.H1(children="Welcome to Foresquad"),
                # html.Img(src="lightning.png", width="200px", style={'margin-top': '10px'}),
            ], width=5),
            dbc.Col(width=5),
        ], justify='center'),
        dbc.Row([
            dbc.Col(
                html.Div([
                    html.H4(
                        children="To what extent do weather, weekday, and primary use determine total electricity consumption? Click a tab below to find out."
                    ),
                    # html.Div(l.BUTTON_LAYOUT),
                ]), width=7
            ),
            dbc.Col(width=3),
        ], justify="center"),

        dcc.Tabs(id='tabs', children=[
            dcc.Tab(label='Home', value='/'),
            dcc.Tab(label='Dataset', value='/dataset'),
            dcc.Tab(label='Moving Average', value='/linear'),
            dcc.Tab(label='Decision Tree', value='/model'),
            dcc.Tab(label='XGBoost', value='/xgboost'),
            dcc.Tab(label='ARIMA', value='/arima'),
        ]),
    ]),
    ],
    style={
        'background-image': f'url({app.get_asset_url("elec.png")})',  # Fix the syntax error here
        'background-size': 'cover',  # Use 'cover' to ensure the image covers the entire background
        'background-position': 'center',
        'background-repeat': 'no-repeat',
        'height': '100vh',
        'display': 'flex',
        'flex-direction': 'column',
        'background-color': 'rgba(255, 255, 255, 0.3)',  # Adjust the alpha channel (0.5 for 50% transparency)
    }
)


dataset_layout = html.Div([
    html.H1('Dataset'),
    html.P(c.Dataset_DESCRIPTION),

    dcc.Dropdown(
    
        id='time-interval-dropdown',
        options=[
            {'label': 'Hourly', 'value': 'hourly'},
            {'label': 'Weekly', 'value': 'weekly'},
            {'label': 'Monthly', 'value': 'monthly'},
        ],
        value='hourly',  # Default selection
        style={'width': '33%', 'height': '%50'}
    ),
    dcc.Dropdown(
        id='site-dropdown',
        options=[{'label': f'Site {i}', 'value': i} for i in range(0, 15)],
        value=1,  # Default selection
        style={'width': '33%'}
    ),

    dcc.Dropdown(
        id='primary-use-dropdown',
        options=[{'label': primary_use, 'value': primary_use} for primary_use in d.merged_data['primary_use'].unique()],
        value='Education',  # Default selection
        style={'width': '33%', 'height': '%50'}
    ),

    

    dcc.Graph(id='model-performance-graph', style={'display': 'inline-block', 'width': '50%', 'height': '50%'}),
    dcc.Graph(id='site-distribution-plot', style={'display': 'inline-block', 'width': '50%', 'height': '50%'}),
    dcc.Graph(id='primary-use-distribution-plot', style={'width': '100%', 'height': '80vh'}),
],    
style={
    'background-color': 'lavender'
})


# Callback to update the site distribution plot based on the selected site
@app.callback(
    Output('site-distribution-plot', 'figure'),
    [Input('site-dropdown', 'value')]
)
def update_site_distribution_plot(selected_site):
    # Filter the data based on the selected site
    filtered_data = d.merged_data[d.merged_data['site_id'] == selected_site]

    # Check if the filtered data is empty
    if filtered_data.empty:
        raise PreventUpdate(f"No data for site ID: {selected_site}")

    # Check if the required column is present in the data
    if 'meter_reading_log' not in filtered_data.columns:
        raise PreventUpdate("Column 'meter_reading_log' not found in the filtered data")

    # Create the KDE plot
    fig = go.Figure()

    # Plot the KDE using Plotly
    fig.add_trace(go.Histogram(
        x=filtered_data['meter_reading_log'],
        histnorm='probability',
        name=f'site_id={selected_site}',
        marker_color='blue',
        opacity=0.7
    ))

    fig.update_layout(
        xaxis_title='Log-transformed Meter Reading',
        yaxis_title='Density',
        title=f'Distribution Plot - Site ID: {selected_site}',
        paper_bgcolor='lavender'  # Set the color of the background
    )

    return fig
# Callback to update the primary use distribution plot based on the selected primary use
@app.callback(
    Output('primary-use-distribution-plot', 'figure'),
    [Input('primary-use-dropdown', 'value')]
)
def update_primary_use_distribution_plot(selected_primary_use):
    # Filter the data based on the selected primary use
    filtered_data = d.merged_data[d.merged_data['primary_use'] == selected_primary_use]

    # Check if the filtered data is empty
    if filtered_data.empty:
        raise PreventUpdate(f"No data for primary use: {selected_primary_use}")

    # Check if the required column is present in the data
    if 'meter_reading_log' not in filtered_data.columns:
        raise PreventUpdate("Column 'meter_reading_log' not found in the filtered data")

    # Create the histogram plot
    fig = go.Figure()

    # Plot the KDE using Plotly
    fig.add_trace(go.Histogram(
        x=filtered_data['meter_reading_log'],
        histnorm='probability',
        name=f'primary_use={selected_primary_use}',
        marker_color='orange',
        opacity=0.7
    ))

    fig.update_layout(
        xaxis_title='Log-transformed Meter Reading',
        yaxis_title='Density',
        title=f'Distribution Plot - Primary Use: {selected_primary_use}',
        paper_bgcolor='lavender'  # Set the color of the background
    )

    return fig



# Callback to update the model performance graph based on the selected time interval
@app.callback(
    Output('model-performance-graph', 'figure'),
    [Input('time-interval-dropdown', 'value')]
)
def update_model_performance(selected_interval):
    # Choose the appropriate graph based on the selected interval
    if selected_interval == 'hourly':
        return d.fig_hour_mean
    elif selected_interval == 'weekly':
        return d.fig8
    elif selected_interval == 'monthly':
        return d.month_mean
    

# Add the layout to the app callback
xgboost_layout = html.Div([
    html.H1('XGBoost Predictions'),
    html.P(c.XGBoost_MODEL_DESCRIPTION),
    
    # Metric Selection Dropdown
    dcc.Dropdown(
        id='metric-dropdown',
        options=[
            {'label': 'MAE', 'value': 'mae'},
            {'label': 'MAPE', 'value': 'mape'},
            {'label': 'R2', 'value': 'r2'},
            {'label': 'MSE', 'value': 'mse'},
        ],
        value='mse',
        multi=False,
        style={'width': '50%'}
    ),

    # Create a single graph for both train and test data
    dcc.Graph(id='metric-plot'),    
],
style= {
    'background-color': 'lavender'
})

dt_layout = html.Div([
    html.H1('Decision Tree Predictions'),
    html.P(c.DecisionTree_MODEL_DESCRIPTION),      
    
],
style= {
    'background-color': 'lavender'
})



'''# Metric Selection Dropdown
    dcc.Dropdown(
        id='metric-dropdown3',
        options=[
            {'label': 'MAE', 'value': 'mae'},
            {'label': 'MAPE', 'value': 'mape'},
            {'label': 'R2', 'value': 'r2'},
            {'label': 'MSE', 'value': 'mse'},
        ],
        value='mse',
        multi=False,
        style={'width': '50%'}
    ),

    # Create a single graph for both train and test data
    dcc.Graph(id='metric-plot3'), '''

moving_layout = html.Div([
    html.H1('Moving Average Predictions'),
    html.P(c.MA_MODEL_DESCRIPTION),   
],
style= {
    'background-color': 'lavender'
})

'''# Metric Selection Dropdown
    dcc.Dropdown(
        id='metric-dropdown2',
        options=[
            {'label': 'MAE', 'value': 'mae'},
            {'label': 'MAPE', 'value': 'mape'},
            {'label': 'R2', 'value': 'r2'},
            {'label': 'MSE', 'value': 'mse'},
        ],
        value='mse',
        multi=False,
        style={'width': '50%'}
    ),

    # Create a single graph for both train and test data
    dcc.Graph(id='metric-plot2'),    '''

arima_layout = html.Div([
    html.H1('ARIMA Predictions'),
    html.P(c.ARIMA_DESCRIPTION),      
    
],
style= {
    'background-color': 'lavender'
})

tabs_content = {
    '/': index_page,
    '/dataset': dataset_layout,
    '/arima': moving_layout,
    '/xgboost': xgboost_layout,
    '/decisiontree': dt_layout,
    '/arima': arima_layout,

}

# Callback to update the graph based on the selected metric
@app.callback(
    Output('metric-plot', 'figure'),
    [Input('metric-dropdown', 'value')]
)
def update_metric_plot(selected_metric):
    # Choose the appropriate graph based on the selected metric
    title = f'Training and Test {selected_metric.upper()}'

    train_column = f'train_{selected_metric}'
    test_column = f'test_{selected_metric}'

    if train_column not in XGBresults.columns or test_column not in XGBresults.columns:
        raise PreventUpdate

    y_train = XGBresults[train_column]
    y_test = XGBresults[test_column]

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=XGBresults['fold'], y=y_train, mode='lines+markers', name='Train'))
    fig.add_trace(go.Scatter(x=XGBresults['fold'], y=y_test, mode='lines+markers', name='Test'))

    fig.update_layout(title=title, xaxis_title='Fold', yaxis_title=selected_metric.upper())

    return fig

'''# Callback to update the graph based on the selected metric
@app.callback(
    Output('metric-plot2', 'figure'),
    [Input('metric-dropdown2', 'value')]
)
def update_metric_plot(selected_metric):
    # Choose the appropriate graph based on the selected metric
    title = f'Training and Test {selected_metric.upper()}'

    train_column = f'train_{selected_metric}'
    test_column = f'test_{selected_metric}'

    if train_column not in DTresults.columns or test_column not in XGBresults.columns:
        raise PreventUpdate

    y_train = DTresults[train_column]
    y_test = DTresults[test_column]

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=DTresults['fold'], y=y_train, mode='lines+markers', name='Train'))
    fig.add_trace(go.Scatter(x=DTresults['fold'], y=y_test, mode='lines+markers', name='Test'))

    fig.update_layout(title=title, xaxis_title='Fold', yaxis_title=selected_metric.upper())

    return fig'''

# Define a function to create content and graph for a given model
def generate_model_tab_content(model_name, selected_dropdown_tab, data):
    content = html.Div([
        html.H3(f"{model_name} {selected_dropdown_tab} Content"),
        # Add more content as needed
    ])

    # Replace this with your actual graph generation code
    graph = px.scatter(data, x="x", y="y", title=f"{model_name} Graph for {selected_dropdown_tab}")

    return content, graph


if __name__ == '__main__':
    app.run_server(debug=True,mode = 'external',host = "0.0.0.0", port=8050)
