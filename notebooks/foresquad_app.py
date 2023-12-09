import pandas as pd
import numpy as np
import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
import plotly.express as px
from dash.dependencies import Input, Output
from IPython.display import Image
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import dash_attempt_2 as d
import constant as c
import utils
from dash.exceptions import PreventUpdate
import os
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import xgboost as xgb

XGBresults = pd.read_csv('results/XGB_results.csv')
# Load the saved model
XGB = utils.load_model("XGB_v2")

df = utils.load_data()
df = utils.log_transformation(df)
df = utils.break_datetime(df)
df = df[df['site_id'].isin([1, 6])]
df = utils.nan_weather_filler(df)
# encoding 
df = utils.circular_encode(df,'month', 12)
df = utils.label_encode(df,'primary_use')
# Add holidays
df = utils.apply_holidays(df)


features = ['hour','air_temperature','month_sin','month_cos','log_square_feet', 'primary_use_encoded','is_holiday']
target = 'log_meter_reading'

# Assuming df is loaded and preprocessed
tscv = TimeSeriesSplit(n_splits=5)  # You can adjust the number of splits
fold_indices = []  # List to store train and test indices for each fold

i = 0  # Start fold index from 1

for train_index, test_index in tscv.split(df):
    df_train, df_test = df.iloc[train_index], df.iloc[test_index]
    
    # Store indices for each fold
    fold_indices.append((train_index, test_index))
    
    print('fold :', i+1)
    print('Train: ', df_train['timestamp'].min(), df_train['timestamp'].max()) 
    print('Test: ', df_test['timestamp'].min(), df_test['timestamp'].max()) 

    i += 1



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


# Construct the path to the background image
background_image_path = app.get_asset_url("elec.png")

index_page = html.Div([
    html.Div([
        dbc.Row([
            dbc.Col([
                html.H1(children="Welcome to Foresquad", style={'margin-bottom': '0'}),
            ], width=5),
            dbc.Col(width=5),
        ], justify='center'),
    ],
    style={
        'background-image': f'url({background_image_path})',
        'background-size': 'cover',
        'background-position': 'center',
        'background-repeat': 'no-repeat',
        'height': '100vh',  # Adjust the height to cover the entire viewport
        'display': 'flex',
        'flex-direction': 'column',
        'align-items': 'left',
        'justify-content': 'center',
        'margin': '0',  # Reset margin to zero
        'padding': '0',  # Reset padding to zero
    }),
    
    html.Div([
        html.Br(),
        dbc.Row([
            dbc.Col(
                html.Div([
                    html.H4(
                        children="To what extent do weather, weekday, and primary use determine total electricity consumption? Click a tab below to find out."
                    ),
                ]), width=7
            ),
            dbc.Col(width=3),
        ], justify="center"),
        html.Br(),
        dcc.Tabs(id='tabs', children=[
            dcc.Tab(label='Home', value='/',
                    style={'background-color': '#FADBD8', 'color': 'grey', 'border': '1px solid #FADBD8'}),
            dcc.Tab(label='Dataset', value='/dataset',
                    style={'background-color': '#AED6F1', 'color': 'grey', 'border': '1px solid #AED6F1'}),
            dcc.Tab(label='XGBoost', value='/xgboost',
                    style={'background-color': '#D2B4DE', 'color': 'grey', 'border': '1px solid #D2B4DE'}),
        ]),
    ],
    style={
        'background-color': 'rgba(255, 255, 255, 0.3)',
        'padding': '20px',
        'flex-grow': 1,
        'margin': '0',
    }),
],
style={'height': '100vh', 'display': 'flex', 'flex-direction': 'column', 'margin': '0', 'padding': '0'}
)



'''index_page = html.Div([
    html.Div([
        html.Br(),
        html.Br(),
        dbc.Row([
            dbc.Col([
                html.H1(children="Welcome to Foresquad"),
                #html.Img(src="lightning.png", width="200px", style={'margin-top': '10px'}),
            ], width=5),
            dbc.Col(width=5),
        ], justify='center'),
        dbc.Row([
            dbc.Col(
                html.Div([
                    html.H4(
                        children="To what extent do weather, weekday, and primary use determine total electricity consumption? Click a tab below to find out."
                    ),
                ]), width=7
            ),
            dbc.Col(width=3),
        ], justify="center"),

        dcc.Tabs(id='tabs', children=[
            dcc.Tab(label='Home', value='/'),
            dcc.Tab(label='Dataset', value='/dataset'),
            dcc.Tab(label='XGBoost', value='/xgboost'),
        ]),
    ]),
    ],
    style={
        'background-image': f'url({background_image_path})', 
        'background-size': 'cover',  # Use 'cover' to ensure the image covers the entire background
        'background-position': 'center',
        'background-repeat': 'no-repeat',
        'height': '100vh',
        'display': 'flex',
        'flex-direction': 'column',
        'background-color': 'rgba(255, 255, 255, 0.3)',  # Adjust the alpha channel (0.5 for 50% transparency)
    }
)

'''
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
    if 'log_meter_reading' not in filtered_data.columns:
        raise PreventUpdate("Column 'log_meter_reading' not found in the filtered data")

    # Create the KDE plot
    fig = go.Figure()

    # Plot the KDE using Plotly
    fig.add_trace(go.Histogram(
        x=filtered_data['log_meter_reading'],
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
    if 'log_meter_reading' not in filtered_data.columns:
        raise PreventUpdate("Column 'log_meter_reading' not found in the filtered data")

    # Create the histogram plot
    fig = go.Figure()

    # Plot the KDE using Plotly
    fig.add_trace(go.Histogram(
        x=filtered_data['log_meter_reading'],
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

    html.Div([
        dcc.Dropdown(
            id='fold-dropdown',
            options=[
                {'label': f'Fold {i + 1}', 'value': i + 1} for i in range(5)
            ],
            value=1,
            clearable=False,
            style={'width': '50%'}
        ),
        dcc.Graph(id='prediction-graph'),

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
            style={'width': '50%'}  # Corrected indentation
        ),

        # Create a single graph for both train and test data
        dcc.Graph(id='metric-plot'),    
    ],
    style={
        'background-color': 'lavender'
    }),
])
# Callback to update the graph based on selected fold
@app.callback(Output('prediction-graph', 'figure'), [Input('fold-dropdown', 'value')])
def update_graph(selected_fold):
    # Get the data for the selected fold
    fold_index = selected_fold - 1
    df_train, df_test = df.iloc[train_index[fold_index]], df.iloc[test_index[fold_index]]
    X_train, X_test = df_train[features], df_test[features]
    y_train, y_test = df_train[target], df_test[target]

    # Generate the plot for the selected fold
    img_base64 = generate_plot(XGB, X_train, y_train, X_test, y_test, selected_fold)

    # Return the figure for the graph
    return {
        'data': [],
        'layout': {
            'images': [
                {
                    'source': f'data:image/png;base64,{img_base64}',
                    'x': 0,
                    'y': 1,
                    'xref': 'paper',
                    'yref': 'paper',
                    'sizex': 1,
                    'sizey': 1,
                    'xanchor': 'left',
                    'yanchor': 'top',
                }
            ]
        }
    }



def generate_plot(XGB, X_train, y_train, X_test, y_test, fold):
    y_pred_train = XGB.predict(X_train)
    y_pred_test = XGB.predict(X_test)

    # DataFrames to store aggregated values for the current fold
    aggregated_actual_df = pd.concat([pd.DataFrame({'timestamp': df_train['timestamp'], 'actual': y_train}),
                                      pd.DataFrame({'timestamp': df_test['timestamp'], 'actual': y_test})])

    aggregated_predicted_df = pd.concat([pd.DataFrame({'timestamp': df_train['timestamp'], 'predicted_train': y_pred_train}),
                                         pd.DataFrame({'timestamp': df_test['timestamp'], 'predicted_test': y_pred_test})])

    # Group by timestamp and resample on a daily basis
    aggregated_actual_df.set_index('timestamp', inplace=True)
    aggregated_predicted_df.set_index('timestamp', inplace=True)

    aggregated_actual_daily = aggregated_actual_df.resample('D').mean()
    aggregated_predicted_daily = aggregated_predicted_df.resample('D').mean()

    # Plotting aggregated daily predictions for the current fold using matplotlib
    plt.figure(figsize=(12, 8))
    plt.plot(aggregated_actual_daily.index, aggregated_actual_daily['actual'], label='Actual', marker='o', color='green')
    plt.plot(aggregated_predicted_daily.index, aggregated_predicted_daily['predicted_train'], label='Predicted Train', marker='x', linestyle='--', color='blue')
    plt.plot(aggregated_predicted_daily.index, aggregated_predicted_daily['predicted_test'], label='Predicted Test', marker='x', linestyle='--', color='red')
    plt.title(f'Aggregated Daily Predictions vs Actual Values (Fold {fold})')
    plt.xlabel('Date')
    plt.ylabel('Target Variable')
    plt.legend()
    # Format x-axis ticks as months
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

    # Save the plot to a BytesIO object
    img_buf = BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)

    # Convert the plot to base64 for Dash
    img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')

    return img_base64



tabs_content = {
    '/': index_page,
    '/dataset': dataset_layout,
    '/xgboost': xgboost_layout,
}
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
    app.run_server( debug=True, host='0.0.0.0', port=8050)
