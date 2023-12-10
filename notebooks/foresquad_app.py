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
from dash.exceptions import PreventUpdate
import os
import matplotlib.pyplot as plt
import joblib
import zipfile
import seaborn as sns

# Load data function
def load_data():
    # Specify the ZIP file name
    zip_filename = "../dataset/filtered.zip"

    # Extract the model file from the ZIP archive
    with zipfile.ZipFile(zip_filename, "r") as archive:
        # Extract the model file (named "your_model.pkl" in this example)
        archive.extract("filtered.pkl")
        
    # Load the model
    df = joblib.load("filtered.pkl")  # Replace with "pickle.load" if you used pickle

    os.remove("filtered.pkl")

    return df
    

# Log transformation function
def log_transformation(data):
    df['log_meter_reading']=np.log1p(df['meter_reading'])
    df['log_square_feet']=np.log1p(df['square_feet'])
    return df

# Load data and perform log transformation
df = load_data()
train_data = df.copy()
train_data = log_transformation(train_data)
train_data['timestamp'] = pd.to_datetime(train_data['timestamp'])

# Create new columns
train_data['hour'] = train_data['timestamp'].dt.hour.astype(np.uint8)
train_data['month'] = train_data['timestamp'].dt.month.astype(np.uint8)
train_data['week'] = train_data['timestamp'].dt.week.astype(np.uint8)
train_data[['year','weekofyear','dayofweek']]= np.uint16(train_data['timestamp'].dt.isocalendar())

merged_data = train_data.copy()

# Assuming train_data is your DataFrame
hour_mean = train_data.groupby('hour')['log_meter_reading'].mean()

# Create Seaborn line plot
hour_mean.plot(kind='line', color='skyblue')

# Convert Seaborn plot to Plotly figure
fig_hour_mean = make_subplots(rows=1, cols=1)

# Add Seaborn data to Plotly subplot
trace_hour_mean = go.Scatter(x=hour_mean.index, y=hour_mean.values, mode='lines', line=dict(color='skyblue'))
fig_hour_mean.add_trace(trace_hour_mean)

# Update layout with axis labels
fig_hour_mean.update_layout(
    xaxis=dict(title='Hour'),
    yaxis=dict(title='Mean of Log Meter readings'),
    paper_bgcolor='lavender',
    showlegend=False
)
    

XGBoost_MODEL_DESCRIPTION = '''
    The XG Boost forecasting model was trained on historical meter readings, weather, and building data
    from 2016-2017. Temperature readings are from site_id - 1 and site_id - 6.
'''

Dataset_DESCRIPTION = '''
    The dataset is taken from [ASHRAE Great Energy Predictor III Competition](https://www.kaggle.com/c/ashrae-energy-prediction) 
'''

XGBresults = pd.read_csv('results/XGB_results.csv')

# Load results information from CSV
results_info = pd.read_csv('results_info.csv')


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
    html.P(Dataset_DESCRIPTION),

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
        options=[{'label': primary_use, 'value': primary_use} for primary_use in merged_data['primary_use'].unique()],
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
    filtered_data = merged_data[merged_data['site_id'] == selected_site]

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
    filtered_data = merged_data[merged_data['primary_use'] == selected_primary_use]

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

day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
day_df = train_data.groupby(['dayofweek']).log_meter_reading.mean().reset_index()

# Create Seaborn plot
p = sns.lineplot(x=day_df['dayofweek'], y=day_df['log_meter_reading'], color='purple')
p.set_xticks(range(5))
p.set_xticklabels(day_labels)
plt.xlabel('Days of the week')
plt.ylabel("Log of Meter readings")

# Convert Seaborn plot to Plotly figure
fig8 = make_subplots(rows=1, cols=1)

# Add Seaborn data to Plotly subplot
trace = go.Scatter(x=day_df['dayofweek'], y=day_df['log_meter_reading'], mode='lines', line=dict(color='purple'))
fig8.add_trace(trace)

# Update layout if necessary
fig8.update_layout(xaxis=dict(title='Days of the week'),
                    yaxis=dict(title='Log of Meter readings'),
                   paper_bgcolor='lavender',
                    showlegend=False)

# Assuming train_data is your DataFrame
month_mean_data = train_data.groupby('month')['log_meter_reading'].mean()

# Create Seaborn line plot
month_mean_data.plot(kind='line', color='skyblue')

# Plot the line chart using Plotly Express
fig_month_mean = px.line(
    data_frame=pd.DataFrame({'Month': month_mean_data.index, 'Mean Meter Reading': month_mean_data.values}),
    x='Month',
    y='Mean Meter Reading',
    labels={'x': 'Month', 'y': 'Mean Meter Reading'},
    title='Monthly Mean Meter Reading',
    line_shape='linear',
    render_mode='svg'
)
plt.figure(figsize=(15, 8))
sns.set(style="whitegrid")

# Update layout with lavender background color
fig_month_mean.update_layout(
    paper_bgcolor='lavender',
)

# Callback to update the model performance graph based on the selected time interval
@app.callback(
    Output('model-performance-graph', 'figure'),
    [Input('time-interval-dropdown', 'value')]
)
def update_model_performance(selected_interval):
    # Choose the appropriate graph based on the selected interval
    if selected_interval == 'hourly':
        return fig_hour_mean
    elif selected_interval == 'weekly':
        return fig8
    elif selected_interval == 'monthly':
        return fig_month_mean
    

# Add the layout to the app callback
xgboost_layout = html.Div([
    html.H1('XGBoost Predictions'),
    html.P(XGBoost_MODEL_DESCRIPTION),

    # Dropdown for selecting fold
    dcc.Dropdown(
        id='fold-dropdown',
        options=[{'label': f'Fold {idx + 1}', 'value': idx} for idx in range(len(results_info))],
        value=0  # Default selected fold
    ),

    # Graph for displaying selected fold
    dcc.Graph(id='fold-plot'),

    # Metric Selection Dropdown
    dcc.Dropdown(
            id='metric-dropdown',
            options=[
                {'label': 'MAE', 'value': 'mae'},
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
})

# Callback to update the displayed graph based on the selected fold
@app.callback(
    Output('fold-plot', 'figure'),
    [Input('fold-dropdown', 'value')]
)
def update_fold_plot(selected_fold):
    actual_path = results_info['actual_data_paths'][selected_fold]
    predicted_path = results_info['predicted_data_paths'][selected_fold]

    actual_data = pd.read_csv(actual_path)
    predicted_data = pd.read_csv(predicted_path)

    figure = {
        'data': [
            {
                'x': actual_data['timestamp'],
                'y': actual_data['actual'],
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Actual'
            },
            {
                'x': predicted_data['timestamp'],
                'y': predicted_data['predicted_test'],
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Predicted'
            }
        ],
        'layout': {
            'title': f'Fold {selected_fold + 1} - Aggregated Daily Predictions vs Actual Values',
            'xaxis': {'title': 'Date'},
            'yaxis': {'title': 'Target Variable'},
        }
    }

    return figure

'''

    # Iterate over the results_info to load and display data
    *[dcc.Graph(
        id=f'fold-{idx + 1}-plot',
        figure={
            'data': [
                {
                    'x': pd.read_csv(actual_path)['timestamp'],
                    'y': pd.read_csv(predicted_path)['predicted_test'],
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': f'Fold {idx + 1}'
                }
                for idx, (actual_path, predicted_path) in enumerate(zip(results_info['actual_data_paths'], results_info['predicted_data_paths']))
            ],
            'layout': {
                'title': f'Fold {idx + 1} - Aggregated Daily Predictions vs Actual Values',
                'xaxis': {'title': 'Date'},
                'yaxis': {'title': 'Target Variable'},
            }
        }
    ) for idx in range(len(results_info))],


'''


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
