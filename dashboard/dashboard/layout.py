from dash import dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go

BUTTON_LAYOUT = [
    dcc.Link(
        html.Button('Dataset', id='dataset-button', className="mr-1"),
        href='/dataset'),
    dcc.Link(
        html.Button('Linear Regression Predictions', id='linear-button', className="mr-1"),
        href='/linear'),
    dcc.Link(
        html.Button('Decision Tree Predictions', id='dt-button', className="mr-1"),
        href='/model'),
    dcc.Link(
        html.Button('XGBoost Predictions', id='xgb-button', className="mr-1"),
        href='/xgboost'),
    dcc.Link(
        html.Button('ARIMA Predictions', id='arima-button', className="mr-1"),
        href='/arima'),
]


