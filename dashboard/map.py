import dash_core_components as dcc
import dash_html_components as html


TEMPLATE = 'ggplot2'

BUTTON_LAYOUT = [
    dcc.Link(
        html.Button('HOME', id='home-button', className="mr-1"),
        href='/'),
    dcc.Link(
        html.Button('Dataset', id='dataset-button', className="mr-1"),
        href='/dataset'),
    dcc.Link(
        html.Button('Model Predictions', id='model-button', className="mr-1"),
        href='/model'),
]

