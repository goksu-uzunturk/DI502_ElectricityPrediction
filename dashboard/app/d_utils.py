import dash
from dash import html


def make_break(num_breaks):
    br_list = [html.Br()] * num_breaks
    
    return br_list

