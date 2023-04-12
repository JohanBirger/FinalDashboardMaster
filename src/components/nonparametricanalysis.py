from dash import html
from dash import dcc
from dash import dash_table
from dash.dependencies import Input, Output,State
import matplotlib.pyplot as plt


km =  html.Div(children=[
    html.Button('Run Kaplan-meier', id='submit-km',style={'width':'100%'}),
    html.Div([
        html.Img(id='survival-plot',style={'width':'100%','alignItems': 'center'}),
        html.Div([dash_table.DataTable(id='survival-table',page_size=20)],style={'width':'100%','alignItems': 'center'}),
        ],style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center'})
    
])