from dash import html
from dash import dcc
from dash import dash_table
from dash.dependencies import Input, Output,State
import matplotlib.pyplot as plt


cox =  html.Div(children=[
    html.Button('Run Cox Proportional Hazard', id='submit-cox',style={'width':'100%'}),
    html.Div(children=[html.Img(id='cox-plot-img',className='six columns'),html.Div(dash_table.DataTable(id='vif'),className='six columns')]), # img element
    #dash_table.DataTable(id='survival-table',page_size=10),
    dash_table.DataTable(id='data-table-cox',page_size=10),
    dcc.Markdown(id='assumptions_output', children='')
])