from dash import Dash, dcc, html
from backend import Data
from datetime import datetime as dt
from components import dataview, nonparametricanalysis,fullanalysis
from dash import dash_table



# Define the layout for the entire app with the four tabs
all_tabs = html.Div([
    dash_table.DataTable(id='data-table',page_size=10),
    dcc.Tabs([
        dcc.Tab(label='Descriptive Statistics', children=dataview.table),
        dcc.Tab(label='Non-paramatric Analysis', children=nonparametricanalysis.km),
        dcc.Tab(label='Parametric Analysis', children=fullanalysis.cox),
        dcc.Tab(label='Documentation', children=None),
        
    ])
])