from dash import Dash, dcc, html
from backend import Data
from datetime import datetime as dt

data = Data()
df = data.df.copy()
columns = data.columns

tools = html.Div(children=[
    dcc.Dropdown(id='variables',multi=True,placeholder='Select Variables',options=[{'label': col, 'value': col} for col in columns]),
    html.Div(
    children=[
        dcc.Dropdown(
            options=[{'label': preset, 'value': preset} for preset in data.init_presets()],
            placeholder="Choose preset",
            id='presets',
            value='basecase',
            style={'width': '60%'}
        ),
        html.Button('Load preset', id='loadpreset', style={'width': '15%'}),
        dcc.Input(id='presetname', type='text', placeholder='Enter preset name', style={'width': '30%'}),
        html.Button('Save preset', id='savepreset', style={'width': '15%'})
    ],
    style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center'}
    ),

    html.Div([
        dcc.Dropdown(
                    options=[{'label': col, 'value': col} for col in columns],
                    placeholder="Select a bin variable",
                    id='binning-drop',
                    multi=True,
                    style={'width': '50%'}
        ),
        dcc.Input(
                    id='bininterval_input',
                    type='text', 
                    placeholder='5 5',
                    style={'width': '30%'}
        )
    ],
    style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center'}
    ),

    html.Button('Confirm Configuration',id='confirmselection'),
    html.Div(id='output_loaded'),
    html.Div(id='output_info')
    
])