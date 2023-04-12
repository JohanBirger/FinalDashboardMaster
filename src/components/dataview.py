from dash import html
from dash import dcc
from dash.dependencies import Input, Output,State
import matplotlib.pyplot as plt
from dash import dash_table

graphtypes = ['bar','scatter','bubble'] 

table =  html.Div(children=[
    #html.Div(children=[html.Img(id='survival-plot',className='six columns')]), # img element
    #dash_table.DataTable(id='survival-table',page_size=10),
    
    html.Div([
    dcc.Loading(
                    id="loading_graph",
                    children=[html.Div([html.Div(id="loading_graph_out")])],
                    type="circle",
                ),
            dcc.Graph(
                figure={},
                id='interactive-graph',
                style={'width':'100%','align-items': 'center'}  
                #className='nine columns'
            ),
            html.Div([
                    dcc.Dropdown(
                        placeholder="Choose x-axis",
                        id='xaxis_drop',
                        style={'width':'100%','align-items': 'center'}
                    ),
                    html.Br(),
                    dcc.Dropdown(
                        placeholder="Choose y-axis",
                        id='yaxis_drop',
                        style={'width':'100%','align-items': 'center'}
                    ),
                    html.Br(),
                    dcc.Dropdown(
                        placeholder="Select graphtype",
                        id='graphtype',
                        options=[{'label': type, 'value': type} for type in graphtypes],
                        style={'width':'100%','align-items': 'center'}
                    ),
                    html.Br(),
                    html.Button('Show',id='show_graph',style={'width':'100%','align-items': 'center'})
                ],style={'width':'50%','align-items': 'center'})],
                style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center'}),
                dash_table.DataTable(id='describe-table')
    
        ])