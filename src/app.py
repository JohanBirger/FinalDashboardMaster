import dash
import time
import pandas
from dash import html, dcc, callback_context
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output,State
from backend import Data
from components import toolcomponents,tabs
import plotly.express as px
import numpy as np
import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server


app.layout = html.Div([toolcomponents.tools,tabs.all_tabs])


data = Data()
df = data.df
initdf = df





#************
# TOOLBAR 
#************
@app.callback(
    Output('output_info', 'children'),
    Input('savepreset', 'n_clicks'),
    State('presetname', 'value'),
    State('variables', 'value'),
)
def save_preset(n_clicks, preset_name, selected_variables):
    if n_clicks is None:    
        raise PreventUpdate 
     
    if preset_name is None or preset_name == '':
        return 'Please enter a valid preset name.'

    if selected_variables is None or len(selected_variables) == 0:
        return 'Please select at least one variable.'
    data.save_preset(preset_name,selected_variables)

    return f'Preset saved: {preset_name}'

#output presets from as loaded
@app.callback(
    Output('output_loaded', 'children'),
    Input('loadpreset', 'n_clicks'),
    State('presets', 'value')
)
def output_loaded_preset(n_clicks, preset_name):
    if n_clicks is None:
        raise PreventUpdate
    if not preset_name:
        return []
        
    return f'Preset loaded {preset_name}'

#load presets into selection
@app.callback(
    Output('variables', 'value'),
    Input('loadpreset', 'n_clicks'),
    #State('binning-drop', 'value'),
    State('presets', 'value')
)
def update_variables_from_preset(n_clicks, preset_name):
    if n_clicks is None:
        raise PreventUpdate
    
    if not preset_name:
        return []

    return data.load_preset(preset_name)

#preset refresh
@app.callback(
    Output('presets', 'options'),
    Input('savepreset', 'n_clicks'),
    prevent_initial_call=True
)
def update_preset_list(n_clicks):
    time.sleep(0.2)
    presets = [{'label': preset, 'value': preset} for preset in data.read_preset()]
    
    return presets


# **************
#  DATA TABLE
# **************

@app.callback(
    Output('data-table', 'data'),
    Input('confirmselection','n_clicks'),
    [State('variables','value')],
    State('binning-drop','value'),
    State('bininterval_input','value'),
    prevent_initial_call=True
)
def update_data_table(clicks, value,bintarget,binintervals):
    #try:
        if bintarget and binintervals:
            try:
                df = data.bin(data.get_part_df(data.range_df(data.interval_df(initdf,10),[1,50]),value),bintarget,binintervals)
            except:
                return pd.DataFrame(['Error','Invalid bintarget or bin interval']).to_dict('records')
        else:
            try:
                df = data.get_part_df(data.range_df(data.interval_df(initdf,10),[1,50]),value)
            except:
                return pd.DataFrame(['Error','Unknown']).to_dict('records')

        delimitors = []
        for i in value:
            if 'country' in i:
                delimitors.append(i)
        print('delimitors:',delimitors)
        many_df = df.loc[(df[delimitors] == 1).any(axis=1)]
        print('many_df:',many_df)
        dataView = data.datetime_for_table(many_df)
        print('dataview',dataView)
        return dataView.to_dict('records')
    #except:
        #return pd.DataFrame(['Error','Updating table']).to_dict('records')
    

@app.callback(
        Output('describe-table','data'),
        Input('data-table','data'),
        prevent_initial_call = True,
)
def show_descriptive(data):
    desc = pd.DataFrame.from_dict(data)
    desc = desc.describe().reset_index()
    return desc.to_dict('records')

#****************
# Kaplan-Meier
#****************
#plot-static trigger(output_selection)
@app.callback(
    Output('survival-plot', 'src'),
    Input('submit-km', 'n_clicks'),
    State('data-table','data'),
    prevent_initial_call=True,
    priority=1  # prioritize this callback over the next one
)
def update_static_plot(clicks,dataframe):
    kmdf = pd.DataFrame.from_dict(dataframe)
    plot,survival_tables = data.run_km(kmdf,clicks)

    return "data:image/png;base64,{}".format(plot)

@app.callback(
    Output('survival-table', 'data'),
    Input('survival-plot','src'),
    State('data-table','data'), 
    prevent_initial_call=True,
    priority=0  # lowest priority
)
def update_data_table_survival(mathilda,dataframe):
    kmfortable = pd.DataFrame.from_dict(dataframe)
    plot,survival_tables = data.run_km(kmfortable,mathilda)
    df_survival = data.make_survival_table(survival_tables)
    return df_survival.to_dict('records')

#**************
# VIF
#**************
@app.callback(
        Output('vif','data'),
        Input('submit-cox','n_clicks'),
        State('binning-drop','value'),
        State('data-table','data'),
        prevent_initial_call= True
)
def run_vif(n_clicks,bin,dataframe):
    dfvif = pd.DataFrame.from_dict(dataframe)
    temp = dfvif.copy()
    if bin is not None:
        if len(bin)>0:
            try:
                vif_df = pd.DataFrame(['VIF Not Run, due to categorical variables.'])
                #print(bin)
                return vif_df.to_dict('records')
            except:
                pass
        else:
            vif_df = data.calculate_vif(temp.drop(columns=['Died', 'Founded', 'Name','Status']))
            #print(bin)
            return vif_df.to_dict('records')  
    vif_df = data.calculate_vif(temp.drop(columns=['Died', 'Founded', 'Name','Status']))
    return vif_df.to_dict('records')

#*************************
# Cox Proportional Hazard
#*************************
#plot-static trigger(output_selection)
@app.callback(
    Output('cox-plot-img', 'src'),
    Input('vif', 'data'),
    #[State('strata-dropdown','value')],
    #[State('use_strata','value')],
    [State('binning-drop','value')],
    [State('data-table','data')]
    ,prevent_initial_call=True
)
def update_static_plot(vif,bin,dataframe):
    dfcox = pd.DataFrame.from_dict(dataframe)
    try:
        dfcox['Num_Firms_In_ZIP'] = dfcox['Num_Firms_In_ZIP'].apply(lambda x: np.log(x) if x != 0 else x)
        dfcox['Num_Firms_In_CITY'] = dfcox['Num_Firms_In_CITY'].apply(lambda x: np.log(x) if x != 0 else x)
    except:
        try:
            dfcox['Num_Firms_In_CITY'] = dfcox['Num_Firms_In_CITY'].apply(lambda x: np.log(x) if x != 0 else x)
        except:
            try:
                dfcox['Num_Firms_In_ZIP'] = dfcox['Num_Firms_In_ZIP'].apply(lambda x: np.log(x) if x != 0 else x)
            except:
                pass
        pass


    
    if bin is not None:
        if len(bin) > 0:
            temp= dfcox.copy()
            plot = data.run_cox(temp.drop(columns=['Died', 'Founded', 'Name','Status',bin[0]]))
            #print(bin)
            return "data:image/png;base64,{}".format(plot)
        else:
            temp= dfcox.copy()
            plot = data.run_cox(temp.drop(columns=['Died', 'Founded', 'Name','Status'])) 
            #print(bin)
            return "data:image/png;base64,{}".format(plot)                 
    else:
        temp= dfcox.copy()
        #print('Should NOT have bins',temp.columns)
        plot = data.run_cox(temp.drop(columns=['Died', 'Founded', 'Name','Status'])) 
        #print(bin)
        return "data:image/png;base64,{}".format(plot)

@app.callback(
        Output('data-table-cox','data'),
        Input('cox-plot-img','src'),prevent_initial_call=True
)
def update_cox_data(something):
    return data.get_cox_summary().to_dict('records')


@app.callback(
        Output('assumptions_output','children'),
        Input('cox-plot-img','src'),prevent_initial_call=True
)
def update_assumptions_out(something):
    return data.get_cox_assumptions()
   
#******************
# Interactive Graph
#******************

@app.callback(
    Output('interactive-graph','figure'),
    Input('show_graph','n_clicks'),
    State('graphtype','value'),
    State('xaxis_drop','value'),
    State('yaxis_drop','value'),
    State('data-table', 'data'),
    prevent_initial_call = True
)
def update_interactive_graph(clicked,type,x,y,data):
    #global df
    #dff=df
    dff = pd.DataFrame.from_dict(data)
    print(dff)
    countries = []
    for col in list(dff.columns):
        if 'country' in col:
            countries.append(col)
    def countryCol(row):
        for col in countries:
            if row[col] != None:
                if row[col]==1:
                    row['country'] = col[8:] 
        return row

    dff = dff.apply(countryCol, axis=1)
    
    if type == 'scatter':
        try:
            fig = px.scatter(dff,x,y,color=x)
        except:
            fig = px.scatter(dff,'Lifetime','event_dead',color='Lifetime')

        fig = fig.update_traces(dict(marker_line_width=0))
    if type == 'bubble':
        dff[x+'_COUNT'] = dff.groupby(x)[x].transform('count')
        try:
            fig = px.scatter(dff,x,y,color=x,size=x+'_COUNT')
        except:
            fig = px.scatter(dff,'Lifetime','event_dead',color='Lifetime')

        fig = fig.update_traces(dict(marker_line_width=0))
    else:
        try:
            fig = px.bar(dff,x,y,color=x)
        except:
            fig = px.bar(dff,'Lifetime','event_dead',color='Lifetime')

        fig = fig.update_traces(dict(marker_line_width=0))
    return fig


@app.callback(
    Output('xaxis_drop','options'),
    Output('yaxis_drop','options'),
    Input('confirmselection','n_clicks'),
    State('variables','value')  
)
def update_interactive_graph_drop(n_clicks, value):
    if n_clicks is None:
        raise PreventUpdate
    selected = []
    for i in value:
        #print(i) 
        selected.append(i)
        #selected.append(i.get('value'))

    xoptions = [{'label': 'X:'+item, 'value': item} for item in selected]
    yoptions = [{'label': 'Y:'+item, 'value': item} for item in selected]
    return xoptions,yoptions

#**************
#Loading wheels
#**************
@app.callback(Output("loading_graph_out", "children"), Input("graphtype", "value"))
def input_triggers_nested(value):
    time.sleep(2.5)
    return None

if __name__ == "__main__":
    app.run_server(debug=False)
