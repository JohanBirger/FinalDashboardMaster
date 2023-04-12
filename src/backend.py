import pandas as pd
from lifelines.utils import survival_table_from_events
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
from lifelines.utils import datetimes_to_durations
from lifelines import CoxPHFitter
import os
import json
import base64
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import ipywidgets as widgets
from IPython.display import display
from IPython.display import clear_output
#from IPython.display import HTML, Javascript, display
import numpy as np
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import cufflinks as cf
import plotly.offline as pyo
import pickle
import time
from statsmodels.stats.outliers_influence import variance_inflation_factor
import sys
import re
   
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib import cm


def get_countries(df):
        countries = []
        for i in df.columns:
            if 'country' in i:
                countries.append(i)
            else:
                pass
        return countries

def convert_datetime(row):
    return str(row)[:4]



def getbinintervals(bininter):
    #'1,4,10'
    try:
        values=bininter.split(' ')
        lenght= len(values)
        first =" "
        last = " "
        midvalues = []
        for n,i in enumerate(values):
            print(n,i)
            if n ==0:
                first = (0.00000,int(i))
            elif n == lenght-1:
                last = (int(i),np.inf)
            else: 
                midvalues.append([values[n-1],i])
        fin =[(int(mid[0]),int(mid[1])) for mid in midvalues]
        fin.insert(0,first)
        fin.append(last)
        print(fin)
        return pd.IntervalIndex.from_tuples(fin)
    except:
        print('Error - Default binrange')
        return pd.IntervalIndex.from_tuples([(0, 1), (2, 4), (4, 10)])

    return pd.IntervalIndex.from_tuples([(0, 1), (2, 4), (4, 10)])

saved_presets = {'basecase':["event_dead", "Lifetime","country_United States"],'Ziplevel':["event_dead", "Lifetime","country_United States","Num_Firms_In_ZIP","ZIP_Business_Students","ZIP_STEM_Students"]}
init_presets = {'basecase':["event_dead", "Lifetime","country_United States"],'Ziplevel':["event_dead", "Lifetime","country_United States","Num_Firms_In_ZIP","ZIP_Business_Students","ZIP_STEM_Students"]}

#place in data object, access year_index dict
valuerange = [1,50]
start = 1973
end =2024
global year_index
year_index= {}
for i in range(valuerange[0]-1,valuerange[1]+2):
    #print(start+i)
    year_index[i] = pd.to_datetime(str(start+i))


class Data():
    # Get the absolute path of the directory containing the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Define the absolute path of the CSV file
    pkl_path = os.path.join(current_dir,'resources','for_dashboard_0331v4.pkl')
    
    df = pd.read_pickle(pkl_path)
    df = df.fillna(0)
    

   
    

    
    columns = list(df.columns)
    columns.remove('Founded_Year')
    columns.remove('Date_Exit')
    columns.remove('Died')
    columns.remove('Founded')
    columns.remove('Name')



    saved_presets['Graphing'] = columns
    init_presets['Graphing'] = columns
    
    
    #- Presets for analysis in report
    saved_presets['ZIPLEVEL- USA'] = ['event_dead','Lifetime','country_United States','Num_Firms_In_ZIP','ZIP_Business_Students','ZIP_STEM_Students']
    saved_presets['CITYLEVEL- USA'] = ['event_dead','Lifetime','country_United States','Num_Firms_In_CITY','City_Business_Students','City_STEM_Students']
    saved_presets['DESCRIPTIVE STATISTICS'] = columns
    saved_presets['ZIPLEVEL- SWEDEN'] = ['event_dead','Lifetime','country_Sweden','Num_Firms_In_ZIP','ZIP_Business_Students','ZIP_STEM_Students']
    saved_presets['CITYLEVEL- SWEDEN']= ['event_dead','Lifetime','country_Sweden','Num_Firms_In_CITY','City_Business_Students','City_STEM_Students']
    
    init_presets['ZIPLEVEL- USA'] = ['event_dead','Lifetime','country_United States','Num_Firms_In_ZIP','ZIP_Business_Students','ZIP_STEM_Students']
    init_presets['CITYLEVEL- USA'] = ['event_dead','Lifetime','country_United States','Num_Firms_In_CITY','City_Business_Students','City_STEM_Students']
    init_presets['DESCRIPTIVE STATISTICS'] = columns
    init_presets['ZIPLEVEL- SWEDEN'] = ['event_dead','Lifetime','country_Sweden','Num_Firms_In_ZIP','ZIP_Business_Students','ZIP_STEM_Students']
    init_presets['CITYLEVEL- SWEDEN']= ['event_dead','Lifetime','country_Sweden','Num_Firms_In_CITY','City_Business_Students','City_STEM_Students']


    global cox_summary
    global cox_assumptions

    def get_year(self,index:int):
         return year_index.get(index)

    def read_preset(self):
         global saved_presets
         ret = []
         for name,content in saved_presets.items():
              ret.append(name)
         return ret
    
    def antilog(self,row,target):
        row[target] = 10** row[target]
        return row 

    def save_preset(self,name,content):
         global saved_presets
         saved_presets[name] = content
         print(name,content)
         return name
    
    def load_preset(self,name):
         global saved_presets
         return saved_presets[name]
         

    def init_presets(self):
         return init_presets
    
    
    def run_km(self,df,count):
        countries = get_countries(df)
        
        country_dfs = {}
        for country in countries:
            country_dfs[country] = df.loc[df[country]==1]
        tables = {}
        plots = {}
        for country,df in country_dfs.items():
            table = survival_table_from_events(df['Lifetime'], df['event_dead'])
            tables[country]=table
            kmf = KaplanMeierFitter()
              # Fit the Kaplan-Meier model to the independent variable and the lifetime
            kmf.fit(df['Lifetime'], df['event_dead'], label=country)
            # Plot the Kaplan-Meier curve for the country selection
            #buf = io.BytesIO() # in-memory files
            buf = io.BytesIO() # in-memory files
            kmf.plot()
            # Save the generated image as a PNG file
        plt.ylabel('Survival probability')
        plt.savefig(buf, format = "png",dpi=500,bbox_inches='tight')
        plt.close()
        data = base64.b64encode(buf.getbuffer()).decode("utf8") # encode to html elements
        #image_pkl_name = f"./components/plots/km_plot_{count}.png"
        #plt.savefig(image_pkl_name)
        buf.close()

        return data,tables

    def calculate_vif(self,dataframe):
        """
        Calculate the VIF (Variance Inflation Factor) for each feature in a pandas dataframe.

        Args:
        df (pandas dataframe): Input dataframe with features to calculate VIF.

        Returns:
        vif_data (pandas dataframe): A dataframe with the VIF value for each feature.
        """
        cols = list(dataframe.columns)
        country = ''
        for col in cols:
            if 'country' in col:
                country = col 
        dataframe= dataframe.drop(columns=[country])
        vif_data = pd.DataFrame()
        
        vif_data["feature"] = dataframe.columns
        vif_data["VIF"] = [variance_inflation_factor(dataframe.values, i) for i in range(dataframe.shape[1])]
        c_dir = os.path.dirname(os.path.abspath(__file__))
        reg_path = os.path.join(c_dir,'regressionslatex','vifs',str(cols[-1])+str(list(country)))
        with open(reg_path,'w') as f:
            f.write(vif_data.to_html())
        return vif_data
    
    def run_cox(self,dataframe):
        global cox_summary
        global cox_assumptions
        cols = list(dataframe.columns)
        country = ''
        for col in cols:
            if 'country' in col:
                country = col 

        cph = CoxPHFitter()
        cph.fit(dataframe.drop(columns=[country]),'Lifetime','event_dead',show_progress=True)
        buf_assumptions = io.StringIO()   
        sys.stdout = buf_assumptions    
        cph.check_assumptions(dataframe.drop(columns=[country]))    
        sys.stdout = sys.__stdout__    
        cox_assumptions = buf_assumptions.getvalue()    
        buf_assumptions.close()    
        c_dir = os.path.dirname(os.path.abspath(__file__))    
        reg_path = os.path.join(c_dir,'regressionslatex',country+str(cols[-1]))    
        with open(reg_path,'w') as f:
            buf_latex = io.StringIO()
            sys.stdout = buf_latex
            cph.print_summary(columns=["coef","exp(coef)", "se(coef)", "p"],style="latex")
            sys.stdout = sys.__stdout__
            cox_latex = buf_latex.getvalue()
            f.write(cox_latex)
            buf_latex.close()
        cox_summary= cph.summary        
        buf_cox = io.BytesIO()        
        plt.figure()        
        cph.plot()        
        plt.title('Grouped on:{}'.format(country))        
        plt.savefig(buf_cox, format="png",bbox_inches='tight')        
        plt.savefig(os.path.join(c_dir,'plots',country+str(cols[-1]))+'.png',format="png",dpi=500,bbox_inches='tight')
        plt.close()
        data = base64.b64encode(buf_cox.getbuffer()).decode("utf8") # encode to html elements
        buf_cox.close()
        
        return data
    
    def get_countries_list(self,df):
        countries = []
        for i in df.columns:
            if 'country' in i:
                countries.append(i)
            else:
                pass
        return countries
 
    def get_cox_summary(self):
        global cox_summary
        
        return cox_summary.reset_index()
    
    def get_cox_assumptions(self):
        global cox_assumptions
        if "Proportional hazard assumption looks okay." in cox_assumptions:
            return str(cox_assumptions)
        elif "Assumption check not compatible with stratified yet" in cox_assumptions:
            return str(cox_assumptions)
        else:
            message = cox_assumptions
            message_listed = message.split('\n') 
            general = ''.join(message_listed[:8])
            advice = message.split('1. Variable')
            
            return advice[1] +' \n '+general

    def get_part_df(self,df,columns):
         try:
            #print(columns)
            columns.append('Date_Exit')
            columns.append('Founded_Year')
            columns.append('Died')
            columns.append('Founded')
            columns.append('Name')
            columns.append('Status')
            return df[columns]
         except:
            columns.append('Died')
            columns.append('Founded')
            columns.append('Name')
            return df[columns]
             
    
  
    
    def bin(self,df,targets,binintervalls):
        bins = getbinintervals(binintervalls)
        dff = df.copy()
        #antilogged = []
        #print(df[targets[0]])
        #for val in dff[targets[0]].values:
        #    if val !=0:
        #        antilogged.append(10**val)
        #    else:
        #        antilogged.append(0)
        #print(antilogged)
        #dff[targets[0]]=antilogged
        #print(df[targets[0]])
        dff[targets[0]+'_BIN'] = pd.cut(dff[targets[0]], bins, labels=False)
        # Convert the binned column into dummy variables
        dummies = pd.get_dummies(dff[targets[0]+'_BIN'], prefix=targets[0]+'_BIN')
        #print(df)
        ret = pd.concat([dff, dummies], axis=1)
        #Because JSON SERIALIZATION CANT HANDLE BRACKETS
        ret = ret.drop(columns=[targets[0]+'_BIN'])
        return ret

    
    def make_survival_table(self,dictonary):
        df_out = {}
        for country, table in dictonary.items():    
            table = table.reset_index()
            #print(table.columns)
            table['event_at'] = range(0,len(table))
            #print(table.index)
            df_out[country] = table
        df_concat = pd.concat(df_out.values())
        df_agg = df_concat.groupby('event_at')['removed','censored','censored','entrance','at_risk'].sum() 
        return df_agg.reset_index()
    
    def range_df(self,dataframe,inputRange):
        try:
            minyear = year_index.get(inputRange[0])
            maxyear = year_index.get(inputRange[1])
            
            #print(dataframe['Founded_Year'][:1])
            #dataframe = dataframe.fillna(pd.to_datetime('1930'))
            dataframe = dataframe.loc[dataframe['Founded_Year']> minyear]
            dataframe = dataframe.loc[dataframe['Date_Exit']< maxyear]
        except:
            return dataframe
        return dataframe
    
    def interval_df(self,dataframe,inputInterval):
        try:
            interval = inputInterval
            dataframe= dataframe.loc[dataframe['Lifetime']<= interval]
            return dataframe
        except:
            return dataframe
        
    def datetime_for_table(self,temp):
        try:
            temp['Founded'] = temp['Founded'].apply(convert_datetime)
            temp['Died'] = temp['Died'].apply(convert_datetime)
        except:
            return temp
        return temp.drop(columns=['Founded_Year','Date_Exit'])    

    def get_static_plot(self,type,id):
        if 'km' in type:
             return (f"km_plot_{id}.pickle")
    


    



