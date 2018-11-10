#import quandl
import io
import pandas as pd
import requests
from bs4 import BeautifulSoup
import numpy as np

##Gather Data
def get_states():
    states = pd.read_html('https://simple.wikipedia.org/wiki/List_of_U.S._states_by_area', header=0)
    states = states[0].drop([50])
    states = pd.Series(states['State'].values, index=states['Rank'])
    return states

states = get_states()

diseases = pd.DataFrame(columns=[])
RefID = ['rxmp-xjpc','ttj2-zsyk','i42d-szcv','efpc-rr7b','6rpz-c2y5','qvvb-s7gu','9qys-crt2','acdz-tk8j','mk6z-83q2','n24i-76tn','x2iq-5477','btcp-84tv','cucp-zsht','m6gf-vfkz','5pe9-px25','fhc9-h3em','5egk-p6rd','6tk5-h85s','8n2k-mkiw']
for x in RefID:
    print(x)
    #data url
    url="https://data.cdc.gov/api/views/"+x+"/rows.csv"
    s=requests.get(url).content
    results_df=pd.read_csv(io.StringIO(s.decode('utf-8')))
    # normalize columns
    if ' Reporting Area'  in results_df.columns:
        results_df.rename(columns={' Reporting Area':'Reporting Area'}, inplace=True)
    if 'MMWRYear' in results_df.columns:
        results_df.rename(columns={'MMWRYear':'MMWR Year'}, inplace=True)
    if 'MMWRWeek' in results_df.columns:
        results_df.rename(columns={'MMWRWeek':'MMWR Week'}, inplace=True)
    # get only data of states
    results_df = results_df[results_df['Reporting Area'].isin(states.str.upper().tolist())]
    results_df = results_df.sort_values(['MMWR Year', 'MMWR Week', 'Reporting Area'], ascending=[False, True, True]).reset_index(drop='True')
    maincolumns = results_df[['Reporting Area', 'MMWR Week', 'MMWR Year']]
    disease = results_df.loc[:,results_df.columns.str.contains('Current')]

    #concatenate the maincolumns once
    if 'Reporting Area' in diseases.columns:
        diseases = pd.concat([diseases, disease], axis=1,join_axes=[diseases.index])
    else:
        diseases = pd.concat([maincolumns, disease], axis=1,join_axes=[maincolumns.index])

    print(len(diseases.columns))

#remove columns which contains the string 'flag' and remove 'Current week'
cols = pd.Series(diseases.columns)
columns = []
for c in cols:
    if 'flag' not in c:
        columns.append(c)
#create column for the total number of notifiable disease cases of each state on each week
total = diseases.loc[:,diseases.columns.str.contains('Current')].sum(axis=1)
total = total.to_frame()
total = total.rename(columns= {0: 'Total'})
total = total['Total']


diseases = pd.concat([diseases[columns],total], axis=1,join_axes=[diseases.index])
diseases.fillna(value=0, inplace=True)
diseases.to_csv('Notifiable Diseases.csv')