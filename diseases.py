##import quandl
import io
import os
import pandas as pd
from sodapy import Socrata
import requests
from bs4 import BeautifulSoup
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
##style.use('fivethirtyeight')
style.use('ggplot')

##Gather Data
def get_states():
    states = pd.read_html('https://simple.wikipedia.org/wiki/List_of_U.S._states_by_area', header=0)
    states = states[0].drop([50])
    states.to_pickle('states.pickle')

if not os.path.isfile('states.pickle'):
    get_states()

states = pd.read_pickle('states.pickle')

states = pd.Series(states['State'].values, index=states['Rank'])

diseases = pd.DataFrame(columns=[])
RefID = ['rxmp-xjpc','ttj2-zsyk','i42d-szcv','efpc-rr7b','6rpz-c2y5','qvvb-s7gu','9qys-crt2','acdz-tk8j','mk6z-83q2','n24i-76tn','x2iq-5477','btcp-84tv','cucp-zsht','m6gf-vfkz','5pe9-px25','fhc9-h3em','5egk-p6rd','6tk5-h85s','8n2k-mkiw']
for x in RefID:
    print(x)
    url="https://data.cdc.gov/api/views/"+x+"/rows.csv"
    s=requests.get(url).content
    results_df=pd.read_csv(io.StringIO(s.decode('utf-8')))
    if ' Reporting Area'  in results_df.columns:
        results_df.rename(columns={' Reporting Area':'Reporting Area'}, inplace=True)
    if 'MMWRYear' in results_df.columns:
        results_df.rename(columns={'MMWRYear':'MMWR Year'}, inplace=True)
    if 'MMWRWeek' in results_df.columns:
        results_df.rename(columns={'MMWRWeek':'MMWR Week'}, inplace=True)

    results_df = results_df[results_df['Reporting Area'].isin(states.str.upper().tolist())]
    results_df = results_df.sort_values(['MMWR Year', 'MMWR Week', 'Reporting Area'], ascending=[False, True, True])
    maincolumns = results_df[['Reporting Area', 'MMWR Week', 'MMWR Year']]
    disease = results_df.loc[:, results_df.columns.str.contains('Current')]
    print(disease.head())
    if 'Reporting Area' in diseases.columns:
        diseases = pd.concat([diseases, disease], axis=1)
    else:
        diseases = pd.concat([maincolumns, disease], axis=1)

    print(len(diseases.columns))

diseases.to_csv('Notifiable Diseases.csv')
# ##Merge and Select Columns
# disease1 = disease[['Reporting Area','MMWR Week','Chlamydia trachomatis infection, Current week']]
# disease2 = disease[['Reporting Area','MMWR Week','Coccidioidomycosis, Current week']]

#output
# print(disease1['Chlamydia trachomatis infection, Current week'])

#visualization
# plt.bar( disease1['MMWR Week'], disease1['Chlamydia trachomatis infection, Current week'])
# weeks = max(disease1['MMWR Week'])
# weeks = weeks + 1
# weeks = list(range(1,weeks)) 

# plt.xlabel('Week')
# plt.ylabel('Cases')
# plt.xticks(weeks, weeks)
# plt.title('Chlamydia trachomatis infection')
# plt.show()
