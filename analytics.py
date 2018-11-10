#import quandl
import io
import math
import os
import pandas as pd
import numpy as np
import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
from matplotlib import style
from pandas.tools.plotting import scatter_matrix
# style.use('fivethirtyeight')


diseases = pd.read_csv('Notifiable Diseases.csv')
popdens = pd.read_csv('popdensity.csv')
popdens = popdens[['State','2018 Population','km²','Population_Density(km²)']]
diseases.rename(columns={'Reporting Area': 'State'}, inplace=True)
diseases['State'] = diseases['State'].str.title()
#state input
def get_state_input():
    temp = 1
    state = ''
    for x in diseases['State'].unique():
        print(str(temp) +': '+str(x))
        temp +=1
    print('Enter State Number: ')
    keyinp = input()
    temp = 1
    for x in diseases['State'].unique():
        if(str(temp) == str(keyinp)):
            state = x
        temp +=1
    print(state + ' selected')
    print('-------')
    return state

#disease input
def get_disease_input():
    temp = 1
    disease = ''
    cols = pd.Series(diseases.columns)
    columns = []
    for c in cols:
        if 'Current week' in c:
            columns.append(c)
        if 'Total' in c:
            columns.append(c)  
    for x in columns:
        dieasename = x.replace('Current week','')[:-1]
        if(dieasename[-1]==','):
            dieasename = dieasename[:-1]
        elif(dieasename[-2:]=='ta'):
            dieasename = 'All Diseases'
        print(str(temp) +': '+str(dieasename))
        temp +=1
    print('Enter Disease Number: ')
    keyinp = input()
    temp = 1
    for x in diseases[columns]:
        if(str(temp) == str(keyinp)):
            disease = x
        temp +=1
    return disease

# Cumulative Total of Cases per State
def cum_total():
    cols = pd.Series(diseases.columns)
    columns = []
    for c in cols:
        if 'Current'or 'Total' in c:
            columns.append(c)
    data = diseases.groupby(['State'],as_index=False)[diseases.columns[3:]].sum().drop(columns = ['MMWR Year'])
    print(data)
    x = np.arange(len(data['State']))
    fig, ax = plt.subplots()
    ax1 = ax.bar(data['State'], data['Total'], .9)
    ax.set_ylabel('Cases')
    ax.set_title('Total Cases of Notifiable diseases per state (week ' +str(diseases['MMWR Week'].max())+ ')')
    ax.set_xticks(x)
    ax.set_xticklabels(data['State'])

    plt.xticks(rotation='vertical')
    plt.subplots_adjust(bottom= 0.20)

    for i,j in zip(x,data['Total']):
            ax.annotate(str(int(j)),xy=(i-0.5,j), size=8)
    plt.show()

# Number of specific disease cases per week
def disease_per_week(disease):
    data = diseases.groupby(['MMWR Week'],as_index=False)[diseases.columns[3:]].sum().drop(columns = ['MMWR Year'])
    return data

# Number of cases per week
def cases_per_week():
    state = get_state_input()
    print(state)
    disease = get_disease_input()
    data = diseases.loc[diseases['State'].isin([state])].drop(columns = ['Unnamed: 0'])
    data2 = disease_per_week(disease)
    diseasename = disease.replace('Current week','')[:-1]
    if(diseasename[-1]==','):
        diseasename = diseasename[:-1]
    x = data['MMWR Week']
    y = data[disease]
    fig, ax = plt.subplots()
    ax1 = ax.bar(x,y, .9)
    ax2 = ax.bar(data2['MMWR Week'], data2[disease], 0.9, bottom =y)
    ax.set_ylabel('Cases')
    ax.set_xlabel('Week')
    ax.set_title('Cases of ' + diseasename + ' on '+state)
    if(diseasename == 'Tota'):
        ax.set_title('All Cases of Notifiable Diseases on '+state)
    ax.set_xticks(x)
    plt.legend((ax1[0], ax2[0]), (state, 'All States'),loc='upper right')
    plt.subplots_adjust(bottom= 0.10, left = 0.05, right = 0.95)
    for i,j in zip(x,y):
        ax.annotate(str(int(j)),xy=(i-0.5,j), size=8)
    plt.show()
#diseases_population denstity correlation
def diseases_popdens_cor():
    data = diseases.groupby(['State'],as_index=False)[diseases.columns[3:]].sum().drop(columns=['MMWR Year'])
    data = pd.merge(data, popdens, on='State')
    print(data.columns)
    cols = pd.Series(data.columns)
    # data.columns = range(data.shape[1])
    data = data.corr()
    print(data.head())
    columns = []
    for c in cols:
            if 'Current' in c:
                diseasename = c.replace('Current week','')[:-1]
                if(diseasename[-1]==','):
                    diseasename = diseasename[:-1]
                columns.append(diseasename)
            elif 'State' not in c:
                columns.append(c)


    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(data, vmin=-1, vmax=1, cmap='jet')
    fig.colorbar(cax)
    ticks = np.arange(0,len(columns),1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(columns)
    ax.set_yticklabels(columns)
    plt.xticks(rotation='vertical')
    plt.subplots_adjust(bottom= 0.04)
    plt.show()

cases_per_week()