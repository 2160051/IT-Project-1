##import quandl
import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
##style.use('fivethirtyeight')
style.use('ggplot')

##Gather Data
def get_population():
    res = requests.get("http://worldpopulationreview.com/states/")
    soup = BeautifulSoup(res.content,'lxml')
    table = soup.find_all('table')[0] 
    state_pop = pd.read_html(str(table))
    state_pop[0][['State','2018 Population']].sort_values(by=['State']).reset_index(drop = True).to_pickle('state_pop.pickle')

def get_land_areas():
    state_areas = pd.read_html('https://simple.wikipedia.org/wiki/List_of_U.S._states_by_area', header=0)
    state_land_areas = state_areas[1].sort_values(by=['State'])
    state_land_areas[['State','km²','sq.miles']].reset_index(drop = True).to_pickle('state_land_areas.pickle')

##Access Data
#land_areas
if os.path.isfile('state_land_areas.pickle'):
    state_land_areas = pd.read_pickle('state_land_areas.pickle')
else:
    get_land_areas()
    state_land_areas = pd.read_pickle('state_land_areas.pickle')
#population
if os.path.isfile('state_pop.pickle'):
    state_pop = pd.read_pickle('state_pop.pickle')
else:
    get_population()
    state_pop = pd.read_pickle('state_pop.pickle')

##Merge and Select Columns
merged = pd.merge(state_pop,state_land_areas, on=['State'])
pop_density = merged[['State','2018 Population','sq.miles','km²']]

##Add Population Density Columns
pop_density['Population_Density(km²)'] = pop_density['2018 Population']/merged['km²']
pop_density['Population_Density(sq.miles)'] = pop_density['2018 Population']/merged['sq.miles']
#Columns:State,2018 Population,sq.miles,km²,Population_Density(km²),Population_Density(sq.miles)

#output
print(pop_density)

#visualization
y_pos = np.arange(len(pop_density['State']))

plt.barh(y_pos, pop_density['Population_Density(km²)'])
plt.yticks(y_pos, pop_density['State'])

plt.xlabel('Population_Density(km²)')
plt.ylabel('States')

plt.show()