import pandas as pd
import csv
import os

us_states = ["Alaska", "Alabama", "Arkansas", "Arizona", "California", "Colorado", "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Iowa", "Idaho", "Illinois", "Indiana", "Kansas", "Kentucky", "Louisiana", "Massachusetts", "Maryland", "Maine", "Michigan", "Minnesota", "Missouri", "Mississippi", "Montana", "North Carolina", "North Dakota", "Nebraska", "New Hampshire", "New Jersey", "New Mexico", "Nevada", "New York", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Virginia", "Vermont", "Washington", "Wisconsin", "West Virginia", "Wyoming"]
#Removed Rhode Island due to dtype problem, download again once noaa is alive and running , "Rhode Island"
dir_name = os.path.dirname(__file__)
for states in us_states:  
    file_name = dir_name + "/" + states + ".csv"   
    file_read = pd.read_csv(file_name, dtype={"DATE": object, "TMIN": float, "TMAX": float})
    avg_tmin = file_read.groupby('DATE')['TMIN'].mean() 
    avg_tmax = file_read.groupby('DATE')['TMAX'].mean()
    file_summarized = dir_name + "/" + states + "_summarized.csv"
    pd.concat([avg_tmin, avg_tmax], axis=1).to_csv(file_summarized)
