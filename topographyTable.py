# To comment a block in Visual Studio Code, highlight then Ctr+K and Ctrl+C
# To uncomment a block, highlight then Ctrl+K and Ctrl+U
import os
import csv
from selenium import webdriver
from bs4 import BeautifulSoup

page = "https://www.anyplaceamerica.com"
driver = webdriver.Chrome()
driver.get(page)
soup = BeautifulSoup(driver.page_source, "html.parser")
a_class = soup.select('a[class*="black-gradient"]')
us_states = ["Alaska", "Alabama", "Arkansas", "Arizona", "California", "Colorado", "Connecticut", "District of Columbia", "Delaware", "Florida", "Georgia", "Hawaii", "Iowa", "Idaho", "Illinois", "Indiana", "Kansas", "Kentucky", "Louisiana", "Massachusetts", "Maryland", "Maine", "Michigan", "Minnesota", "Missouri", "Mississippi", "Montana", "North Carolina", "North Dakota", "Nebraska", "New Hampshire", "New Jersey", "New Mexico", "Nevada", "New York", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Virginia", "Vermont", "Washington", "Wisconsin", "West Virginia", "Wyoming", "United States Naval Base Guantanamo Bay"]
#states = input("Which state do you want to check? ")
row = []
dir_name = os.path.dirname(__file__)
topo_file = dir_name + "/" + "Topography.csv"
inp = "Yes"
states = ""
glaciers = locales = beach = areas = lakes = streams = swamps = forests = plains = woods = 0

def topography_table(): #to make the code output just a state, comment out states and add an argument which accepts the state then change the loop argument
    temp_file = open(topo_file,"w",newline='')
    writer = csv.writer(temp_file)
    row = ["State", "Glaciers", "Locales", "Beaches", "Areas", "Lakes", "Streams", "Swamps", "Forests", "Plains", "Woods"]
    writer.writerow(row)

    for states in us_states:
        for a_links in a_class:
            name_class = a_links.find(class_="name")
            if name_class.text == "Glaciers":
                driver.get(page + a_links["href"])
                soup = BeautifulSoup(driver.page_source, "html.parser")
                ul_class = soup.find(class_="state-list")
                for a_text in ul_class.find_all("a"):
                    if a_text.text == states:
                        glaciers = 1
            if name_class.text == "Locales":
                driver.get(page + a_links["href"])
                soup = BeautifulSoup(driver.page_source, "html.parser")
                ul_class = soup.find(class_="state-list")
                for a_text in ul_class.find_all("a"):
                    if a_text.text == states:
                        locales = 1
            if name_class.text == "Beaches":
                driver.get(page + a_links["href"])
                soup = BeautifulSoup(driver.page_source, "html.parser")
                ul_class = soup.find(class_="state-list")
                for a_text in ul_class.find_all("a"):
                    if a_text.text == states:
                        beach = 1
            if name_class.text == "Areas":
                driver.get(page + a_links["href"])
                soup = BeautifulSoup(driver.page_source, "html.parser")
                ul_class = soup.find(class_="state-list")
                for a_text in ul_class.find_all("a"):
                    if a_text.text == states:
                        areas = 1
            if name_class.text == "Lakes":
                driver.get(page + a_links["href"])
                soup = BeautifulSoup(driver.page_source, "html.parser")
                ul_class = soup.find(class_="state-list")
                for a_text in ul_class.find_all("a"):
                    if a_text.text == states:
                        lakes = 1
            if name_class.text == "Streams":
                driver.get(page + a_links["href"])
                soup = BeautifulSoup(driver.page_source, "html.parser")
                ul_class = soup.find(class_="state-list")
                for a_text in ul_class.find_all("a"):
                    if a_text.text == states:
                        streams = 1
            if name_class.text == "Swamps":
                driver.get(page + a_links["href"])
                soup = BeautifulSoup(driver.page_source, "html.parser")
                ul_class = soup.find(class_="state-list")
                for a_text in ul_class.find_all("a"):
                    if a_text.text == states:
                        swamps = 1
            if name_class.text == "Forests":
                driver.get(page + a_links["href"])
                soup = BeautifulSoup(driver.page_source, "html.parser")
                ul_class = soup.find(class_="state-list")
                for a_text in ul_class.find_all("a"):
                    if a_text.text == states:
                        forests = 1
            if name_class.text == "Plains":
                driver.get(page + a_links["href"])
                soup = BeautifulSoup(driver.page_source, "html.parser")
                ul_class = soup.find(class_="state-list")
                for a_text in ul_class.find_all("a"):
                    if a_text.text == states:
                        plains = 1
            if name_class.text == "Woods":
                driver.get(page + a_links["href"])
                soup = BeautifulSoup(driver.page_source, "html.parser")
                ul_class = soup.find(class_="state-list")
                for a_text in ul_class.find_all("a"):
                    if a_text.text == states:
                        woods = 1

        row = [states, glaciers, locales, beach, areas, lakes, streams, swamps, forests, plains, woods]
        writer.writerow(row)
        glaciers = locales = beach = areas = lakes = streams = swamps = forests = plains = woods = 0

    driver.close()
    temp_file.close()

try:
    if os.path.getsize(topo_file) > 0:
        print("Topography.csv already exists")
    else: 
        topography_table()
except OSError as e:
    topography_table()