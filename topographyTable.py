# To comment a block in Visual Studio Code, highlight then Ctr+K and Ctrl+C
# To uncomment a block, highlight then Ctrl+K and Ctrl+U
import re
import os
import csv
from selenium import webdriver
from bs4 import BeautifulSoup

page = "https://www.anyplaceamerica.com"
driver = webdriver.Chrome()
driver.get(page)
soup = BeautifulSoup(driver.page_source, "html.parser")
a_class = soup.select('a[class*="black-gradient"]')
us_states = ["Alaska", "Alabama", "Arkansas", "Arizona", "California", "Colorado", "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Iowa", "Idaho", "Illinois", "Indiana", "Kansas", "Kentucky", "Louisiana", "Massachusetts", "Maryland", "Maine", "Michigan", "Minnesota", "Missouri", "Mississippi", "Montana", "North Carolina", "North Dakota", "Nebraska", "New Hampshire", "New Jersey", "New Mexico", "Nevada", "New York", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Virginia", "Vermont", "Washington", "Wisconsin", "West Virginia", "Wyoming"]
#states = input("Which state do you want to check? ")
row = []
dir_name = os.path.dirname(__file__)
topo_file = dir_name + "/" + "Number of Certain Topo Char Per State.csv"
inp = "Yes"
states = ""
glaciers = locales = beach = areas = lakes = streams = swamps = forests = plains = woods = 0

def toDigit(char_string, digit_out = None):
    digit_out = re.sub("[^0-9]", '', char_string)
    return digit_out

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
                for li_links in ul_class.find_all("li"):
                    a_text = li_links.find("a")
                    if a_text.text == states:
                        span_txt = li_links.find("span")
                        span_char = toDigit(span_txt.text)
                        glaciers = int(span_char)
            if name_class.text == "Locales":
                driver.get(page + a_links["href"])
                soup = BeautifulSoup(driver.page_source, "html.parser")
                ul_class = soup.find(class_="state-list")
                for li_links in ul_class.find_all("li"):
                    a_text = li_links.find("a")
                    if a_text.text == states:
                        span_txt = li_links.find("span")
                        span_char = toDigit(span_txt.text)
                        locales = int(span_char)
            if name_class.text == "Beaches":
                driver.get(page + a_links["href"])
                soup = BeautifulSoup(driver.page_source, "html.parser")
                ul_class = soup.find(class_="state-list")
                for li_links in ul_class.find_all("li"):
                    a_text = li_links.find("a")
                    if a_text.text == states:
                        span_txt = li_links.find("span")
                        span_char = toDigit(span_txt.text)
                        beach = int(span_char)
            if name_class.text == "Areas":
                driver.get(page + a_links["href"])
                soup = BeautifulSoup(driver.page_source, "html.parser")
                ul_class = soup.find(class_="state-list")
                for li_links in ul_class.find_all("li"):
                    a_text = li_links.find("a")
                    if a_text.text == states:
                        span_txt = li_links.find("span")
                        span_char = toDigit(span_txt.text)
                        areas = int(span_char)
            if name_class.text == "Lakes":
                driver.get(page + a_links["href"])
                soup = BeautifulSoup(driver.page_source, "html.parser")
                ul_class = soup.find(class_="state-list")
                for li_links in ul_class.find_all("li"):
                    a_text = li_links.find("a")
                    if a_text.text == states:
                        span_txt = li_links.find("span")
                        span_char = toDigit(span_txt.text)
                        lakes = int(span_char)
            if name_class.text == "Streams":
                driver.get(page + a_links["href"])
                soup = BeautifulSoup(driver.page_source, "html.parser")
                ul_class = soup.find(class_="state-list")
                for li_links in ul_class.find_all("li"):
                    a_text = li_links.find("a")
                    if a_text.text == states:
                        span_txt = li_links.find("span")
                        span_char = toDigit(span_txt.text)
                        streams = int(span_char)
            if name_class.text == "Swamps":
                driver.get(page + a_links["href"])
                soup = BeautifulSoup(driver.page_source, "html.parser")
                ul_class = soup.find(class_="state-list")
                for li_links in ul_class.find_all("li"):
                    a_text = li_links.find("a")
                    if a_text.text == states:
                        span_txt = li_links.find("span")
                        span_char = toDigit(span_txt.text)
                        swamps = int(span_char)
            if name_class.text == "Forests":
                driver.get(page + a_links["href"])
                soup = BeautifulSoup(driver.page_source, "html.parser")
                ul_class = soup.find(class_="state-list")
                for li_links in ul_class.find_all("li"):
                    a_text = li_links.find("a")
                    if a_text.text == states:
                        span_txt = li_links.find("span")
                        span_char = toDigit(span_txt.text)
                        forests = int(span_char)
            if name_class.text == "Plains":
                driver.get(page + a_links["href"])
                soup = BeautifulSoup(driver.page_source, "html.parser")
                ul_class = soup.find(class_="state-list")
                for li_links in ul_class.find_all("li"):
                    a_text = li_links.find("a")
                    if a_text.text == states:
                        span_txt = li_links.find("span")
                        span_char = toDigit(span_txt.text)
                        plains = int(span_char)
            if name_class.text == "Woods":
                driver.get(page + a_links["href"])
                soup = BeautifulSoup(driver.page_source, "html.parser")
                ul_class = soup.find(class_="state-list")
                for li_links in ul_class.find_all("li"):
                    a_text = li_links.find("a")
                    if a_text.text == states:
                        span_txt = li_links.find("span")
                        span_char = toDigit(span_txt.text)
                        woods = int(span_char)

        row = [states, glaciers, locales, beach, areas, lakes, streams, swamps, forests, plains, woods]
        writer.writerow(row)
        glaciers = locales = beach = areas = lakes = streams = swamps = forests = plains = woods = 0

    driver.close()
    temp_file.close()

try:
    if os.path.getsize(topo_file) > 0:
        print("Number of Certain Topo Char Per State.csv")
    else: 
        topography_table()
except OSError as e:
    topography_table()