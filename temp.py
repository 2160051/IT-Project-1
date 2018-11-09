import os
import time
import csv
from selenium import webdriver
from bs4 import BeautifulSoup

driver = webdriver.Chrome()
driver.get("https://www.accuweather.com/en/browse-locations/nam/us")
soup = BeautifulSoup(driver.page_source, "html.parser")
li_class = soup.select('li[class*="drilldown cl"]')
states = []
state_links_extracted = []
#month_links_extracted = []
state_ctr = 0
dir_name = os.path.dirname(__file__)

for li_tag in li_class:
    for a_links in li_tag.find_all("a"):
        state_links_extracted.append(a_links["href"])
        states.append(a_links.text)
        folder_path = dir_name + "/" + a_links.text 
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

state_length = len(states)

for state_ctr in range(state_length):
    driver.get(state_links_extracted[state_ctr])
    soup = BeautifulSoup(driver.page_source, "html.parser")
    li_class = soup.select('li[class*="drilldown cl"]')
    for li in li_class:
        for a_cities in li.find_all("a"):
            city_temp_file = dir_name + "/" + states[state_ctr] + "/" + a_cities.text + ".csv"
            try:
                if os.path.getsize(city_temp_file) > 0:
                    print(a_cities.text + ".csv already exists")
                else:
                    temp_file = open(city_temp_file,"w",newline='')
                    writer = csv.writer(temp_file)
                    driver.get(a_cities["href"])
                    soup = BeautifulSoup(driver.page_source, "html.parser")
                    ul_class = soup.find_all('ul', class_='subnav-tab-buttons')
                    for ul in ul_class:
                        for a_month in ul.find_all("a"):
                            if(a_month.text == "Month"):  
                                driver.get(a_month["href"])
                                soup = BeautifulSoup(driver.page_source, "html.parser")
                                ul_tag = soup.find_all('ul', class_='g')
                #Sets the view to list i.e. table
                                for ul_cl in ul_tag:
                                    for a_view in ul_cl.find_all("a"):
                                        if(a_view.text == "As List"):
                                            month_today = a_view["href"]
                                            driver.get(a_view["href"])
                                            soup = BeautifulSoup(driver.page_source, "html.parser")
                #Goes thru tables for months Jan 2018 to Jun 2018
                                            for month_counter in range(0, 7):
                                                if time.strftime("%m")[:1] == "1":
                                                    for_replacement = "monyr=" + time.strftime("%m") + "/1/" + time.strftime("%Y")
                                                else:
                                                    for_replacement = "monyr=" + time.strftime("%m").replace("0", "") + "/1/" + time.strftime("%Y")
                                                if month_counter == 0:
                                                    month_replaced = month_today.replace(for_replacement, "monyr=12/1/2017")
                                                    temp_file.write("December 2017 \n")
                                                else :
                                                    month_replaced = month_today.replace(for_replacement, "monyr=" + str(month_counter) + "/1/2018")
                                                    if(month_counter == 1):
                                                        temp_file.write("Jan 2018\n")
                                                    elif(month_counter == 2):
                                                        temp_file.write("Feb 2018\n")
                                                    elif(month_counter == 3):
                                                        temp_file.write("Mar 2018\n")
                                                    elif(month_counter == 4):
                                                        temp_file.write("Apr 2018\n")
                                                    elif(month_counter == 5):
                                                        temp_file.write("May 2018\n")
                                                    elif(month_counter == 6):
                                                        temp_file.write("Jun 2018\n")
                                                driver.get(month_replaced)
                                                soup = BeautifulSoup(driver.page_source, "html.parser")
                                                #put here the reader for table, store and export to csv if possible
                                                table = soup.find('table', class_="calendar-list")
                                                rows = []
                                                for row in table.findAll('tr'):
                                                    cells_arr = []
                                                    for cell in row.findAll(["th","td"]):
                                                        text = cell.text
                                                        cells_arr.append(text)
                                                    rows.append(cells_arr)

                                                for item in rows:
                                                    writer.writerow(item)
                        temp_file.close()
            except OSError as e:
                temp_file = open(city_temp_file,"w",newline='')
                writer = csv.writer(temp_file)
                driver.get(a_cities["href"])
                soup = BeautifulSoup(driver.page_source, "html.parser")
                ul_class = soup.find_all('ul', class_='subnav-tab-buttons')
                for ul in ul_class:
                    for a_month in ul.find_all("a"):
                        if(a_month.text == "Month"):  
                            driver.get(a_month["href"])
                            soup = BeautifulSoup(driver.page_source, "html.parser")
                            ul_tag = soup.find_all('ul', class_='g')
            #Sets the view to list i.e. table
                            for ul_cl in ul_tag:
                                for a_view in ul_cl.find_all("a"):
                                    if(a_view.text == "As List"):
                                        month_today = a_view["href"]
                                        driver.get(a_view["href"])
                                        soup = BeautifulSoup(driver.page_source, "html.parser")
            #Goes thru tables for months Jan 2018 to Jun 2018
                                        for month_counter in range(0, 7):
                                            if time.strftime("%m")[:1] == "1":
                                                for_replacement = "monyr=" + time.strftime("%m") + "/1/" + time.strftime("%Y")
                                            else:
                                                for_replacement = "monyr=" + time.strftime("%m").replace("0", "") + "/1/" + time.strftime("%Y")
                                            if month_counter == 0:
                                                month_replaced = month_today.replace(for_replacement, "monyr=12/1/2017")
                                                temp_file.write("December 2017 \n")
                                            else :
                                                month_replaced = month_today.replace(for_replacement, "monyr=" + str(month_counter) + "/1/2018")
                                                if(month_counter == 1):
                                                    temp_file.write("Jan 2018\n")
                                                elif(month_counter == 2):
                                                    temp_file.write("Feb 2018\n")
                                                elif(month_counter == 3):
                                                    temp_file.write("Mar 2018\n")
                                                elif(month_counter == 4):
                                                    temp_file.write("Apr 2018\n")
                                                elif(month_counter == 5):
                                                    temp_file.write("May 2018\n")
                                                elif(month_counter == 6):
                                                    temp_file.write("Jun 2018\n")
                                            driver.get(month_replaced)
                                            soup = BeautifulSoup(driver.page_source, "html.parser")
                                            #put here the reader for table, store and export to csv if possible
                                            table = soup.find('table', class_="calendar-list")
                                            rows = []
                                            for row in table.findAll('tr'):
                                                cells_arr = []
                                                for cell in row.findAll(["th","td"]):
                                                    text = cell.text
                                                    cells_arr.append(text)
                                                rows.append(cells_arr)

                                            for item in rows:
                                                writer.writerow(item)
                    temp_file.close()

driver.close()