#!/usr/bin/env python
# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from random import randint
import datetime
import os
import time
import pandas as pd



url = "https://www.amazon.co.uk/gp/deals/ref=sv_uk_0"

chromedriver = "/Applications/chromedriver"
os.environ["webdriver.chrome.driver"] = chromedriver
driver = webdriver.Chrome(chromedriver)
time_delay = randint(3,6)


driver.get(url)

time.sleep(time_delay)
# start_date = current_time + datetime.timedelta(days=week*7) + datetime.timedelta(days=weeks_from_now_to_look*7)
# end_date = start_date + datetime.timedelta(days=nights_stay)
selectElem=driver.find_element_by_xpath('//*[@id="nav-link-shopall"]/span[2]/span')
selectElem.clear()
site_data = driver.find_elements_by_class_name('nav-line-2')
time.sleep(time_delay)
print(site_data)


#     property_data = []
# for i in site_data:
#         if len(i.text) != 0:
#              property_data.append(i.text)
#
#     camping_availability_dictionary[start_date.strftime("%a %b %d %Y") + ' to ' + end_date.strftime("%a %b %d %Y")] = property_data
#
#     time.sleep(time_delay)
#
#
#
# import requests
# response = requests.get(url)
# page = response.text
# soup = BeautifulSoup(page,"lxml")
# print(soup.prettify())
# campsite_info=(soup.find("title"))
#  campsite_info.get_text()
#
# print(soup.find_all(class_='contable'))
# table_body = soup.find_all(class='contable')
#
#
# rows = table_body[0].find_all('tr')
# for row in rows:
#     columns = row.find_all('td')
#     for column in columns:
#         string = column.get_text()
#         if string == 'Within Facility':
#             print("Items Within Facility: \n")
#         if string == 'Within 10 Miles':
#             print("\nItems Within 10 Miles: \n")
#         else:
#             items = column.find_all('li')
#             for item in items:
#                 words = item.get_text()
#                 print(words)
