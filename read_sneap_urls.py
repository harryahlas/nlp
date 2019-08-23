# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 22:25:07 2019

@author: Anyone
"""

import requests
import re

link = "http://www.ultimatemetal.com/forum/threads/tube-screamer-faq-version-1.346068/page-2"
link = "http://www.ultimatemetal.com/forum/forums/andy-sneap-backline/page-101"
f = requests.get(link)
#print(f.text)

filehandle = open('x.html', 'w', encoding='utf-8')
filehandle.write(f.text)
filehandle.close()


# Function to search for text between 2 markers
def find_text(text, start_marker, end_marker):
    result = re.search(start_marker + "(.*)" + end_marker, text)
    return result.group(1)


# section_name = Backline, section_number = 1
section_name = Backline
section_number = 1


# Look up value for last page in data-last=
data_last_page = int(find_text(f.text,'data-last="', '"' ))



from bs4 import BeautifulSoup
soup = BeautifulSoup(f.text, "html.parser")
headers = soup.select('h3.title')
headers[0].text.strip()


##  This will become loop

# for i in 1:data_last_page:

    #USE BELOW IF PAGE 1 DOESNT WORK
#if i == 1 then url_lookup = "http://www.ultimatemetal.com/forum/forums/andy-sneap-backline/"
#else 

url_lookup = "http://www.ultimatemetal.com/forum/forums/andy-sneap-backline/page-" + str(i)



# Look up titles/links on current page
# Start Marker for link
# End marker for link
# Retrieve link

# Add link, section name (eg Backline), and number (eg 1) to url_list


### Retrieve initial post for each item in url_list
# Pull page (initial page only for now)
# Look up initial text start marker
# Look up initial text end marker
# add text to data frame

### Run spacy_text_classification to predict which forum.
