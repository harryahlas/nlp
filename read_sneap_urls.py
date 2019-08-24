# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 22:25:07 2019

@author: Anyone
"""

import requests
import re
import pandas as pd
from bs4 import BeautifulSoup

#print(f.text)

filehandle = open('x.html', 'w', encoding='utf-8')
filehandle.write(f.text)
filehandle.close()


# Function to search for text between 2 markers
def find_text(text, start_marker, end_marker):
    result = re.search(start_marker + "(.*)" + end_marker, text)
    return result.group(1)



# section_name = Backline, section_number = 1
section_name = 'Backline'
section_number = 1

section_link = "http://www.ultimatemetal.com/forum/forums/andy-sneap-backline/"
url_prefix = "http://www.ultimatemetal.com/forum/forums/andy-sneap-backline/"

def get_pages(section_name, section_number, url_prefix):

    # List of pages to be pulled
    df_pages = pd.DataFrame(None, columns = ['section_name', 
                                             'section_number', 
                                             'url', 
                                             'thread_title'])
    # Retrieve section links
    f = requests.get(url_prefix)

    # Look up value for last page in data-last=
    data_last_page = int(find_text(f.text,'data-last="', '"' ))
    
    # Threads prefix
    threads_prefix = "http://www.ultimatemetal.com/forum/"
    
    for i in range(1,5 + 1):
        print("loop" + str(i))
    
        link = url_prefix + "page-" + str(i)
        f = requests.get(link)
        
        soup = BeautifulSoup(f.text, "html.parser")
        headers = soup.select('h3.title')
        
        for j in range(len(headers)):
            print("going through headers " + str(j))
            
            # Get end of each thread link
            url_suffix = find_text(str(headers[j]), 'href="', '/"')
            url = threads_prefix + url_suffix 
            thread_title = headers[j].text.strip()
            
            # Get page to append to data set
            df_pages_temp = pd.DataFrame({"section_name": [section_name],
                                          "section_number": [section_number],
                                          "url": [url],
                                          "thread_title": [thread_title]})
                                          
            df_pages = df_pages.append(df_pages_temp)
            
    return df_pages
    
df_pages_main = get_pages('Main', 
                              0,
                              "http://www.ultimatemetal.com/forum/forums/andy-sneap/")

df_pages_backstage = get_pages('Backstage', 
                              1,
                              "http://www.ultimatemetal.com/forum/forums/andy-sneap-backstage/")

df_pages_foh = get_pages('FOH', 
                              2,
                              "http://www.ultimatemetal.com/forum/forums/andy-sneap-foh/")

df_pages_practice_room = get_pages('Practice_Room', 
                              3,
                              "http://www.ultimatemetal.com/forum/forums/andy-sneap-practice-room/")

df_pages_backline = get_pages('Backline', 
                              4,
                              "http://www.ultimatemetal.com/forum/forums/andy-sneap-backline/")

df_pages_merch_stand = get_pages('Merch_Stand', 
                              5,
                              "http://www.ultimatemetal.com/forum/forums/andy-sneap-merch-stand/")

df_pages_bar = get_pages('Bar', 
                              6,
                              "http://www.ultimatemetal.com/forum/forums/andy-sneap-bar/")


df_pages_all = pd.concat([df_pages_main,
                          df_pages_backstage,
                          df_pages_foh,
                          df_pages_practice_room,
                          df_pages_backline,
                          df_pages_merch_stand,
                          df_pages_bar], ignore_index=True)

df_pages_all['html'] = None
    
for idx, url in enumerate(df_pages_all.url[0:10]):
    print(idx, url)
    f = requests.get(url)
    df_pages_all.html[idx] = f.text
    
df_pages_all.html[3]

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
