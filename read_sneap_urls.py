# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 22:25:07 2019

@author: Anyone
"""

import requests

link = "http://www.ultimatemetal.com/forum/threads/tube-screamer-faq-version-1.346068/page-2"
link = "http://www.ultimatemetal.com/forum/forums/andy-sneap-backline/page-101"
f = requests.get(link)
#print(f.text)

filehandle = open('x.html', 'w', encoding='utf-8')
filehandle.write(f.text)
filehandle.close()


# section_name = Backline, section_number = 1

# Look up value for last page in data-last=
##  This will become loop

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
