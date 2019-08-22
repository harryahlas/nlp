# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 20:35:54 2019

https://www.dataquest.io/blog/tutorial-text-classification-in-python-using-spacy/

@author: Anyone
"""

# Word tokenization
from spacy.lang.en import English

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = English()

text = """When learning data science, you shouldn't get discouraged!
Challenges and setbacks aren't failures, they're just part of the journey. You've got this!"""

#  "nlp" Object is used to create documents with linguistic annotations.
my_doc = nlp(text)

# Create list of word tokens
token_list = []
for token in my_doc:
    token_list.append(token.text)
print(token_list)



# sentence tokenization

# Create the pipeline 'sentencizer' component
sbd = nlp.create_pipe('sentencizer')

# Add the component to the pipeline
nlp.add_pipe(sbd)

#  "nlp" Object is used to create documents with linguistic annotations.
doc = nlp(text)

# create list of sentence tokens
sents_list = []
for sent in doc.sents:
    sents_list.append(sent.text)
print(sents_list)


# Removing stop words
import spacy
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS

#Printing the total number of stop words:
print('Number of stop words: %d' % len(spacy_stopwords))

#Printing first ten stop words:
print('First ten stop words: %s' % list(spacy_stopwords)[:20])

#from spacy.lang.en.stop_words import STOP_WORDS

#Implementation of stop words:
filtered_sent=[]

#  "nlp" Object is used to create documents with linguistic annotations.
doc = nlp(text)

# filtering stop words
for word in doc:
    if word.is_stop==False:
        filtered_sent.append(word)
print("Filtered Sentence:",filtered_sent)


# Implementing lemmatization
lem = nlp("run runs running runner")
# finding lemma for each word
for word in lem:
    print(word.text, word.lemma_)



# POS tagging

# importing the model en_core_web_sm of English for vocabluary, syntax & entities
import en_core_web_sm


# load en_core_web_sm of English for vocabluary, syntax & entities
nlp = en_core_web_sm.load()

#  "nlp" Objectis used to create documents with linguistic annotations.
doc = nlp(u"up yours valto")

for word in doc:
    print(word.text, word.pos_)
    
# Entity detection
#, also called entity recognition, is a more advanced form
# of language processing that identifies important elements
# This is really helpful for quickly 
# extracting information from text, since 
# you can quickly pick out important topics 
 
#for visualization of Entity detection importing displacy from spacy:
from spacy import displacy
nytimes = nlp(u"""New York City on Tuesday declared a public health emergency and ordered mandatory measles vaccinations amid an outbreak, becoming the latest national flash point over refusals to inoculate against dangerous diseases.
At least 285 people have contracted measles in the city since September, mostly in Brooklyn’s Williamsburg neighborhood. The order covers four Zip codes there, Mayor Bill de Blasio (D) said Tuesday.
The mandate orders all unvaccinated people in the area, including a concentration of Orthodox Jews, to receive inoculations, including for children as young as 6 months old. Anyone who resists could be fined up to $1,000.""")

# See entities
nytimes.ents

entities = [(i.label, i.label_, i.ents) for i in nytimes.ents]
entities 

# This is an issue. Probably works on Jupyter https://github.com/explosion/spaCy/commit/52658c80d5dcc35469d2006317190009e7f43763#diff-bc6983072591fe3fa303cc3a6eac64dc
displacy.render(nytimes, style = "ent", jupyter=None)

# Dependency Parsing
# Dependency parsing is a language processing technique that allows us to better determine the meaning of a sentence by analyzing how it’s constructed to determine how the individual words relate to each other.
docp = nlp (" In pursuit of a wall, President Trump ran into one.")

for noun in docp.noun_chunks:
    print(noun)

for chunk in docp.noun_chunks:
   print(chunk.text, chunk.root.text, chunk.root.dep_,
          chunk.root.head.text)

# Word vectors
import en_core_web_sm
nlp = en_core_web_sm.load()
mango = nlp(u"mango")
mango.vector.shape
print(mango.vector)

#next: start with text classification here:
#    https://www.dataquest.io/blog/tutorial-text-classification-in-python-using-spacy/