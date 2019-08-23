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


# Text Classification
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
# Data:
#https://www.kaggle.com/sid321axn/amazon-alexa-reviews/home
df_amazon = pd.read_csv("datasets/amazon_alexa.tsv", sep = "\t")
df_amazon.head()
df_amazon.feedback.value_counts()

import string
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English

# Create our list of punctuation marks
punctuations = string.punctuation

#nlp = spacy.load('en')
nlp = spacy.load('en_core_web_sm') # can use this instead of above
stop_words = spacy.lang.en.stop_words.STOP_WORDS

# Load English tokenizer, tagger, parser, NER and word vectors
parser = English()

# Tokenize, strip punctuation and stop words, lemmatize, lower case

# Creating our tokenizer function
def spacy_tokenizer(sentence):
    # Creating our token object, which is used to create documents with linguistic annotations.
    mytokens = parser(sentence)

    # Lemmatizing each token and converting each token into lowercase
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

    # Removing stop words
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]

    # return preprocessed list of tokens
    return mytokens

# To further clean our text data, we’ll also want to create a custom transformer for removing initial and end spaces and converting text into lower case
# Custom transformer using spaCy
class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        # Cleaning Text
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

# Basic function to clean the text
def clean_text(text):
    # Removing spaces and converting text into lowercase
    return text.strip().lower()

# Vectorization Feature Engineering (TF-IDF)
# Set ngrams for tokenizer
bow_vector = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))
#TF-IDF
tfidf_vector = TfidfVectorizer(tokenizer = spacy_tokenizer)

# Split into train and test
from sklearn.model_selection import train_test_split

X = df_amazon['verified_reviews'] # the features we want to analyze
ylabels = df_amazon['feedback'] # the labels, or answers, we want to test against

X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.3)

# Creating a Pipeline and Generating the Model
# Logistic Regression Classifier
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()

# Create pipeline using Bag of Words
pipe = Pipeline([("cleaner", predictors()),
                 ('vectorizer', bow_vector),
                 ('classifier', classifier)])

# model generation
pipe.fit(X_train,y_train)

# Evaluate
from sklearn import metrics
# Predicting with a test dataset
predicted = pipe.predict(X_test)

# Model Accuracy
print("Logistic Regression Accuracy:",metrics.accuracy_score(y_test, predicted))
print("Logistic Regression Precision:",metrics.precision_score(y_test, predicted))
print("Logistic Regression Recall:",metrics.recall_score(y_test, predicted))

