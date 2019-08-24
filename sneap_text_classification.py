# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 20:35:54 2019

https://www.dataquest.io/blog/tutorial-text-classification-in-python-using-spacy/

@author: Anyone
"""

import spacy
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
import string
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English

# import data from read_sneap_urls.py
df_pages_all = pd.read_pickle("datasets/df_pages_all_backup.pkl")
df_pages_all['section_number'] = np.int64(df_pages_all['section_number'])
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

X = df_pages_all['initial_message_text'] # the features we want to analyze
ylabels = df_pages_all['section_number'] # the labels, or answers, we want to test against

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
print("Logistic Regression Precision:",metrics.precision_score(y_test, predicted, average=None))
print("Logistic Regression Recall:",metrics.recall_score(y_test, predicted, average=None))

# Sample
sample_message_text = "Andy sneap is ina a new interview with Eyal Levi...what do you think? Check it out"
pipe.predict(pd.core.series.Series(sample_message_text ))
