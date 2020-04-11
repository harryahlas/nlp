# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 20:34:36 2020
https://medium.com/mlreview/topic-modeling-with-scikit-learn-e80d33668730
@author: Harry Ahlas
"""

from sklearn.datasets import fetch_20newsgroups

dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
documents = dataset.data

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

no_features = 1000 #Number of words



# NMF is able to use tf-idf
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(documents)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()

# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform(documents)
tf_feature_names = tf_vectorizer.get_feature_names()

