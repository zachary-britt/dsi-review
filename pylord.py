from pymongo import MongoClient
import urllib.parse
from bs4 import BeautifulSoup
import requests

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from nltk.stem import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import re
import random

def mongo_client():
    username = urllib.parse.quote_plus('admin')
    password = urllib.parse.quote_plus('admin123')
    client = MongoClient('mongodb://%s:%s@127.0.0.1' % (username, password))
    return client

def mongo_table(client):
    db = client['a_database']
    table = db['a_table']
    return table
    # tab.insert_one({'item':'Testing', 'test_count':5})

def get_nyt_corpus():
    client = mongo_client()
    db = client.nyt_dump
    coll = db.articles
    corpus = []
    lables = []
    for doc in coll.find():
        labels.append(doc['news_desk'])
        corpus.append(doc['content'])
    return corpus, labels


def number_of_jobs(query):
    '''
    INPUT: string
    OUTPUT: int

    Return the number of jobs on the indeed.com for the search query.
    '''

    url = "http://www.indeed.com/jobs?q=%s" % query.replace(' ', '+')
    html = requests.get(url).text
    soup = BeautifulSoup(html, 'html.parser')
    search_count = soup.find('div', id='searchCount')
    return int(search_count.text.split('of ')[-1].replace(',', ''))


def _html_parser(html):
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.get_text().replace('\n', ' ')
    return text


class TFIDF:
    def __init__(self):

        self._vectorizer = TfidfVectorizer(stop_words='english', norm='l1')
        self._lemmer = WordNetLemmatizer()

    def fit_transform(self, X):
        X_l = np.array([self._lemmer.lemmatize(doc) for doc in  X])
        X_vec = self._vectorizer.fit_transform(X_l)
        return X_vec

    def transform(self, X):
        X_l = np.array([self._lemmer.lemmatize(doc) for doc in  X])
        X_vec = self._vectorizer.transform(X_l)
        return X_vec

def slns():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel
    from nltk.stem.snowball import SnowballStemmer
    from nltk import word_tokenize
    from string import punctuation

    documents = ['Dogs like dogs more than cats.',
                 'The dog chased the bicycle.',
                 'The cat rode in the bicycle basket.',
                 'I have a fast bicycle.']

    sbs = SnowballStemmer('english')
    punctuation = set(punctuation)
    def my_tokenizer(text):
        return [sbs.stem(token) for token in word_tokenize(text)
                if token not in punctuation]

    vectorizer = TfidfVectorizer(tokenizer=my_tokenizer, stop_words='english')
    tfidf_docs = vectorizer.fit_transform(documents)
    cos_sims = linear_kernel(tfidf_docs, tfidf_docs)
    print(cos_sims)





if __name__ == '__main__':
    # client = mongo_client()
    # table = mongo_table(client)

    corpus, labels= get_nyt_corpus()
    tfidf = TFIDF()
    X_vec = tfidf.fit_transform(corpus)





    #
