# coding=utf-8

import pandas as pd
import itertools
import unicodecsv
import os
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

training_set = os.path.abspath("./small-set-3.csv")
testing_set = os.path.abspath("../DCM/testing.csv")

training_df = pd.read_csv(training_set, sep=",", names=["category", "text"])

vect = CountVectorizer()
X = vect.fit_transform(training_df['text'])
Y = training_df['category']

""" Faster Method """
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X)
X_train_tfidf.shape

mnb_classifier = MultinomialNB().fit(X_train_tfidf, Y)

testing_df = pd.read_csv(testing_set, usecols=[3], encoding="ISO-8859-1").values.tolist()
new_docs = list(itertools.chain(*testing_df))
new_docs = [f.encode('UTF8') for f in new_docs]

tweetId_df = pd.read_csv(testing_set, usecols=[1], encoding="ISO-8859-1").values.tolist()
tweetId = list(itertools.chain(*tweetId_df))
tweetId = [f.encode('UTF8') for f in tweetId]

timestamp_df = pd.read_csv(testing_set, usecols=[6], encoding="ISO-8859-1").values.tolist()
timestamp = list(itertools.chain(*timestamp_df))
timestamp = [f.encode('UTF8') for f in timestamp]

X_test_counts = vect.transform(new_docs)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

predicted = mnb_classifier.predict(X_test_tfidf)

output = []
for id, doc, cat, ts in zip(tweetId, new_docs, predicted, timestamp):
    a = id, doc, cat, ts
    output.append(a)

with open('mb-output.csv', 'wb') as output_file:
     wr = unicodecsv.writer(output_file, encoding='utf-8', delimiter=',', quoting=unicodecsv.QUOTE_ALL)
     wr.writerows(output)
     print '[log] Output file saved'