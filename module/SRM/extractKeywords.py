# coding=utf-8
# Source: Used the following tutorial to build this
# http://joshbohde.com/blog/document-summarization

from nltk.tokenize.punkt import PunktSentenceTokenizer
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import simplejson
import networkx as nx

def rank_sentence(document):
    sentence_tokenizer = PunktSentenceTokenizer()
    sentences = sentence_tokenizer.tokenize(document)

    matrix = CountVectorizer().fit_transform(sentences)
    normalized = TfidfTransformer().fit_transform(matrix)

    sim_graph = normalized * normalized.T

    nx_graph = nx.from_scipy_sparse_matrix(sim_graph)
    scores = nx.pagerank(nx_graph)
    return sorted(((scores[i],s) for i,s in enumerate(sentences)),
                  reverse=True)


with open("../../dataset/skytrax-sia.json") as json_file:
    json_data = simplejson.load(json_file)
    res = json_data["results"]

    for value in res:
        # print value
        a = value["review_text"]
        print rank_sentence(a)

