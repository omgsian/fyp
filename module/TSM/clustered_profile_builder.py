# coding=utf-8

from itertools import chain
import re
import csv
import sys

from nltk.corpus import stopwords
from gensim import corpora
from gensim.models import ldamodel
import unicodecsv
import numpy as np


contractions_dictionary = {
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "i'll": "i will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
}

contractions_re = re.compile('(%s)' % '|'.join(contractions_dictionary.keys()))


def expand_contractions(s, contractions_dictionary=contractions_dictionary):
    def replace(match):
        return contractions_dictionary[match.group(0)]

    return contractions_re.sub(replace, s)


def preprocess():
    global tweets, i, a, b, sw, tweets_without_stopwords, word, all_tokens, tokens_once, text
    print '[log] Preprocessing tweets'
    tweets = []
    for i in range(0, len(labelled_tweet)):
        a = labelled_tweet[i][1]
        b = unicode(a, "utf-8").lower()
        b = re.sub('((www\.[\s]+)|(https?://[^\s]+))', 'T_URL', b)
        b = re.sub('(?:\s+|$)+', ' ', b)
        b = expand_contractions(b)
        b = re.sub('[^A-Za-z0-9-#-@]+', ' ', b)
        tweets.append(b)
    sw = set(stopwords.words('english'))
    sw.update((u'us', u'thank', u'hi', u'you', u'yours', u'please', u'we', u'this', u'&amp;', u'T_URL', u'we\'re',
               u'you.', u'hi,', u'-', u'.'))
    tweets_without_stopwords = []
    for i in range(0, len(tweets)):
        a = tweets[i]
        tweets_without_stopwords.append([word for word in a.split() if word.lower() not in sw])

    print '[log] Removing stopwords and words that occur once only in corpus'
    # Remove all words that occur only once in entire corpus
    all_tokens = sum(tweets_without_stopwords, [])
    tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
    tweets_without_stopwords = [[word for word in text if word not in tokens_once] for text in tweets_without_stopwords]
    print '[log] Preprocessing completed.'


def hellinger_distance(prob_a, prob_b):
    i = [t[0] for t in sorted(prob_a, key=lambda word: word[1])]
    j = [t[0] for t in sorted(prob_b, key=lambda word: word[1])]

    return 1 - np.sqrt(np.sum((np.sqrt(i) - np.sqrt(j)) ** 2)) / np.sqrt(2)


def main(argv, argv2):
    global train_set_path, pos_file, rec, labelled_tweet, dictionary, text, corpus, lda, corpus_lda, i, topic

    print '[log] Reading tweets from ' + argv

    train_set_path = '../../dataset/' + argv
    with open('%s' % train_set_path, 'rU') as pos_file:
        labelled_tweet = list(tuple(rec) for rec in csv.reader(pos_file, delimiter=','))

    preprocess()

    print '[log] Extracting keywords for topics'
    dictionary = corpora.Dictionary(tweets_without_stopwords)
    corpus = [dictionary.doc2bow(text) for text in tweets_without_stopwords]
    # lda = ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=20, iterations=100, update_every=1, passes=3)
    lda = ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=4, update_every=1, chunksize=10000, passes=1)
    lda.save('cpb_model.lda')
    corpus_lda = lda[corpus]

    i = 0
    output = []
    # for topic in lda.show_topics(num_topics=10):
    for topic in lda.print_topics():
        temp_output = [str(i), topic]
        output.insert(0, temp_output)
        i += 1

    # print [i for i in corpus_lda]

    # Source: http://stackoverflow.com/a/20991190
    scores = list(chain(*[[score for topic_id, score in topic] for topic in [doc for doc in corpus_lda]]))
    threshold = sum(scores) / len(scores)

    # Given 4 topics, maximum clusters I can have is 4. It really is a hit-and-miss. No one knows
    # exactly how many clusters I can have
    cluster1 = [j for i, j in zip(corpus_lda, tweets) if i[0][1] > threshold]
    cluster2 = [j for i, j in zip(corpus_lda, tweets) if i[1][1] > threshold]
    cluster3 = [j for i, j in zip(corpus_lda, tweets) if i[2][1] > threshold]
    cluster4 = [j for i, j in zip(corpus_lda, tweets) if i[3][1] > threshold]

    print cluster1
    print cluster2
    print cluster3
    print cluster4

    print '[log] Saving to ' + argv2 + ' file'
    with open(argv2, 'wb') as output_file:
        wr = unicodecsv.writer(output_file, encoding='utf-8', delimiter=',', quoting=unicodecsv.QUOTE_ALL)
        header = ["Topic", "Keywords"]
        wr.writerow(header)
        wr.writerows(output)

    print '[log] Output file ' + argv2 + ' has been created successfully.'

    prob_a = [i[1] for i in corpus_lda]
    prob_b = [i[2] for i in corpus_lda]
    result = hellinger_distance(prob_a, prob_b)
    print result


if __name__ == '__main__':

    if (len(sys.argv) < 3) or (len(sys.argv) > 3):
        print 'Usage: python profile_builder.py [dataset.csv] [output_file_name.csv]'
        print 'Note: [dataset.csv] must be contained in the "dataset" folder of the project.'
    else:
        main(sys.argv[1], sys.argv[2])