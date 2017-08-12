# coding=utf-8

from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier
from textblob.taggers import NLTKTagger
from nltk.corpus import stopwords
import csv
import re
import unicodecsv
import timing

nltk_tagger = NLTKTagger()

def preprocess(labelled_tweet, unlabelled_tweet):

    sw = set(stopwords.words('english'))

    train_set = []
    test_set = []

    for i in range(0, len(labelled_tweet)):
        a = labelled_tweet[i][0]
        sentiment = labelled_tweet[i][1]
        b = unicode(a, 'utf-8')
        # b = b.translate(string.maketrans("",""), string.punctuation)
        b = re.sub('[^A-Za-z0-9-#-@]+', ' ', b)
        # s = ' '.join([word for word in b.split() if word not in sw])
        s = [word for word in b.split() if word not in sw]
        temp_train_set = [s, sentiment]

        train_set.append(temp_train_set)

    for i in range(0, len(unlabelled_tweet)):
        a = unlabelled_tweet[i][0]
        b = unicode(a, 'utf-8')
        # b = b.translate(string.maketrans("",""), string.punctuation)
        b = re.sub('[^A-Za-z0-9-#-@-\-]+', ' ', b)
        # s = ' '.join([word for word in b.split() if word not in sw])
        s = [word for word in b.split() if word not in sw]

        test_set.append(s)

    return train_set, test_set


if __name__ == '__main__':

    print '[log] Opening train and test files'
    try:
        train_set_path = './train_set3.csv'
        with open('%s' % train_set_path, 'rU') as pos_file:
            labelled_tweet = list(tuple(rec) for rec in csv.reader(pos_file, delimiter=','))

        test_set_path = './test_set3.csv' # small dataset
        with open('%s' % test_set_path, 'rU') as test_set:
            unlabelled_tweet = list(tuple(rec) for rec in csv.reader(test_set, delimiter=','))
    except:
        print '[log] Files train.csv and test.csv are not available'

    try:
        print '[log] Preprocessing tweets'
        labelled_tweet, unlabelled_tweet = preprocess(labelled_tweet, unlabelled_tweet)
    except:
        print '[log] Something went wrong with the preprocessing!'

    # print labelled_tweet
    print '[log] Training classifier'
    cl = NaiveBayesClassifier(labelled_tweet)
    print '[log] Classifier trained successfully'

    result = []
    accuracyTest = []

    print '[log] Assigning polarities to the test set'
    for t in unlabelled_tweet:
        tweet = ' '.join(t)
        blob = TextBlob(tweet, classifier=cl)
        # temp_list1 = [blob, blob.classify(), blob.SCM]
        temp_list1 = [blob, blob.classify()]
        result.insert(0, temp_list1)
        temp_list2 = [blob, blob.classify()]
        accuracyTest.insert(0, temp_list2)

    # label = ["Subjective", "Objective", "Neutral"]
    #
    # for i in range(0, len(result)):
    #     if result[i][2] < 0.49:
    #         result[i][2] = label[0]
    #     elif result[i][2] > 0.51:
    #         result[i][2] = label[1]
    #     elif result[i][2] == 0.5:
    #         result[i][2] = label[2]

    print '[log] Saving output to file'
    with open('nb_output.csv', 'wb') as output_file:
        wr = unicodecsv.writer(output_file, encoding='utf-8', delimiter=',', quoting=unicodecsv.QUOTE_ALL)
        wr.writerows(result)
    print '[log] Output file saved'

    accuracyValue = cl.accuracy(accuracyTest)
    print '[log] Classifier Accuracy: {0} ' .format(accuracyValue)
    cl.show_informative_features(5)