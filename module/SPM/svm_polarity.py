from nltk.classify.scikitlearn import SklearnClassifier as sklc
from sklearn.svm import LinearSVC
import csv
import unicodecsv


def getAccuracy():
    global right, wrong, msg, feat, result
    right, wrong = 0, 0
    for msg in neg_testing_set:
        feat = extract_features_in_tweet(msg)
        result = clf.classify(feat)
        if result == "negative":
            right += 1
        else:
            wrong += 1
    for msg in pos_testing_set:
        feat = extract_features_in_tweet(msg)
        result = clf.classify(feat)
        if result == "positive":
            right += 1
        else:
            wrong += 1
    print "Accuracy: {}".format(right / float(right + wrong))


def extract_features_in_tweet(tweet):
    words = tweet.split()
    return dict((i, True) for i in words)


def get_train_features(tweets, sentiment):
    messageFeatures = []
    for t in tweets:
        feat = extract_features_in_tweet(t)
        messageFeatures.append((feat, sentiment))
    return messageFeatures

positiveTweets = []
negativeTweets = []

train_set = csv.reader(open('train_set.csv', 'rU'), delimiter=',')
tweets = []
for row in train_set:
    sentmt = row[1]
    msg = row[0]
    msg_sent = (msg, sentmt)
    tweets.append(msg_sent)
    if sentmt == "positive":
        positiveTweets += msg_sent
    else:
        negativeTweets += msg_sent


neg_cutoff, pos_cutoff = len(negativeTweets) * 3/4, len(positiveTweets) * 3/4
pos_training_set, pos_testing_set = positiveTweets[:pos_cutoff], positiveTweets[pos_cutoff:]
neg_training_set, neg_testing_set = negativeTweets[:neg_cutoff], negativeTweets[neg_cutoff:]

train_neg_feature = get_train_features(neg_training_set, 'negative')
train_pos_feature = get_train_features(pos_training_set, 'positive')
train_feats = train_neg_feature + train_pos_feature

clf = sklc(LinearSVC()).train(train_feats)

getAccuracy()

test_set = csv.reader(open('test_set.csv', 'rU'), delimiter=',')
output_res = []
for row in test_set:
    msg = row[0]
    temp_list = [msg, clf.classify(extract_features_in_tweet(msg))]
    output_res.insert(0, temp_list)

with open('svm_output.csv', 'wb') as output_file:
    wr = unicodecsv.writer(output_file, encoding='utf-8', delimiter=',', quoting=unicodecsv.QUOTE_ALL)
    wr.writerows(output_res)

print '[log] Output file saved'