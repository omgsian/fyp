__author__ = 'omgwhut'

from difflib import SequenceMatcher as sm
# from replacers import SpellingReplacer
# from enchant.checker import SpellChecker

import pandas as pd
import time
import re

min_t = 0.82  # min 0.77
max_t = 1.0  # max 1.00
nl = '\n'

# Very naive approach of a list of contractions, but it'll do for now.
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

def sim_check(tweet):
    # Adds a new row to the dataframe tweet with a unique index for every row item
    tweet['new_row'] = range(1, len(tweet) + 1)

    # Assesses every pair of adjacent tweets and determines the similarity ratio
    # If ratio is between min and max threshold set above, get the length of each of the paired tweets i and j
    # Of the paired tweets i and j, whichever is the shorter tweet, drop them from the tweet dataframe based on the tweet id
    for i, j, k, l in zip(tweet['text'], tweet['text'][1:], tweet['new_row'], tweet['new_row'][1:]):
        adj_sim = sm(a=i, b=j)

        if min_t < adj_sim.ratio() < max_t:
            if len(i) < len(j):
                tweet = tweet.drop(tweet.index[tweet.new_row == k])
            else:
                tweet = tweet.drop(tweet.index[tweet.new_row == l])

    # print tweet
    # print 'l2: ' ,len(tweet)
    return tweet


def expand_contractions(s, contractions_dictionary=contractions_dictionary):
    def replace(match):
        return contractions_dictionary[match.group(0)]

    return contractions_re.sub(replace, s)


def process_tweet(tweet):
    overall_time = time.time()

    original_dataset_length = str(len(tweet))

    print "Length of tweet dataframe prior to processing:" + original_dataset_length

    # Replaces all URL beginning with www with a T_URL token
    tweet = tweet.replace(value='T_URL', regex='((www\.[\s]+)|(https?://[^\s]+))')

    # Replaces all URL beginning with HTTP with a T_URL token
    tweet = tweet.replace(value='T_URL', regex='http:?\S+')

    # Removes all hashtags
    tweet = tweet.replace(value='', regex='#([^\s]+)')

    # Replaces all username
    # tweet = tweet.replace(value='T_USERNAME', regex='@([A-Za-z0-9_]+)')

    # tweet = tweet.replace(value=' ', regex='[\s]+')
    tweet = tweet.replace(value=' ', regex='(?:\s+|$)+')

    # TODO: Unsure if i should replace any 'rt' as T_RETWEET... regexp doesn't seem to cover all occurences of RTs
    # tweet = pd.DataFrame(tweet).replace(value='T_RETWEET', regex='\s[rt]+\s[@]+')

    # TODO: Need to consider the removal of emojis which are essentially unicode characters.
    # See: http://bit.ly/1uNREx1)

    # Replaces all text emoticons with appropriate tokens
    tweet = tweet.replace(value='T_POS_EMO', regex='[:=;]-?[)DPp]|[D(]-?[:=;]')
    tweet = tweet.replace(value='T_NEG_EMO', regex='[:=;]-?[(/]|[D(]-?[:=;]')

    # Remove ending whitespace in the dataframe
    # tweet = tweet['text].applymap(lambda x: x.rstrip(' '))

    # Sort text column in dataframe
    tweet = pd.DataFrame(tweet).sort(columns='text')

    # Drop all duplicate tweets
    # tweet = pd.DataFrame(tweet).drop_duplicates()
    tweet = tweet.drop_duplicates('text')

    tweet = sim_check(tweet)

    print "Length of tweet dataframe after processing:" + str(len(tweet))

    expandedList = []  # empty list to store tweets with expanded contractions

    for i in range(0, len(tweet)):
        s = tweet['text'].iloc[i]
        _tweetList = expand_contractions(s)
        expandedList.append(_tweetList)

    tweet['text'] = expandedList

    # replacer = SpellingReplacer()
    #
    # chkr = SpellChecker("en_US")
    #
    # for i in range(0,len(tweet)):
    #     s = tweet['text'].iloc[i]
    #     chkr.set_text(s) # NoneType
    #
    #     for err in chkr:
    #         print(err.word + " at position " + str(err.wordpos)) # prints out misspells in every tweet
    #         err.replace(replacer.replace(err.word)) # seems to be a problem on this line...  'NoneType' object has no attribute 'encode'
    #
    #     print chkr.get_text()

    # Uses textblob spelling correction - but no way of doing exclusion
    # for i in range(0, len(tweet['text'])):
    #     b = TextBlob(tweet['text'].iloc[i].decode('utf-8')).correct()
    #     print b
    #     # c = TextBlob(b)
    #     # print c.correct()

    print tweet['text']

    # Saves processed data to a CSV file
    tweet.to_csv(path_or_buf='./postprocessed_data.csv')

    print 'Time taken to DPM entire dataset of ' + original_dataset_length + ' in size is :' \
          + str(time.time() - overall_time)


if __name__ == '__main__':
    data = pd.read_csv(
        '../../dataset/allTweets.csv',
    )

    # Convert values in every column in dataset to lowercase
    data['text'] = data.text.str.lower()
    data['user.screen_name'] = data['user.screen_name'].str.lower()
    data['coordinates.coordinates'] = data['coordinates.coordinates'].str.lower()
    data['created_at'] = data['created_at'].str.lower()
    data['user.location'] = data['user.location'].str.lower()
    data['source'] = data['source'].str.lower()

    pd.set_option('display.max_colwidth', -1)
    pd.set_option('max_rows', 500000)

    process_tweet(data)