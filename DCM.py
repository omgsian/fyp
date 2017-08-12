__author__ = 'omgwhut'

from twython import TwythonStreamer
import pymongo


class StreamingModule(TwythonStreamer):
    def on_success(self, data):
        tweets = []
        if data['lang'] == 'en':
            if 'text' in data:
                tweet = data['text'].encode('utf-8')
                tweets.append(tweet)
                db.fyp.save(data)
                # print data
                print '---------------------------------------------------------------------------'
                print 'Screen Name : ' + data['user']['screen_name'].encode('utf-8') + '\n'
                print tweet
                print '---------------------------------------------------------------------------\n'
            else:
                print 'Found a tweet mentioning an airline, but not in English... Ignore'


def on_error(self, status_code, data):
    print status_code
    self.disconnect()


if __name__ == '__main__':
    # replace these with the details from your Twitter Application
    consumer_key = 'BJzNZRGhfqtlWoFfsXQxtnIw0'
    consumer_secret = 'CAK4PFd8d49zY2bMHcf3lAirjwJb658IR9oXnfZiZ64iLYFN5P'
    access_token = '21082829-wZ1nV9w58RImLTeaJzw3Yr4230KAm7rDhTblbCQfr'
    access_token_secret = 'yaCC0uXiqxjEypBtufI1aHNkdwqkhPLevp3s0ADPHT0QI'

    streamer = StreamingModule(consumer_key, consumer_secret,
                               access_token, access_token_secret)

    # Setup a connection to mongodb
    connection = pymongo.Connection('mongodb://128.199.169.238/', 27017)
    db = connection.fyp

    # Track the following airlines with variants included
    # Ignores case sensitivity
    streamer.statuses.filter(track=[x.lower() for x in
                                    ['@singaporeair', '@cathaypacific', '@mas', 'Singapore Airlines',
                                     'Malaysia Airlines', 'Cathay Pacific', '@British_Airways',
                                     'British Airways']])