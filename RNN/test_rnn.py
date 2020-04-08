import unittest

from bayes_classifier import *
from RNN.rnn import *


class MyTestCase(unittest.TestCase):
    def test_preprocessing(self):

        ##insults.csv needs to be copied into RNN directory
        type_tweet = import_data()

        preprocess_tweets_for_keras(type_tweet)
        tokens = tokenize_tweets(type_tweet)

        pass






if __name__ == '__main__':
    unittest.main()
