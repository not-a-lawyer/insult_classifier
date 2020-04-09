import unittest

from bayes_classifier import *
from RNN.rnn import *


class MyTestCase(unittest.TestCase):
    def test_general(self):

        ##insults.csv needs to be copied into RNN directory
        type_tweet = import_data()

        preprocess_tweets_for_keras(type_tweet)
        tokenized_tweets, tokens = tokenize_tweets(type_tweet)

        print(tokens.word_index)
        print(len(tokens.word_index))
        model = set_up_triple_lstm_model()
        pass






if __name__ == '__main__':
    unittest.main()
