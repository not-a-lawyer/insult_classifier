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
        model.fit(tokenized_tweets["tweet"], tokenized_tweets["class"], batch_size=32, epochs=2, validation_split=0.2)

        print(model.predict("She may or may not be a Jew but she 's certainly stupid , she seems to think the Blacks wo n't kill her alongside every other White they can get their dirty hands on , what a muppet !"))
        pass






if __name__ == '__main__':
    unittest.main()
