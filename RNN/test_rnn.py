import unittest

from bayes_classifier import *
from RNN.rnn import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class MyTestCase(unittest.TestCase):
    def test_general(self):

        ##insults.csv needs to be copied into RNN directory
        type_tweet = import_data()

        preprocess_tweets_for_keras(type_tweet)
        tokenized_tweets, token = tokenize_tweets(type_tweet)

        padded_tweets, padding_length = pad_tweets(tokenized_tweets)

        input_length = len(padded_tweets)
        print(token.word_index)
        print(len(token.word_index))
        model = set_up_triple_lstm_model(padding_length)
        model.fit(padded_tweets, type_tweet["class"], batch_size=32, epochs=2, validation_split=0.2, verbose=1)



        tokenized_test_tweet = token.texts_to_sequences(["""She may or may not be a Jew
         but she 's certainly stupid , she seems to think the Blacks wo n't kill her alongside every other 
         White they can get their dirty hands on , what a muppet !""", "Test", "Bitch", """DevilGrimz: @VigxRArts you're fucking gay, blacklisted hoe"" Holding out for #TehGodClan anyway http://t.co/xUCcwoetmn"""])

        padded_test_tweet, _ = pad_tweets(tokenized_test_tweet, padding_length)

        result = model.predict(padded_test_tweet)

        pass

    def test_german(self):
        #german data txt also needs to be copied into this subdirectory
        type_tweet = relabel_german_data()

        preprocess_tweets_for_keras(type_tweet, stopwords="German")
        tokenized_tweets, token = tokenize_tweets(type_tweet)

        padded_tweets, padding_length = pad_tweets(tokenized_tweets)

        input_length = len(padded_tweets)
        print(token.word_index)
        print(len(token.word_index))
        model = set_up_triple_lstm_model(padding_length)
        model.fit(padded_tweets, type_tweet["class"], batch_size=32, epochs=2, validation_split=0.2, verbose=1)

        tokenized_test_tweet = token.texts_to_sequences(["""She may or may not be a Jew
                 but she 's certainly stupid , she seems to think the Blacks wo n't kill her alongside every other 
                 White they can get their dirty hands on , what a muppet !""", "Test", "Bitch",
                                                         """DevilGrimz: @VigxRArts you're fucking gay, blacklisted hoe"" Holding out for #TehGodClan anyway http://t.co/xUCcwoetmn"""])

        padded_test_tweet, _ = pad_tweets(tokenized_test_tweet, padding_length)

        result = model.predict(padded_test_tweet)
        print(result)

    def test_evaluate_german(self):
        # german data txt also needs to be copied into this subdirectory
        type_tweet = relabel_german_data()

        preprocess_tweets_for_keras(type_tweet, 'German')
        tokenized_tweets, token = tokenize_tweets(type_tweet)

        padded_tweets, padding_length = pad_tweets(tokenized_tweets)

        model = set_up_triple_lstm_model(padding_length)

        x_train, x_test, y_train, y_test = train_test_split(padded_tweets, type_tweet["class"],
                                                                          random_state=1)

        model.fit(x_train, y_train, batch_size=32, epochs=2, validation_split=0.2, verbose=1)

        probabilities = model.predict(x_test)
        predictions = []

        """for probability in probabilities:
            #set threshold value for hate speech at 10 percent
            if probability[0] >= 0.1:
                predictions.append(0)
            else:
                predictions.append(1)"""



        print(accuracy_score(y_test.tolist(), predictions))
        print(precision_score(y_test.tolist(), predictions))
        print(recall_score(y_test.tolist(), predictions))
        print(f1_score(y_test.tolist(), predictions))










if __name__ == '__main__':
    unittest.main()
