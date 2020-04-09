## Using Keras to classify some tweets as hate speech
## Following this guide at
# https://medium.com/@armandj.olivares/detecting-toxic-comments-with-keras-and-interpreting-the-model-with-eli5-dbe734f3e86b

from bayes_classifier import *

from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Input, LSTM, Bidirectional, Activation, Conv1D, GRU, TimeDistributed, Dropout, Embedding, GlobalMaxPooling1D, MaxPooling1D, Add, Flatten, SpatialDropout1D

#import insults.csv. see import_data() for details on data or README
type_tweet = import_data()

def preprocess_tweets_for_keras(type_tweet):
    """
    preprocesses the type_tweet["tweet"] column to fit into keras
    :param type_tweet:
    :changes type_tweet:
    """
    ##Will do when refinement of accuracy is necessary

    ##remove stopwords

    stop_words = set(stopwords.words('english'))

    for tweet in type_tweet["tweet"]:

        tweet = tweet.split(" ")



        for word in tweet:
            if word in stop_words:
                word = " "


    ##remove links

def tokenize_tweets(type_tweet):
    """
    Takes the tweets and tokenizes them

    :param type_tweet:
    :return tweets_tokens:
    """


    tkn = Tokenizer(num_words=10000)
    tkn.fit_on_texts(type_tweet["tweet"])
    return tkn.texts_to_sequences(type_tweet["tweet"]), tkn





def set_up_model():
    """
    Set up the keras model
    :return model:
    """


    input_tweet = Input(shape=(None,))
    tweet_embedding = Embedding(10000, 200)
    tweets = type_tweet["tweet"]
    hate_speech_labels = type_tweet["tweet"]
    pass