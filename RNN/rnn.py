## Using Keras to classify some tweets as hate speech
## Following this guide at
# https://medium.com/@armandj.olivares/detecting-toxic-comments-with-keras-and-interpreting-the-model-with-eli5-dbe734f3e86b

from bayes_classifier import *

from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Input, LSTM, Bidirectional, Activation, Conv1D, GRU, TimeDistributed, Dropout, Embedding, GlobalMaxPooling1D, MaxPooling1D, Add, Flatten, SpatialDropout1D
from keras.optimizers import Adam
from keras.models import Model
from keras.losses import sparse_categorical_crossentropy

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



    # DataFrame cell has to be changed this way. Normal iterator only changes a copy
    for i in range(0, len(type_tweet["tweet"])):
        type_tweet.at[i, "tweet"] = type_tweet.at[i, "tweet"].split(" ")

        #iterate over words in tweet
        for j in range(0, len(type_tweet.at[i, "tweet"] )):
            if type_tweet.at[i, "tweet"][j] in stop_words:
                type_tweet.at[i, "tweet"][j] = " "


    ##remove links

def tokenize_tweets(type_tweet):
    """
    Takes the tweets and tokenizes them

    :param type_tweet:
    :return tweets_tokens:
    """


    tkn = Tokenizer(num_words=5000)
    tkn.fit_on_texts(type_tweet["tweet"])
    return tkn.texts_to_sequences(type_tweet["tweet"]), tkn





def set_up_triple_lstm_model():
    """
    Set up the keras model
    :return model:
    """

    learning_rate = 1e-3

    input_tweet = Input(shape=(None,))
    tweet_embedding = Embedding(5000, 200)(input_tweet)
    lstm = LSTM(64, recurrent_dropout=0.2, return_sequences = True)(tweet_embedding)
    dropout = Dropout(0.2)(lstm)
    lstm2 = LSTM(64, recurrent_dropout=0.2,  return_sequences= True)(dropout)
    dropout2 = Dropout(0.2)(lstm2)
    output = Dense(2, activation='softmax')(dropout2)

    model = Model(input_tweet, output)

    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])
    return model



