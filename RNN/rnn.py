## Using Keras to classify some tweets as hate speech
## Following this guide at
# https://medium.com/@armandj.olivares/detecting-toxic-comments-with-keras-and-interpreting-the-model-with-eli5-dbe734f3e86b

from bayes_classifier import *

from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Bidirectional, Activation, Conv1D, GRU, TimeDistributed, Dropout, Embedding, GlobalMaxPooling1D, MaxPooling1D, Add, Flatten, SpatialDropout1D
from keras.optimizers import Adam
from keras.models import Model
from keras.losses import sparse_categorical_crossentropy

#import insults.csv. see import_data() for details on data or README
type_tweet = import_data()

def preprocess_tweets_for_keras(type_tweet, stopwords = 'English'):
    """
    preprocesses the type_tweet["tweet"] column to fit into keras
    :param type_tweet:
    :changes type_tweet:
    """
    ##Will do when refinement of accuracy is necessary

    ##remove stopwords

    if stopwords == 'English':

        stop_words = set(stopwords.words('english'))
    elif stopwords == 'German':
        # Source: https://github.com/gosia-malgosia/german-stop-words
        german_stop_words = "aber alle allem allen aller alles als also am an ander andere anderem anderen anderer anderes anderm andern anders auch auf aus bei bin bis bist da damit dann das dass dasselbe dazu daß dein deine deinem deinen deiner deines dem demselben den denn denselben der derer derselbe derselben des desselben dessen dich die dies diese dieselbe dieselben diesem diesen dieser dieses dir doch dort du durch ein eine einem einen einer eines einig einige einigem einigen einiger einiges einmal er es etwas euch euer eure eurem euren eurer eures für gegen gewesen hab habe haben hat hatte hatten hier hin hinter ich ihm ihn ihnen ihr ihre ihrem ihren ihrer ihres im in indem ins ist jede jedem jeden jeder jedes jene jenem jenen jener jenes jetzt kann kein keine keinem keinen keiner keines können könnte machen man manche manchem manchen mancher manches mein meine meinem meinen meiner meines mich mir mit muss musste nach nicht nichts noch nun nur ob oder ohne sein seine seinem seinen seiner seines selbst sich sie sind so solche solchem solchen solcher solches soll sollte sondern sonst um und uns unse unsem unsen unser unses unter vom von vor war waren warst was weil weiter welche welchem welchen welcher welches wenn werde werden wie wieder will wir wird wirst wo wollen wollte während würde würden zu zum zur zwar zwischen über "
        stop_words = german_stop_words.split(" ")




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

def pad_tweets(tokenized_tweets, length = None):
    """
    pads the already tokenized tweets
    :param tokenized_tweets: list of tokenized tweets\
    :param length:
    :return padded_tweets:
    :return length: length of padding
    """


    if length == None:
        length = max([len(tweet) for tweet in tokenized_tweets])

    return pad_sequences(tokenized_tweets, maxlen=length, padding='post'), length






def set_up_triple_lstm_model(input_length):
    """
    Set up the keras model
    :return model:
    """

    learning_rate = 1e-3

    input_tweet = Input(shape=(input_length,))
    tweet_embedding = Embedding(5000, 200)(input_tweet)
    lstm = LSTM(64, recurrent_dropout=0.2, return_sequences = True)(tweet_embedding)
    dropout = Dropout(0.2)(lstm)
    lstm2 = LSTM(64, recurrent_dropout=0.2,  return_sequences= True)(dropout)
    dropout2 = Dropout(0.2)(lstm2)
    flatten = Flatten()(dropout2)
    #TODO adjust output to 1 and make it work
    output = Dense(2, activation='softmax')(flatten)

    model = Model(input_tweet, output)

    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])
    return model



