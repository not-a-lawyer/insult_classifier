from sklearn.model_selection import train_test_split
import pandas as pd

def train_model(type_tweet):
    data_train, data_test, label_train, label_test = train_test_split(type_tweet['tweet'],
                                                        type_tweet['class'],
                                                        random_state=1)

    pass



def import_data():
    # Dataset "insults.csv" from https://github.com/t-davidson/hate-speech-and-offensive-language/tree/master/data

    df = pd.read_csv("insults.csv")

    #Get for each index the class.
    # The class "0" is hate speech
    # and everything else won't be considered hate speech.
    names = ["class", "tweet"]
    type_tweet = df[names]

    #change all other labels to 1
    #DataFrame cell has to be changed this way. Normal iterator only changes a copy
    for i in range(0, len(type_tweet["class"])):
        if type_tweet.at[i, "class"] > 0:
            type_tweet.at[i, "class"] = 1


    return type_tweet







