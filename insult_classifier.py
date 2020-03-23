#from sklearn.cross_validation import train_test_split
import pandas as pd


def import_data():
    # Dataset "insults.csv" from https://github.com/t-davidson/hate-speech-and-offensive-language/tree/master/data

    df = pd.read_csv("insults.csv")

    #Get for each index the class.
    # The class "0" is hate speech
    # and everything else won't be considered hate speech.

    classes = df["class"]
    tweets = df["tweet"]

    hate_speech = []
    other_speech = []
    index = 0
    for class_ in classes:

        if class_ == 0:
            hate_speech.append(index)
        else:
            other_speech.append(index)
        index += 1

    first_hate_speech = tweets[hate_speech[1]]
    pass




# split into training and testing sets