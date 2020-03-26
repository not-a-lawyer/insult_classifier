from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_model(label_test, predictions):
    print('Accuracy score: ', format(accuracy_score(label_test, predictions)))
    print('Precision score: ', format(precision_score(label_test, predictions)))
    print('Recall score: ', format(recall_score(label_test, predictions)))
    print('F1 score: ', format(f1_score(label_test, predictions)))


def apply_model(training_data, testing_data , label_train):
    naive_bayes = MultinomialNB()
    training_data = training_data.astype('int')
    label_train = label_train.astype('int')

    naive_bayes.fit(training_data, label_train)
    predictions = naive_bayes.predict(testing_data)
    return predictions


def train_model(type_tweet, custom_tweet_data = pd.Series([])):
    data_train, data_test, label_train, label_test = train_test_split(type_tweet['tweet'],
                                                        type_tweet['class'],
                                                        random_state=1)
    count_vector = CountVectorizer(stop_words="english")

    # Fit training data and return a matrix
    training_data = count_vector.fit_transform(data_train)

    # Transform testing data and return s matrix.
    if not custom_tweet_data.empty:
        testing_data = count_vector.transform(custom_tweet_data)
    else:
        testing_data = count_vector.transform(data_test)

    return training_data, testing_data , label_train, label_test



def import_data(filename = "insults.csv"):
    # Dataset "insults.csv" from https://github.com/t-davidson/hate-speech-and-offensive-language/tree/master/data

    df = pd.read_csv(filename)

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

def relabel_german_data():
    names = ["tweet", "class", "detail"]

    #Downloaded the data from https://github.com/uds-lsv/GermEval-2018-Data/blob/master/germeval2018.training.txt. See readme for citation.
    data = pd.read_table("germeval2018.training.txt", names=names)



    #fitting dataframe to only have two columns
    reversed_column_labels = ["class", "tweet"]

    removed_colum = data[["tweet", "class"]]

    #swap columns
    type_tweet = removed_colum.reindex(columns = reversed_column_labels)

    # change all other labels to 1 and OFFENSE to 0
    # DataFrame cell has to be changed this way. Normal iterator only changes a copy
    for i in range(0, len(type_tweet)):
        if type_tweet.at[i, "class"] == "OTHER":
            type_tweet.at[i, "class"] = 1.
        else:
            type_tweet.at[i, "class"] = 0.
        i += 1




    return type_tweet

#Primtive approach to filter German and special characters
def filter_characters(value):
    allowed_characters = " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return ' '.join(c for c in value if c in allowed_characters )






