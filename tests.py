import unittest
from bayes_classifier import *

import pandas as pd
import string

class MyTestCase(unittest.TestCase):
    def test_english_data(self):
        training_data, testing_data, label_train, label_test = train_model(import_data())
        predictions = apply_model(training_data, testing_data, label_train)
        evaluate_model(label_test, predictions)


    def test_unique_tweet(self):
        tweet_data = pd.Series([
                                   "great, every other day theres maintenance issues, fix your shit blizzard im sick and tired of your piece of shit client",
                                   "test", "She may or may not be a Jew but she 's certainly stupid , she seems to think the Blacks wo n't kill her alongside every other White they can get their dirty hands on , what a muppet !"])
        training_data, testing_data, label_train, label_test = train_model(import_data(), tweet_data)


        predictions = apply_model(training_data, testing_data, label_train)

        print(predictions)

    def test_german_data(self):


        training_data, testing_data, label_train, label_test = train_model(relabel_german_data())
        predictions = apply_model(training_data, testing_data, label_train)
        #required for German data
        evaluate_model(label_test.astype('int'), predictions)

    def test_label(self):
        type_tweet = relabel_german_data()

        for class_ in type_tweet["class"]:
            assert(class_ == 0 or class_ == 1)

    def test_data_sets_merged(self):
        english = import_data()
        german = relabel_german_data()




        #see whether same error occurs as with sole use of German data
        merged_data = english.append(german, ignore_index=True)

        training_data, testing_data, label_train, label_test = train_model(merged_data)
        predictions = apply_model(training_data, testing_data, label_train)
        evaluate_model(label_test.astype('int'), predictions)















if __name__ == '__main__':
    unittest.main()
