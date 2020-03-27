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

        #Source: https://github.com/gosia-malgosia/german-stop-words
        german_stop_words = "aber alle allem allen aller alles als also am an ander andere anderem anderen anderer anderes anderm andern anders auch auf aus bei bin bis bist da damit dann das dass dasselbe dazu daß dein deine deinem deinen deiner deines dem demselben den denn denselben der derer derselbe derselben des desselben dessen dich die dies diese dieselbe dieselben diesem diesen dieser dieses dir doch dort du durch ein eine einem einen einer eines einig einige einigem einigen einiger einiges einmal er es etwas euch euer eure eurem euren eurer eures für gegen gewesen hab habe haben hat hatte hatten hier hin hinter ich ihm ihn ihnen ihr ihre ihrem ihren ihrer ihres im in indem ins ist jede jedem jeden jeder jedes jene jenem jenen jener jenes jetzt kann kein keine keinem keinen keiner keines können könnte machen man manche manchem manchen mancher manches mein meine meinem meinen meiner meines mich mir mit muss musste nach nicht nichts noch nun nur ob oder ohne sein seine seinem seinen seiner seines selbst sich sie sind so solche solchem solchen solcher solches soll sollte sondern sonst um und uns unse unsem unsen unser unses unter vom von vor war waren warst was weil weiter welche welchem welchen welcher welches wenn werde werden wie wieder will wir wird wirst wo wollen wollte während würde würden zu zum zur zwar zwischen über "
        german_stop_words = german_stop_words.split(" ")

        training_data, testing_data, label_train, label_test = train_model(relabel_german_data(), stop_words=[])
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

    def test_english_train_on_german_test(self):
        english = import_data()
        german = relabel_german_data()

        merged_data = english.append(german, ignore_index=True)

        #split the data within the function at this index
        split_index = len(english)
        training_data, testing_data, label_train, label_test = train_model_mixed_data(merged_data, split_index)
        predictions = apply_model(training_data, testing_data, label_train)
        evaluate_model(label_test.astype('int'), predictions)
        pass
















if __name__ == '__main__':
    unittest.main()
