import unittest
from bayes_classifier import *


class MyTestCase(unittest.TestCase):
    def test_something(self):
        training_data, testing_data, label_train, label_test = train_model(import_data())
        predictions = apply_model(training_data, testing_data, label_train, label_test)
        evaluate_model(label_test, predictions)
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
