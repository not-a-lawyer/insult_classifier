import unittest
from bayes_classifier import *


class MyTestCase(unittest.TestCase):
    def test_something(self):
        train_model(import_data())
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
