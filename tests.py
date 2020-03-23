import unittest
from insult_classifier import *


class MyTestCase(unittest.TestCase):
    def test_something(self):
        import_data()
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
