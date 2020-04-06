## Using Keras to classify some tweets as hate speech
## Following this guide at
# https://medium.com/@armandj.olivares/detecting-toxic-comments-with-keras-and-interpreting-the-model-with-eli5-dbe734f3e86b

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

# Keras specific
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical