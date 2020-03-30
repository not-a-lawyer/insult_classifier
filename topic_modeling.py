import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np

#What does this do?
import nltk
nltk.download('wordnet')

def preprocess_data(text_samples):
    pass

def stemming_text_samples(text_samples):
    stemmer = SnowballStemmer("english")


    processed_text = []

    for token in gensim.utils.simple_preprocess(text_samples):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            processed_text.append(stemmer.stem(WordNetLemmatizer().lemmatize(text_samples, pos='v')))

    return processed_text

