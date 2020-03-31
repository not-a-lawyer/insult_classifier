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



    processed_text_sample = []

    for text in text_samples:
        processed_text = []

        for token in gensim.utils.simple_preprocess(text):
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
                processed_text.append(stemmer.stem(WordNetLemmatizer().lemmatize(token, pos='v')))

        processed_text_sample.append(processed_text)

    return processed_text_sample

def bag_of_words(preprocessed_sample):
    dictionary = gensim.corpora.Dictionary(preprocessed_sample)

    #create BoW for each Tweet
    bow_corpus = [dictionary.doc2bow(doc) for doc in preprocessed_sample]

    return dictionary, bow_corpus

def train_lda_model(dictionary, bow_corpus):
    lda_model = gensim.models.LdaMulticore(bow_corpus,
                                           num_topics=10,
                                           id2word=dictionary,
                                           passes=2,
                                           workers=2)
    return lda_model




