import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import torch
import transformers as ppb
import warnings

from bayes_classifier import import_data
warnings.filterwarnings('ignore')

"""Following this tutorial: 
https://github.com/
jalammar/jalammar.github.io/blob/master/notebooks/bert/A_Visual_Notebook_to_Using_BERT_for_the_First_Time.ipynb
"""

def main():

    # For DistilBERT:
    model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

    #get insults data
    type_tweet = import_data()

    # Load pretrained model/tokenizer
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)

    #cleaning
    #TODO

    #tokenize tweet
    tokenized_tweet = type_tweet["tweet"].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))

    max_len = 0
    for i in tokenized_tweet.values:
        if len(i) > max_len:
            max_len = len(i)

    #_______________________why [0]?_________________#
    padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized_tweet.values])

    attention_mask = np.where(padded != 0, 1, 0)
    attention_mask.shape

    input_ids = torch.tensor(padded)
    attention_mask = torch.tensor(attention_mask)

    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)

    features = last_hidden_states[0][:, 0, :].numpy()
    labels = type_tweet["class"]

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels)

    lr_clf = LogisticRegression()
    lr_clf.fit(train_features, train_labels)
    print(lr_clf.score(test_features, test_labels))



    pass

if __name__ == "__main__":
    main()