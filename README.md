# Insult Classifier
A prototype to identify insults with different NLP approaches.

# List of Test Data
[Twitter Tweets labelled as Hate Speech or Other](https://github.com/t-davidson/hate-speech-and-offensive-language/tree/master/data)

[GermEval-2018-Data](https://github.com/uds-lsv/GermEval-2018-Data) cited from Michael Wiegand, Melanie Siegel, and Josef Ruppenhofer: "Overview of the GermEval 2018 Shared Task on the Identification of Offensive Language", in Proceedings of the GermEval, 2018, Vienna, Austria.

# Label nothing as hate speech

Computing metrics in `mark_everything.py` pretending that a machine labelled nothing as hate speech.

| Scores        | English Data           | German Data <sup>[1](#myfootnote1)</sup> |
| ------------- |:-------------:| -----:|
| **Accuracy**      | 0.94      | 0.66  |
| **Precision**     | 0.94      | 0.66  |
| **Recall**        | 1         | 1  |
| **F1**            | 0.97    | 0.8  |

# Naive Bayes Approach

| Scores        | English Data           | German Data <sup>[1](#myfootnote1)</sup> | German Data with [German stop words](https://github.com/gosia-malgosia/german-stop-words) | English train German test | German train English test | Merged data
| ------------- |:-------------:| -----:|-----:|  -----:|-----:|-----:|
| **Accuracy**      | 0.94      | 0.76  |0.739 |  0.64  |0.71 |0.86 |
| **Precision**     | 0.94      | 0.78  |0.787 |  0.67  |0.94 |0.94 |
| **Recall**        | 0.9973    | 0.89  |0.83  |  0.90  |0.73 |0.89 |
| **F1**            | 0.9692    | 0.83  |0.81  |  0.77  |0.82 |0.92 |

with `train_test_split(random_state=1)`

# Conclusions so far

Intuitive interpretation of the given metrics for the English data is misleading, because seeing a score of 0.94 might seem good,
but it's as bad as marking everything the same label as the most frequent label. On the other side, the Naive Bayes metrics on the German data are a real improvement.


# Topic Modeling

Did not provide any useful classifications.

# RNN

Accuracy is at roughly 55%. There's still a lot of room for improvement. E. g. increased training time, more cleaning or [BERT approach](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification).

# TO DO

Trying out RNN Bert approach.

# Legal
[MIT](https://github.com/not-a-lawyer/insult_classifier/blob/master/LICENSE)

## Footnotes
<a name="myfootnote1">1</a>: with the standard English stop words list.
