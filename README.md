# Insult Classifier
A prototype to identify insults with different NLP approaches.

# List of Test Data
[Twitter Tweets labelled as Hate Speech or Other](https://github.com/t-davidson/hate-speech-and-offensive-language/tree/master/data)

[GermEval-2018-Data](https://github.com/uds-lsv/GermEval-2018-Data) cited from Michael Wiegand, Melanie Siegel, and Josef Ruppenhofer: "Overview of the GermEval 2018 Shared Task on the Identification of Offensive Language", in Proceedings of the GermEval, 2018, Vienna, Austria.

# Naive Bayes Approach
**Accuracy**:  0.9404454486765655

**Precision**:  0.9427226999837741

**Recall**:  0.9972536903535874

**F1**:  0.9692217866377512

| Scores        | English Data           | German Data  | German Data with [German stop words](https://github.com/gosia-malgosia/german-stop-words)
| ------------- |:-------------:| -----:|-----:|
| **Accuracy**      | 0.94      | 0.76  |0.739 |
| **Precision**     | 0.94      | 0.78  |0.787 |
| **Recall**        | 0.9973    | 0.89  |0.83  |
| **F1**            | 0.9692    | 0.83  |0.81  |


with `train_test_split(random_state=1)`

(Hoping everything has been done correctly and no test-train mixup happenend)


# Legal
[MIT](https://github.com/not-a-lawyer/insult_classifier/blob/master/LICENSE)
