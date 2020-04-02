def count_true_positive(two_column_data_set):

    positive_count = 0

    for data in two_column_data_set["class"]:
        ##Hate Speech is labelled 0 in this project
        if data == 0:
            positive_count += 1

    return positive_count

def compute_precision(positive_count, two_column_data_set):

    #positive count is false positives and rest of data set is true positive if all data is marked non hate speech
    return (len(two_column_data_set["class"])-positive_count)/len(two_column_data_set["class"])

def compute_recall(positive_count, two_column_data_set):
    #always one, because there's never a true negative, because hate speech is never labelled as such
    return (len(two_column_data_set["class"])-positive_count)/(len(two_column_data_set["class"])-positive_count)

