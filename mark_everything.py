def count_true_positive(two_column_data_set):

    positive_count = 0

    for data in two_column_data_set["class"]:
        ##Hate Speech is labelled 0 in this project
        if data == 0:
            positive_count += 1

    return positive_count

