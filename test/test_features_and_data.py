"""
    ML Testing the StackOverflow label predictor for features and data. Making use of the mltest library.
"""
import joblib
import numpy as np
import pytest

import libtest.features_and_data as lib
from src.text_preprocessing import text_prepare
from src.vectorization import my_bag_of_words


@pytest.mark.fast
def test_no_unsuitable_features():
    lib.no_unsuitable_features(['title'], [])


def prepare_correlation_analysis():
    mybag, tfidf, _, _ = joblib.load("output/vectorized_x.joblib")
    y_train, _ = joblib.load("output/y_preprocessed.joblib")

    unique_labels = {None}
    for i in y_train:
        for j in i:
            unique_labels.add(j)

    labels_id, id_labels = {}, {}
    counter = 0
    for i in unique_labels:
        labels_id[counter] = i
        id_labels[i] = counter
        counter += 1

    labels_matrix = np.zeros([mybag.shape[0], len(unique_labels)])
    print(labels_matrix.shape)

    for i in range(len(y_train)):
        for j in y_train[i]:
            labels_matrix[i][id_labels[j]] = 1

    return mybag, tfidf, labels_matrix


@pytest.mark.fast
def test_pairwise_feature_correlations():
    mybag, tfidf, _ = prepare_correlation_analysis()

    lib.pairwise_feature_correlations(mybag, sample_size=100000)
    lib.pairwise_feature_correlations(tfidf, sample_size=100000)


@pytest.mark.fast
def test_feature_target_correlations():
    mybag, tfidf, labels_matrix = prepare_correlation_analysis()

    for i in range(10):
        # Only calculate for mybag. Tf-Idf has too many features and the correlation matrix would grow out of memory.
        lib.feature_target_correlations(mybag, labels_matrix[:, i])


@pytest.mark.slow
def test_feature_values():
    mybag, tfidf, _, _ = joblib.load("output/vectorized_x.joblib")
    lib.feature_values(mybag, 0, [0.0, 1.0, 2.0])


def test_top_feature_values():
    mybag, tfidf, _, _ = joblib.load("output/vectorized_x.joblib")
    lib.top_feature_values(mybag, 0, [0.0, 1.0],at_least_top_k_account_for= 0.8)


@pytest.mark.fast
def test_preprocessing_prepare():
    examples = ["SQL Server - any equivalent of Excel's CHOOSE function?",
                "How to free c++ memory vector<int> * arr?"]
    answers = ["sql server equivalent excels choose function",
               "free c++ memory vectorint arr"]

    lib.preprocessing_validation(examples, answers, text_prepare)


@pytest.mark.fast
def test_preprocessing_bag_of_words():
    words_to_index = {'hi': 0, 'you': 1, 'me': 2, 'are': 3}
    examples = ['hi how are you']
    answers = [[1, 1, 0, 1]]

    lib.preprocessing_validation(examples, answers,
                                 lambda x: my_bag_of_words(x, words_to_index, 4),
                                 equals=lambda a, b: (a == b).all())
