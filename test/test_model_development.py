"""
    ML Testing the StackOverflow label predictor for model development. Making use of the [todo library name] library.
"""
import unittest
import joblib
import numpy as np

import libtest.model_development as lib

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd

output_directory = "../output"


def test_tfidf_against_baseline():
    # Run own model and get score
    _, X_train_tfidf, _, X_val_tfidf= joblib.load(output_directory + "/vectorized_x.joblib")
    Y_train, Y_val = joblib.load(output_directory + "/fitted_y.joblib")

    (accuracy, f1, avg_precision) = joblib.load(output_directory + "/TFIDF_scores.joblib")
    scores = {"ACC": accuracy, "F1": f1, "AP": avg_precision}

    baseline_scores, score_differences = lib.compare_against_baseline(scores, X_train_tfidf, X_val_tfidf, Y_train, Y_val, model="linear")

    # Assert every score differs at least 10 percent from the baseline
    for score, diff in score_differences.items():
        print(score, " score difference: ", diff)
        assert (diff > 0.1)


def test_tunable_hyperparameters():
    X_train, X_val, _ = joblib.load(output_directory + "/X_preprocessed.joblib")
    Y_train, Y_val = joblib.load(output_directory + "/y_preprocessed.joblib")
    curr_params = joblib.load(output_directory + "/logistic_regression_params.joblib")
    classifier = joblib.load(output_directory + "/logistic_regression.joblib")
    classifier_mybag, classifier_tfidf = joblib.load(output_directory + "/classifiers.joblib")

    mlb = MultiLabelBinarizer()
    X_train = mlb.fit_transform(X_train)
    Y_train = mlb.fit_transform(Y_train)

    tunable_parameters = {
        "estimator__penalty": ['l1', 'l2'],
        "estimator__C": [0.1, 1.0],
    }

    percentage_mybag, optimal_parameters_mybag = lib.tunable_hyperparameters(classifier_mybag, tunable_parameters,
                                                                             curr_params, X_train, Y_train)
    print("dissimilar percentage_mybag: " + percentage_mybag + ", current: " + curr_params + ", optimal: "
          + optimal_parameters_mybag)

    percentage_tfidf, optimal_parameters_tfidf = lib.tunable_hyperparameters(classifier_tfidf, tunable_parameters,
                                                                             curr_params, X_train, Y_train)
    print("dissimilar percentage_tfidf: " + percentage_tfidf + ", current: " + curr_params + ", optimal: "
          + optimal_parameters_tfidf)

def test_data_slicing():
    X_train, X_val, _ = joblib.load(output_directory + "/X_preprocessed.joblib")
    Y_train, Y_val = joblib.load(output_directory + "/y_preprocessed.joblib")
    # print("x train", X_train[:5])
    # print("y train", Y_train[:5])
    length = (len(x.split()) for x in X_train)
    X_train_mybag, _, X_val_mybag, _ = joblib.load(output_directory + "/vectorized_x.joblib")
    tuples = list((x, y, z) for x,y,z in zip(X_train_mybag, Y_train, length))
    tuples.sort(key=lambda y: y[2])

    # print(len(tuples))

    slices = {}
    # count = 0
    for t in tuples:
        # count += 1
        # print(count)
        if t[2] not in slices.keys():
            slices[t[2]] = []
        slices[t[2]].append((t[0], t[1]))


    model = OneVsRestClassifier(LogisticRegression(penalty='l1', C=1, dual=False, solver='liblinear'))
    # lib.data_slices(model, slices, X_val, Y_val)
    min = 100
    max = 0

    tags_counts = joblib.load(output_directory + "/tags_counts.joblib")
    mlb = MultiLabelBinarizer(classes=sorted(tags_counts.keys()))
    # y_train = mlb.fit_transform(y_train)
    Y_val = mlb.fit_transform(Y_val)


    for key in slices.keys():
        x_slice = []
        for x in slices[key]:

            # nsamples, nx, ny = x[0].toarray().shape
            # to_add = x[0].reshape((nsamples,nx*ny))
            x_slice.append(x[0].toarray())
        x_slice = np.stack(x_slice, axis=0 )

        y_slice = []
        for y in slices[key]:
            y_slice.append(y[1])
        y_slice = mlb.fit_transform(y_slice)
        #
        # print(x_slice.ndim)
        # print(y_slice.ndim)
        nsamples, nx, ny = x_slice.shape
        x_slice = x_slice.reshape((nsamples,nx*ny))
        model.fit(x_slice, y_slice)
        score = model.score(X_val_mybag, Y_val)
        print(score)
        if score < min:
            min = score
        if score > max:
            max = score

    assert max-min < 0.2







if __name__ == '__main__':
    # test_tunable_hyperparameters()
    test_data_slicing()

