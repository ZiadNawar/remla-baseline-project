"""
    ML Testing the StackOverflow label predictor for model development. Making use of the mltest library.
"""
import math
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

import libtest.model_development as lib

output_directory = "output"


def test_tfidf_against_baseline():
    # Run own model and get score
    _, X_train_tfidf, _, X_val_tfidf = joblib.load(output_directory + "/vectorized_x.joblib")
    Y_train, Y_val = joblib.load(output_directory + "/fitted_y.joblib")

    (accuracy, f1, avg_precision) = joblib.load(output_directory + "/TFIDF_scores.joblib")
    scores = {"ACC": accuracy, "F1": f1, "AP": avg_precision}

    baseline_scores, score_differences = lib.compare_against_classification_baseline(scores, X_train_tfidf, X_val_tfidf,
                                                                                     Y_train, Y_val, model="linear")

    # Assert every score differs at least 10 percent from the baseline
    for score, diff in score_differences.items():
        # print(score, " score difference: ", diff)
        assert (diff > 0.01)


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
    X_train, _, _ = joblib.load(output_directory + "/X_preprocessed.joblib")
    Y_train, Y_val = joblib.load(output_directory + "/y_preprocessed.joblib")

    X_train_mybag, _, X_val_mybag, _ = joblib.load(output_directory + "/vectorized_x.joblib")
    length = (len(x.split()) for x in X_train)
    tuples = list((x, y, z) for x, y, z in zip(X_train_mybag, Y_train, length))
    tuples.sort(key=lambda y: y[2])

    slices = {}
    for t in tuples:
        slice = t[2] % 5
        if slice not in slices.keys():
            slices[slice] = []
        slices[slice].append((t[0], t[1]))

    model = OneVsRestClassifier(LogisticRegression(penalty='l1', C=1, dual=False, solver='liblinear'))
    tags_counts = joblib.load(output_directory + "/tags_counts.joblib")
    mlb = MultiLabelBinarizer(classes=sorted(tags_counts.keys()))
    Y_val = mlb.fit_transform(Y_val)

    for key in slices.keys():
        x_slice = []
        for x in slices[key]:
            x_slice.append(x[0].toarray())
        x_slice = np.stack(x_slice, axis=0)

        y_slice = []
        for y in slices[key]:
            y_slice.append(y[1])
        y_slice = mlb.fit_transform(y_slice)

        nsamples, nx, ny = x_slice.shape
        x_slice = x_slice.reshape((nsamples, nx * ny))

        slices[key] = [x_slice, y_slice]

    lib.data_slices(model, slices, X_val_mybag, Y_val)


def _create_data_segment(X, Y):
    three_fourth_x = math.floor(X.shape[0] * 0.75)
    three_fourth_y = math.floor(len(Y) * 0.75)
    old_x = X[:three_fourth_x]
    # new_train_x = X_train_mybag[three_fourth_x_train:]
    old_y = Y[:three_fourth_y]
    # new_train_y = Y_train[three_fourth_y_train:]
    return old_x, old_y


def _create_metrics(estimator, tunable_parameters, X_train, Y_train, X_val, Y_val):
    # Train model for new set
    grid_new = GridSearchCV(estimator=estimator, param_grid=tunable_parameters)
    grid_new.fit(X_train, Y_train)
    y_pred_new = grid_new.predict(X_val)

    metrics = {}
    f1_new = f1_score(Y_val, y_pred_new, average='samples')
    metrics["F1"] = f1_new
    acc_new = accuracy_score(Y_val, y_pred_new)
    metrics["ACC"] = acc_new
    roc_auc_new = roc_auc_score(Y_val, y_pred_new)
    metrics["ROC_AUC"] = roc_auc_new
    aps_new = average_precision_score(Y_val, y_pred_new)
    metrics["AP"] = aps_new

    return metrics


def test_model_staleness():
    X_train_mybag, _, X_val_mybag, _ = joblib.load(output_directory + "/vectorized_x.joblib")
    Y_train, Y_val = joblib.load(output_directory + "/y_preprocessed.joblib")

    tags_counts = joblib.load(output_directory + "/tags_counts.joblib")
    mlb = MultiLabelBinarizer(classes=sorted(tags_counts.keys()))
    Y_train = mlb.fit_transform(Y_train)
    Y_val = mlb.fit_transform(Y_val)

    old_train_x, old_train_y = _create_data_segment(X_train_mybag, Y_train)
    old_test_x, old_test_y = _create_data_segment(X_val_mybag, Y_val)

    # Train model for old set
    model = OneVsRestClassifier(LogisticRegression())
    tunable_parameters = {
        "estimator__penalty": [None, 'l2'],
        "estimator__C": [0.01, 0.1, 1.0],
    }

    old_model_metrics = _create_metrics(model, tunable_parameters, old_train_x, old_train_y, old_test_x, old_test_y)
    new_model_metrics = _create_metrics(model, tunable_parameters, X_train_mybag, Y_train, X_val_mybag, Y_val)

    lib.model_staleness(new_model_metrics, old_model_metrics)
