"""
    ML Testing the StackOverflow label predictor for the ML infrastructure. Making use of the mltest library.
"""

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from libtest.ml_infrastructure import reproducibility_training, improved_model_quality, integration_test
from src.text_preprocessing import main as preprocess_main
from src.vectorization import main as vectorization_main
from src.model_training import main as training_main
from src.evaluation import main as eval_main


def test_reproducibility_training_specific():
    print("Reproducibility Training example")
    X_train_mybag, X_train_tfidf, X_val_mybag, X_val_tfidf = joblib.load("output/vectorized_x.joblib")
    y_train, y_val = joblib.load("output/y_preprocessed.joblib")
    tags_counts = joblib.load("output/tags_counts.joblib")

    mlb = MultiLabelBinarizer(classes=sorted(tags_counts.keys()))
    y_train = mlb.fit_transform(y_train)
    y_val = mlb.fit_transform(y_val)

    clf = LogisticRegression(penalty='l1', C=1, dual=False, solver='liblinear')
    clf = OneVsRestClassifier(clf)

    reproducibility_training(clf, X_train_mybag, y_train, X_val_mybag, y_val)


def test_model_quality():
    X_train_mybag, X_train_tfidf, X_val_mybag, X_val_tfidf = joblib.load("output/vectorized_x.joblib")
    y_train, y_val = joblib.load("output/y_preprocessed.joblib")
    tags_counts = joblib.load("output/tags_counts.joblib")

    mlb = MultiLabelBinarizer(classes=sorted(tags_counts.keys()))
    y_train = mlb.fit_transform(y_train)
    y_val = mlb.fit_transform(y_val)

    m_1 = OneVsRestClassifier(LogisticRegression(penalty='l2', C=1, dual=False, solver='liblinear')).fit(X_train_mybag,
                                                                                                         y_train)
    m_2 = OneVsRestClassifier(LogisticRegression(penalty='l1', C=1, dual=False, solver='liblinear')).fit(X_train_mybag,
                                                                                                         y_train)

    improved_model_quality(m_1, m_2, X_val_mybag, y_val)


def test_integration():
    pipeline = [preprocess_main, vectorization_main, training_main, eval_main]
    integration_test(pipeline)
