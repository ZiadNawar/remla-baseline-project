"""
    ML Testing the StackOverflow label predictor for monitoring ML. Making use of the mltest library.
"""
import libtest.monitoring_ml as lib

from src.text_preprocessing import read_data, data_directory
from src.serve_model import vectorize_instance
import joblib

output_directory = "output"


def test_feature_in_serving():
    train = read_data(data_directory + '/train.tsv')
    X_train = train['title'].values

    words_counts = joblib.load(output_directory + "/words_counts.joblib")
    tfidf_vectorizer = joblib.load(output_directory + '/tfidf_vectorizer.joblib')

    X_train_mybag, X_train_tfidf, _, _ = joblib.load(output_directory + "/vectorized_x.joblib")

    n = 100
    lib.compare_train_embedding_to_serve_embedding(X_train[:n], X_train_mybag[:n],
                                                   lambda x: vectorize_instance(x, words_counts,
                                                                                tfidf_vectorizer)[0])
    lib.compare_train_embedding_to_serve_embedding(X_train[:n], X_train_tfidf[:n],
                                                   lambda x: vectorize_instance(x, words_counts,
                                                                                tfidf_vectorizer)[1])


def test_data_invariants():
    X_train, _, _ = joblib.load("output/X_preprocessed.joblib")

    min = 100
    max = 0
    input_lengths = []
    for x in X_train:
        length = len(x.split())
        if length < min:
            min = length
        if length > max:
            max = length
        input_lengths.append(length)

    lib.data_invariants([input_lengths], [(min, max)])


def test_nan_infinity():
    X_train_mybag, X_train_tfidf, _, _ = joblib.load("../output/vectorized_x.joblib")

    for x in X_train_mybag:
        x = x.toarray()
        lib.nan_infinity(x)

    for x in X_train_tfidf:
        x = x.toarray()
        lib.nan_infinity(x)
