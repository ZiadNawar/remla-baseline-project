"""
    ML Testing the StackOverflow label predictor for monitoring ML. Making use of the [todo library name] library.
"""
import libtest.monitoring_ml

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
    libtest.monitoring_ml.compare_train_embedding_to_serve_embedding(X_train[:n], X_train_mybag[:n],
                                                                     lambda x: vectorize_instance(x, words_counts,
                                                                                                  tfidf_vectorizer)[0])
    libtest.monitoring_ml.compare_train_embedding_to_serve_embedding(X_train[:n], X_train_tfidf[:n],
                                                                     lambda x: vectorize_instance(x, words_counts,
                                                                                                  tfidf_vectorizer)[1])

# todo add more
