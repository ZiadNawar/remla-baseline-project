"""
    ML Testing the StackOverflow label predictor for monitoring ML. Making use of the [todo library name] library.
"""
import libtest.monitoring_ml as lib
import joblib


# todo add more
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



