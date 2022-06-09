"""
    ML Test library functions for features and data.
    Based on section 2 of the paper referenced below.

    Eric Breck, Shanqing Cai, Eric Nielsen, Michael Salib, D. Sculley (2016). What’s your ML test score? A rubric for ML production systems. Reliable Machine Learning in the Wild - NIPS 2016 Workshop (2016).
    Available: https://storage.googleapis.com/pub-tools-public-publication-data/pdf/45742.pdf
"""
import collections
import numpy as np


def no_unsuitable_features(used_features, unsuitable_features):
    """
    Compares the list of used features to the list of unsuitable features. The size of the intersection should be 0.
    :param used_features: List of used features (list of strings)
    :param unsuitable_features: List of unsuitable features (list of strings)
    Also works if both unused_features and unsuitable_features are lists of integers
    (like index representations of features).
    """
    illegal_features = [f for f in used_features if f in unsuitable_features]
    assert len(illegal_features) == 0


def feature_target_correlations(dataset, target, sample_size=10000):
    """"
    Calculates the correlation of each individual feature with the target.
    A sample of the points is taken for speedup.
    :param dataset: A numpy array of shape (#datapoints, #features)
    :param target: A numpy array of shape (#datapoints)
    :param sample_size: Size of the sample. Default is 10.000. Set to dataset.shape[0] to use all data.
    """
    n, f = dataset.shape

    # Assure that sample_size is not too big -> out of bounds
    if n < sample_size:
        sample_size = n

    correlations = np.corrcoef(np.transpose(dataset[:sample_size].toarray()), target[:sample_size])

    # None of the correlations should be exactly 0
    assert np.all(correlations), "At least one feature has 0 correlation with the target. " \
                                 "Perhaps increasing the sample_size solves the issue."


def pairwise_feature_correlations(dataset, sample_size=10000, feature_sample=5):
    """"
    Calculates the correlation of each pair of features.
    Takes a sample of the points and a sample of the features for speedup.
    :param dataset: A numpy array of shape (#datapoints, #features)
    :param sample_size: Size of the sample. Default is 10.000. Set to dataset.shape[0] to use all data.
    :param feature_sample: Size of sample of features. Default is 5. Set to dataset.shape[1] to use all data.
    """
    n, f = dataset.shape

    # Assure that sample_size is not too big -> out of bounds
    if n < sample_size:
        sample_size = n
    if f < feature_sample:
        feature_sample = f

    correlations = np.corrcoef(np.transpose(dataset[:sample_size, :feature_sample].toarray()),
                               np.transpose(dataset[:sample_size, :feature_sample].toarray()))

    # Matrix is 4 concatenations of the matrix of interest, chop off
    correlations = correlations[:feature_sample, :feature_sample]
    # Delete 1.0 from the diagonal (because correlations with itself is always 1.0, we are not interested in that)
    correlations -= np.eye(correlations.shape[0])

    assert 1.0000000 not in correlations, "At least one pair of features has perfect correlation. " \
                                          "Perhaps increasing the sample_size solves the issue." \
                                          "Feature pairs with perfect correlation:" \
                                          f"{list(zip(np.where(correlations == 1.0)[0], np.where(correlations == 1.0)[1]))}"


def preprocessing_validation(examples, answers, preprocess_function, equals=lambda a, b: a == b):
    """
    Asserts that preprocessing works by assessing it on some examples with a known answer.
    :param examples: List of examples to preprocess (of which the answers are known).
    :param answers: List of correctly preprocessed data, corresponding to the examples on each index.
    :param preprocess_function: The data preprocessing function under inspection
    :param equals: Equality function to compare the results to the answers (per element)
    """
    for ex, ans in zip(examples, answers):
        assert equals(preprocess_function(ex), ans), f"Preprocessing went wrong for {ex}"


def feature_values(dataset, feature_column_id, expected_values):
    """
    TODO ziad
    """
    arr = dataset[:, feature_column_id].toarray().reshape(-1)
    for i in set(arr):
        assert i in expected_values


def top_feature_values(dataset, feature_column_id, expected_values, topK=2, at_least_top_k_account_for=0.5):
    """
    TODO ziad
    """
    arr = dataset[:, feature_column_id].toarray().reshape(-1)
    n_data = len(arr)
    features_distribution = collections.Counter(arr).most_common()
    top_features = features_distribution[:topK]
    summation = 0
    for l, r in top_features:
        print(l, r / n_data)
        assert l in expected_values
        summation += r
    assert summation / n_data >= at_least_top_k_account_for
