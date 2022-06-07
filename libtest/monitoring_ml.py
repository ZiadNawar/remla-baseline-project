"""
    ML Test library functions for monitoring ML.
    Based on section 5 of the paper referenced below.

    Eric Breck, Shanqing Cai, Eric Nielsen, Michael Salib, D. Sculley (2016). What’s your ML test score? A rubric for ML production systems. Reliable Machine Learning in the Wild - NIPS 2016 Workshop (2016).
    Available: https://storage.googleapis.com/pub-tools-public-publication-data/pdf/45742.pdf
"""


# todo add more

def data_invariants(feature_invariants_serving, value_range):
    """
        Checks that the feature invariants for the server model are within the same range as those in the training model
        :params feature_invariants_serving: list of feature invariants each with a list of feature invariants values for all datapoints
        :params value_range: list of tuples with the ranges for each feature invariant
    """

    for i, feature in enumerate(feature_invariants_serving):
        for data in feature:
            assert value_range[i][0] <= data <= value_range[i][1]


