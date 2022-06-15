"""
    ML Test library functions for monitoring ML.
    Based on section 5 of the paper referenced below.

    Eric Breck, Shanqing Cai, Eric Nielsen, Michael Salib, D. Sculley (2016). What’s your ML test score? A rubric for ML production systems. Reliable Machine Learning in the Wild - NIPS 2016 Workshop (2016).
    Available: https://storage.googleapis.com/pub-tools-public-publication-data/pdf/45742.pdf
"""
import numpy as np


def compare_train_embedding_to_serve_embedding(raw_instances, train_embedding, serve_feature_extraction):
    """
    Tests that the feature extraction for model serving works. This is done by embedding data from the train set,
    using the embedding function from the model server, and comparing to the known embedding.
    :param raw_instances: numpy array of raw instances
    :param train_embedding: numpy array of correct embeddings, corresponding to the raw instances
    :param serve_feature_extraction: function that extracts the features for a single query
    """
    for i in range(len(raw_instances)):
        instance = raw_instances[i]
        vectorized_instance = serve_feature_extraction(instance)

        assert np.allclose(train_embedding[i].toarray(), vectorized_instance.toarray())


# todo add more
