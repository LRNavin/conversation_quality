import numpy as np

def concatenate_pairwise_features(feature_set1, feature_set2):
    concatenated_features = feature_set1.copy()
    for pairs in concatenated_features.keys():
        concatenated_features[pairs] = np.concatenate((feature_set1[pairs], feature_set2[pairs]), axis=0)
    return concatenated_features