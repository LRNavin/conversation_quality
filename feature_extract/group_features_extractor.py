import numpy as np
from scipy import stats

def aggregate_to_group_features(pairwise_features, agg_features=[np.min, np.max, np.mean, np.var, np.median, stats.mode]):
    group_level_features = []
    concatenated_features = []

    # Concatenate Pairwise features to 1 matrix
    for i, pair in enumerate(pairwise_features.keys()):
        pair_feature = pairwise_features[pair].reshape((1,len(pairwise_features[pair])))
        if len(concatenated_features) == 0:
            concatenated_features = pair_feature
        else:
            concatenated_features = np.concatenate((concatenated_features, pair_feature), axis=0)
    # print(concatenated_features)
    #Compute aggregate features
    for feature in agg_features:
        if feature == stats.mode:
            agg_feat = feature(concatenated_features, axis=0).mode[0]
        else:
            agg_feat = feature(concatenated_features, axis=0)
        group_level_features.extend(agg_feat)

    return np.array(group_level_features)