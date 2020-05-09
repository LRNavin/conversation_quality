from dataset_creation import dataset_creator as data_generator
from feature_extract import group_features_extractor as group_feat_extractor
import constants as const

import numpy as np
from tqdm import tqdm

# Group-Level Dataset Generation
def generate_aggregated_group_features_dataset(dataset, missing_data_thresh, agreeability_thresh, annotators, zero_mean):

    filtered_dataset, convq_ids, convq_scores = data_generator.filter_dataset(dataset,
                                                                              missing_data_thresh, agreeability_thresh,
                                                                              "group", annotators, True, zero_mean)
    print("Dataset for modeling ConvQ Generated !!!!!!")

    X, Y = [], []
    for i, group_id in tqdm(enumerate(filtered_dataset.keys())): # Use filtered Dataset as already filtered for reliability and missings
        x = group_feat_extractor.aggregate_to_group_features(pairwise_features=filtered_dataset[group_id])
        if len(X) == 0:
            X = x
        else:
            X = np.vstack((X,x))
        Y.append(convq_scores[i])

    return X, np.array(Y), convq_ids

# Indiv-Level Dataset Generation
def generate_aggregated_indiv_features_dataset(dataset, missing_data_thresh, agreeability_thresh, annotators, only_involved_pairs, zero_mean):

    filtered_dataset, convq_ids, convq_scores = data_generator.filter_dataset(dataset,
                                                                              missing_data_thresh, agreeability_thresh,
                                                                              "indiv", annotators, only_involved_pairs, zero_mean)
    X, Y = [], []
    for i, group_indiv_id in tqdm(enumerate(filtered_dataset.keys())): # Use filtered Dataset as already filtered for reliability and missings
        x = group_feat_extractor.aggregate_to_group_features(pairwise_features=filtered_dataset[group_indiv_id])
        if len(X) == 0:
            X = x
        else:
            X = np.vstack((X,x))
        Y.append(convq_scores[i])

    return X, Y, convq_ids



# Public Function - Receive external requests
# 1. Statistical Aggreagtion Based Dataset generation
def get_dataset_for_experiment(dataset, manifest, missing_data_thresh, agreeability_thresh, annotators, only_involved_pairs, zero_mean):
    print("Generating Dataset for modeling - " + manifest + " ConvQ, ...........")
    if manifest == "group":
        return generate_aggregated_group_features_dataset(dataset, missing_data_thresh, agreeability_thresh, annotators, zero_mean)
    else:
        return generate_aggregated_indiv_features_dataset(dataset, missing_data_thresh, agreeability_thresh, annotators, only_involved_pairs, zero_mean)