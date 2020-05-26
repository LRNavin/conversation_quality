from sklearn import mixture
from statistics import mean, variance, median, mode

import numpy as np

def get_aggregate_features_for_mimicry(distance_array, features=[min, max, mean, mode, median, variance]):
    mimicry_features = []
    for feature in features:
        mimicry_features.append(feature(distance_array))
    return mimicry_features

#Nanninga et al. -> IMplemented for Convergence Mixture Gauss based Mimicry,
def learn_mixuture_gaussian_model(individual_acc, n_components=1, covariance_type='full'):
    # TODO: Implement Model selection - USing BIC
    model = mixture.GaussianMixture(n_components=n_components, covariance_type=covariance_type).fit(individual_acc)
    return model

def get_log_likelihood_features_in_model(model, individual1_acc, individual2_acc):
    distance_array=[]
    if model is None:
        model = learn_mixuture_gaussian_model(individual1_acc.reshape(-1, 1), 1, "full")
    distance_array.extend(model.score_samples(individual2_acc.reshape(-1, 1)))
    return distance_array

# Distance Based Mimicry - Based on Oyku's Thesis
def get_mimicry_sq_distance(individual1_acc, individual2_acc, agg_features):
    #From Indiv1's Sample 0 to n-1
    mimicry_features = []
    for channel in range(individual1_acc.shape[1]):
        channel_len = individual1_acc.shape[0]
        curr_channel_acc1, curr_channel_acc2 = individual1_acc[:channel_len-1,channel], individual2_acc[1:,channel]
        current_distance = (curr_channel_acc1-curr_channel_acc2)**2
        mimicry_features.extend(get_aggregate_features_for_mimicry(current_distance, agg_features))
        # print(len(mimicry_features))
    return mimicry_features


def get_asymmetric_mimicry(individual1_acc, individual2_acc, model_type="sq-distance"):
    # print("Calculating Mimicry with Model - " + str(model_type))
    mimicry_features=[]
    if model_type == "sq-distance":
        mimicry_features = get_mimicry_sq_distance(individual1_acc=individual1_acc,
                                                   individual2_acc=individual2_acc,
                                                   agg_features=[min, max, mean, variance])
    elif model_type == "mix-gaussian":
        mimicry_features = get_log_likelihood_features_in_model(model=None, individual1_acc=individual1_acc,
                                                                individual2_acc=individual2_acc)
    return mimicry_features

def get_mimicry_features(individual1_acc, individual2_acc, model_type="sq-distance"):
    # Indv1 Lagged Indv2
    mimicry_features = get_asymmetric_mimicry(individual1_acc, individual2_acc, model_type)
    # Indv2 Lagged Indv1
    mimicry_features.extend(get_asymmetric_mimicry(individual2_acc, individual1_acc, model_type))
    return mimicry_features