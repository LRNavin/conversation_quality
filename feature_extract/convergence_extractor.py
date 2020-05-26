from scipy.spatial import distance
import numpy as np

import feature_extract.synchrony_extractor as sync_extractor
import feature_extract.mimicry_model as mimicry_model

def get_distance_between(individual1_acc, individual2_acc, metric='euc'):
    distance_i = 0
    if metric == 'euc':
        distance_i = distance.euclidean(individual1_acc, individual2_acc)
    elif metric == 'city':
        distance_i = distance.cityblock(individual1_acc, individual2_acc)
    elif metric == 'cosine':
        distance_i = distance.cosine(individual1_acc, individual2_acc)
    else:
        #l1-norm , same as city ...
        distance_i = np.sum(np.abs(individual1_acc - individual2_acc))

    return distance_i

def get_correlation_with_time(distance_array):
    '''
    Correlation returned is expected to be more negative for converging interactions (-vely correlated with Time),
    meaning that the participants tend to show similar behavior over time
    '''
    return sync_extractor.extract_correlation_between(distance_array, np.arange(len(distance_array)))

def get_symmetric_convergence_between(individual1, individual2, metric='euc'):
    '''
    Correlation returned is expected to be more negative for converging interactions (-vely correlated with Time),
    meaning that the participants tend to show similar behavior over time
    '''
    sym_convergence = []
    for i in range(0, individual1.shape[1]): #i.e For each channel of Person 1 and 2
        indiv1_sample, indiv2_sample = individual1[:,i], individual2[:,i]
        squared_distance = (indiv1_sample-indiv2_sample)**2
        # Correlation between Time and Evolving Distance ->
        sym_convergence.append(get_correlation_with_time(squared_distance))

    return sym_convergence


def get_asymmetric_convergence_between(individual1, individual2, learning_period = 2/3, metric='euc'):
    '''
        Correlation returned is expected to be more negative for converging interactions (-vely correlated with Time),
        meaning that the participants tend to show similar behavior over time

        Currently - Mix Gauss Implemented
    '''
    sym_convergence = []
    for i in range(0, individual1.shape[1]):  # i.e For each channel of Person 1 and 2
        indiv1_sample, indiv2_sample = individual1[:,i], individual2[:,i]
        # Split Samples into equal halves @ int(len/2)
        splitter = int(len(indiv1_sample)*learning_period)
        # model = mimicry_model.learn_mixuture_gaussian_model(indiv1_sample[:splitter])
        distance_array = mimicry_model.get_log_likelihood_features_in_model(model=None, individual1_acc=indiv1_sample[:splitter,],
                                                                            individual2_acc=indiv2_sample[splitter:])
        sym_convergence.append(get_correlation_with_time(distance_array))

    return sym_convergence

def get_global_convergence_between(individual1, individual2):
    '''
    Similarity between both person’s first half’s features are computed using squared differences and saved as d0,
    and similarity between their second half’s features are computed and saved as d1. After that,
    the difference between these similarities is computed by subtraction as: c = d1 − d0
    '''
    # TODO: Check whether usage of euclidean is fine??
    sym_convergence = []
    for i in range(0, individual1.shape[1]):  # i.e For each channel of Person 1 and 2
        indiv1_sample, indiv2_sample = individual1[:,i], individual2[:,i]
        # Split Samples into equal halves @ int(len/2)
        splitter = int(len(indiv1_sample)/2)
        convergence_i = get_distance_between(indiv1_sample[:splitter], indiv2_sample[:splitter]) - get_distance_between(indiv1_sample[splitter:],indiv2_sample[splitter:])
        sym_convergence.append(convergence_i)
    return sym_convergence

def get_features_for(individual1_data, individual2_data, features):
    convergence_features = []
    for feature in features:
        # print("---- Extracting Convergence Features ----> " + feature)
        synchrony_windowed_feature = []
        for window in individual1_data.keys():
            curr_window_data1 = individual1_data[window]
            curr_window_data2 = individual2_data[window]

            if feature == "sym-conv":
                synchrony_windowed_feature.extend(get_symmetric_convergence_between(curr_window_data1, curr_window_data2))
            elif feature == "asym-conv":
                #-1 Convergence of Indiv 2 to Indiv 1
                synchrony_windowed_feature.extend(get_asymmetric_convergence_between(curr_window_data1, curr_window_data2))
                #-1 Convergence of Indiv 1 to Indiv 2 -> Inverted
                synchrony_windowed_feature.extend(get_asymmetric_convergence_between(curr_window_data2, curr_window_data1))
            elif feature == "global-conv":
                synchrony_windowed_feature.extend(get_global_convergence_between(curr_window_data1, curr_window_data2))

            # print(feature + " DONE")

        convergence_features.extend(synchrony_windowed_feature)

    return np.array(convergence_features)

# Main Extractor
def get_convergence_features_for(group_accel_data, features=["sym-conv", "asym-conv", "global-conv"]):
    # Main Extractor
    group_pairwise_features = {}
    members = group_accel_data.keys()
    for member1 in members:
        for member2 in members:
            # No Same Person and Not same pair if already calculated # TODO: Remove this same pair check, Useful for assymetric features
            if member1 != member2: # and (str(member2) + "_" + str(member1) not in group_pairwise_features.keys()):
                # print("For Members - " + str(member1) + " and " + str(member2))
                if len(group_accel_data[member1]) != 0 and len(group_accel_data[member2]) != 0 : #Missing Acc
                    # print("===Convergence Members - " + str(member1) + " and " + str(member2) + " ===")
                    pairwise_features = get_features_for(group_accel_data[member1], group_accel_data[member2], features)
                else:
                    pairwise_features = np.array([])
                group_pairwise_features[str(member1)+"_"+str(member2)] = pairwise_features
    return group_pairwise_features