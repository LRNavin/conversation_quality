from pyitlib import discrete_random_variable as drv
from feature_extract.MutualInformation import MutualInformation # SyncPy model
import feature_extract.mimicry_model as mimicry_extractor
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_mutual_info_score

import numpy as np
import pandas as pd

def get_syncpy_mutual_info_between(individual1_acc, individual2_acc):
    mi = []
    for segment in range(individual1_acc.shape[1]):
        indiv1_acc = individual1_acc[:, segment]
        indiv2_acc = individual2_acc[:, segment]
        curr_mi = MutualInformation(n_neighbours=10).compute([pd.DataFrame(indiv1_acc), pd.DataFrame(indiv2_acc)])["MI"]
        print("SyncPy = " + str(curr_mi))
        mi.append(curr_mi)
    return mi


def get_mutual_info_between(individual1_acc, individual2_acc, norm=True):
    '''
    Other methods tried
        # indiv1_entropy = scipy_stat.entropy(individual1_acc)
        # indiv2_entropy = scipy_stat.entropy(individual2_acc)
        # joint_entropy = scipy_stat.entropy(individual1_acc, individual2_acc)
        #
        # print(indiv1_entropy)
        # print(indiv2_entropy)
        # print((indiv1_entropy + indiv2_entropy - joint_entropy) / (np.sqrt((indiv1_entropy * indiv2_entropy))))
        # print((joint_entropy - indiv2_entropy) / (np.sqrt((indiv1_entropy * indiv2_entropy))))
        #
        # print(sklearn_met.mutual_info_score(individual1_acc, individual2_acc))
        # print(drv.information_mutual(individual1_acc, individual2_acc))
    '''

    # print("Calculating mutual-info.....")
    mi=[]
    for segment in range(individual1_acc.shape[1]):
        indiv1_acc = individual1_acc[:,segment]
        indiv2_acc = individual2_acc[:,segment]
        if norm:
            curr_mi = drv.information_mutual_normalised(indiv1_acc, indiv2_acc, 'SQRT')
            print("NORM MI-----")
        else:
            curr_mi = drv.information_mutual(indiv1_acc, indiv2_acc)
            print("MI-----")
        print(curr_mi)
        mi.append(curr_mi)
    return mi

def get_min_max_from_lagged_correlations(individual1_acc, individual2_acc, agg_fn=max):

    # HERE Segments are -> Different channels
    # If not Windowed (e.g. x,y,z raw acc channels or magnitude channel)
    # If Windowed (e.g. windowed-mean or windowed-var basic features of respective channels)
    lagged_correlation=[]
    for segment in range(individual1_acc.shape[1]):
        indiv1_acc = individual1_acc[:, segment]
        indiv2_acc = individual2_acc[:, segment]

        # print(str(agg_fn) + " values in Correlation ->")
        nx = np.linalg.norm(indiv1_acc, 2)
        ny = np.linalg.norm(indiv2_acc, 2)
        # Scaled (/(nx*ny)) Full XCorrel
        full_correl = np.correlate(indiv1_acc, indiv2_acc, "full")/(nx*ny)
        # print(agg_fn(full_correl))
        # Get the Max/Min of all possible lagged values (Gets the Correl-coeff values @ best lagged posisition)
        lagged_correlation.append(agg_fn(full_correl))

    return lagged_correlation

def get_timelagged_correlation_between(individual1_acc, individual2_acc, inverted=False, lag=0):
    '''
    Lag-N Pearson correlation.
    Result: correl (float)
    '''

    #Below Segmentation is kinda expensive and not efficient. But sticking to it for segment-wise tests/readability
    # if inverted:
    #     print("Calculating correlations with LAG- -" + str(lag) + "....")
    # else:
    #     print("Calculating correlations with LAG- " + str(lag) + "....")

    # HERE Segments are -> Different channels
    # If not Windowed (e.g. x,y,z raw acc channels or magnitude channel)
    # If Windowed (e.g. windowed-mean or windowed-var basic features of respective channels)

    lagged_correlation=[]
    for segment in range(individual1_acc.shape[1]):
        indiv1_acc = individual1_acc[:, segment]
        indiv2_acc = individual2_acc[:, segment]

        curr_lagged_correlation = extract_correlation_between(indiv1_acc, indiv2_acc, lag)
        lagged_correlation.append(curr_lagged_correlation)

    return lagged_correlation

def extract_correlation_between(indiv1_acc, indiv2_acc, lag=0):
    mean_x = np.mean(indiv1_acc)
    mean_y = np.mean(indiv2_acc)
    std_x = np.std(indiv1_acc)
    std_y = np.std(indiv2_acc)

    mean_diff = 0
    counter = 0
    for i in range(0, len(indiv1_acc) - lag):
        x_i = indiv1_acc[i]
        y_i = indiv2_acc[i + lag]
        mean_diff = mean_diff + ((x_i - mean_x) * (y_i - mean_y))
        counter = counter + 1
    mean_diff = mean_diff / counter

    curr_lagged_correlation = mean_diff / (std_x * std_y)
    return curr_lagged_correlation

def get_correlation_between(individual1_acc, individual2_acc):
    # print("Calculating correlations.....")
    # correlation, p_value = scipy_stat.pearsonr(individual1_acc, individual2_acc)
    correlation = get_timelagged_correlation_between(individual1_acc, individual2_acc)
    return correlation

def get_features_for(individual1_data, individual2_data, features):
    synchrony_features = []
    for feature in features:
        synchrony_windowed_feature = []
        for window in individual1_data.keys():
            # print("Current Window -> " + str(window))
            curr_window_data1 = individual1_data[window]
            curr_window_data2 = individual2_data[window]
            # print("Member Shape -> " + str(curr_window_data1.shape))
            # print("Member Shape -> " + str(curr_window_data2.shape))

            if feature == "correl":
                synchrony_windowed_feature.extend(get_correlation_between(curr_window_data1, curr_window_data2))
            elif feature == "lag-correl":
                # Min correlation-coeff @ n th lag index (Max "positive" Correl between 2 person)
                synchrony_windowed_feature.extend(get_min_max_from_lagged_correlations(curr_window_data1,
                                                                                       curr_window_data2, min))
                # Max correlation-coeff @ n th lag index (Max "negative" Correl between 2 person 'or' Min "positive" Correl between 2 person)
                synchrony_windowed_feature.extend(get_min_max_from_lagged_correlations(curr_window_data1,
                                                                                       curr_window_data2, max))
                # INDEX of Lag, where correlation-coeff is MIN (How much lag is required for max coordination)
                synchrony_windowed_feature.extend(get_min_max_from_lagged_correlations(curr_window_data1,
                                                                                       curr_window_data2, np.argmin))
                # INDEX of Lag, where correlation-coeff is MAX (How much lag is required for min coordination)
                synchrony_windowed_feature.extend(get_min_max_from_lagged_correlations(curr_window_data1,
                                                                                       curr_window_data2, np.argmax))
                #+1 Lag -> Lag to INDIV 2 i.e Test for Indiv 2 as follower and Indiv 1 as driver
                # synchrony_windowed_feature.extend(get_timelagged_correlation_between(curr_window_data1, curr_window_data2, inverted=False, lag=1))
                #-1 Lag -> Inverted
                # synchrony_windowed_feature.extend(get_timelagged_correlation_between(curr_window_data2, curr_window_data1, inverted=True, lag=1))
            elif feature == "mi":
                # synchrony_windowed_feature.extend(get_mutual_info_between(curr_window_data1, curr_window_data2, norm=False))
                synchrony_windowed_feature.extend(get_syncpy_mutual_info_between(curr_window_data1, curr_window_data2))
            elif feature == "norm-mi":
                # synchrony_windowed_feature.extend(get_mutual_info_between(curr_window_data1, curr_window_data2, norm=True))
                continue
            elif feature == "mimicry":
                synchrony_windowed_feature.extend(mimicry_extractor.get_mimicry_features(curr_window_data1, curr_window_data2, model_type="sq-distance"))

            # print(feature + " DONE")

        synchrony_features.extend(synchrony_windowed_feature)

    return np.array(synchrony_features)

# Main Extractor
def get_synchrony_features_for(group_accel_data, features=["correl", "lag-correl", "mi", "norm-mi", "mimicry"]):
    group_pairwise_features = {}
    members = group_accel_data.keys()
    for member1 in members:
        for member2 in members:
            # No Same Person and Not same pair if already calculated # TODO: Remove this same pair check, Useful for assymetric features (DONE)
            if member1 != member2: #and (str(member2)+"_"+str(member1) not in group_pairwise_features.keys()):
                if len(group_accel_data[member1]) != 0 and len(group_accel_data[member2]) != 0 : #Missing Acc
                    # print("===Synchrony Members - " + str(member1) + " and " + str(member2) + " ===")
                    pairwise_features = get_features_for(group_accel_data[member1], group_accel_data[member2], features)
                else:
                    pairwise_features = np.array([])
                group_pairwise_features[str(member1)+"_"+str(member2)] = pairwise_features
    return group_pairwise_features

