from pyitlib import discrete_random_variable as drv
import feature_extract.mimicry_model as mimicry_extractor

import numpy as np

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

    print("Calculating mutual-info.....")
    mi=[]
    for segment in range(individual1_acc.shape[1]):
        indiv1_acc = individual1_acc[:,segment]
        indiv2_acc = individual2_acc[:,segment]
        if norm:
            curr_mi = drv.information_mutual_normalised(indiv1_acc, indiv2_acc, 'SQRT')
        else:
            curr_mi = drv.information_mutual(indiv1_acc, indiv2_acc)
        mi.append(curr_mi)

    return mi

def get_timelagged_correlation_between(individual1_acc, individual2_acc, inverted=False, lag=0):
    '''
    Lag-N Pearson correlation.
    Result: correl (float)
    '''

    #Below Segmentation is kinda expensive and not efficient. But sticking to it for segment-wise tests/readability
    if inverted:
        print("Calculating correlations with LAG- -" + str(lag) + "....")
    else:
        print("Calculating correlations with LAG- " + str(lag) + "....")

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
        print("---- Extracting Synchrony Features ----> " + feature)
        synchrony_windowed_feature = []
        for window in individual1_data.keys():
            curr_window_data1 = individual1_data[window]
            curr_window_data2 = individual2_data[window]

            if feature == "correl":
                synchrony_windowed_feature.extend(get_correlation_between(curr_window_data1, curr_window_data2))
            elif feature == "lag-correl":
                #+1 Lag -> Lag to INDIV 2 i.e Test for Indiv 2 as follower and Indiv 1 as driver
                synchrony_windowed_feature.extend(get_timelagged_correlation_between(curr_window_data1, curr_window_data2, inverted=False, lag=1))
                #-1 Lag -> Inverted
                synchrony_windowed_feature.extend(get_timelagged_correlation_between(curr_window_data2, curr_window_data1, inverted=True, lag=1))
            elif feature == "mi":
                synchrony_windowed_feature.extend(get_mutual_info_between(curr_window_data1, curr_window_data2, norm=False))
            elif feature == "norm-mi":
                synchrony_windowed_feature.extend(get_mutual_info_between(curr_window_data1, curr_window_data2, norm=True))
            elif feature == "mimicry":
                synchrony_windowed_feature.extend(mimicry_extractor.get_mimicry_features(curr_window_data1, curr_window_data2, model_type="sq-distance"))

        synchrony_features.extend(synchrony_windowed_feature)

    return np.array(synchrony_features)

# Main Extractor
def get_synchrony_features_for(group_accel_data, features=["correl", "lag-correl", "mi", "norm-mi", "mimicry"]):
    group_pairwise_features = {}
    members = group_accel_data.keys()
    for member1 in members:
        for member2 in members:
            # No Same Person and Not same pair if already calculated
            if member1 != member2 and (str(member2)+"_"+str(member1) not in group_pairwise_features.keys()):
                print("For Members - " + str(member1) + " and " + str(member2))
                pairwise_features = get_features_for(group_accel_data[member1], group_accel_data[member2], features)
                print("Pairwise-Feature Shape = " + str(pairwise_features.shape))
                group_pairwise_features[str(member1)+"_"+str(member2)] = pairwise_features
    # print("# Pairwise Features = " + str(len(group_pairwise_features.keys())))
    # print("Pairs -> " + str(group_pairwise_features.keys()))

    return group_pairwise_features

