import scipy.stats as scipy_stat
from pyitlib import discrete_random_variable as drv
from numpy.random import randn
import sklearn.metrics as sklearn_met

import pandas as pd
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
    if norm:
        mi = drv.information_mutual_normalised(individual1_acc, individual2_acc, 'SQRT')
    else:
        mi = drv.information_mutual(individual1_acc, individual2_acc)
    return mi

def get_timelagged_correl_between(individual1_acc, individual2_acc, lag=0):
    '''
    Lag-N Pearson correlation.
    Result: correl (float)
    '''
    print("Calculating cross-correlations with LAG-" + str(lag) + "....")

    mean_x = np.mean(individual1_acc)
    mean_y = np.mean(individual2_acc)
    std_x  = np.std(individual1_acc)
    std_y  = np.std(individual2_acc)

    mean_diff = 0
    counter = 0
    for i in range(0,len(individual1_acc)-lag):
        x_i = individual1_acc[i]
        y_i = individual2_acc[i+lag]
        mean_diff = mean_diff + ((x_i-mean_x)*(y_i-mean_y))
        counter = counter+1
    mean_diff = mean_diff/counter
    lagged_correlation = mean_diff/(std_x*std_y)

    return lagged_correlation


def get_correlation_between(individual1_acc, individual2_acc):
    print("Calculating correlations.....")
    # correlation, p_value = scipy_stat.pearsonr(individual1_acc, individual2_acc)
    correlation = get_timelagged_correl_between(individual1_acc, individual2_acc)
    return correlation
