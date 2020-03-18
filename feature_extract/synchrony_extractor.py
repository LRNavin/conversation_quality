import scipy.stats as scipy_stat
from scipy.spatial import distance
from pyitlib import discrete_random_variable as drv
import sklearn

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
    return get_correlation_between(distance_array, np.arange(len(distance_array)))

def get_symmetric_convergence_between(individual1, individual2, metric='euc'):
    '''
    Correlation returned is expected to be more negative for converging interactions (-vely correlated with Time),
    meaning that the participants tend to show similar behavior over time
    '''
    distance_array = []
    for i in range(0, individual1.shape[0]): #i.e For each sample of Person 1 and 2
        indiv1_sample = individual1[i,:]
        indiv2_sample = individual2[i,:]
        distance_array.append(get_distance_between(indiv1_sample, indiv2_sample, metric))

    # Correlation between Time and Evolving Distance ->
    convergence = get_correlation_with_time(distance_array)
    return convergence


def learn_mixuture_gaussian_model(individual_acc, n_components=2, covariance_type='full'):
    # TODO: Implement Model selection - USing BIC
    model = sklearn.mixture.GaussianMixture(n_components, covariance_type).fit(individual_acc)
    return model

def get_log_likelihood_in_model(model, individual_acc):
    distance_array=[]
    for samples in individual_acc:
        distance_array.append(model.score(samples))
    return distance_array

def get_asymmetric_convergence_between(individual1, individual2, model=None, learning_period = 1, metric='euc'):
    '''
        Correlation returned is expected to be more negative for converging interactions (-vely correlated with Time),
        meaning that the participants tend to show similar behavior over time
    '''
    #TODO : Check Dimension of features and samples
    if model is None:
        model = learn_mixuture_gaussian_model(individual1)
    distance_array = get_log_likelihood_in_model(model, individual2)
    convergence = get_correlation_with_time(distance_array)
    return convergence
