from modeling import dataset_provider as data_gen
import constants

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, accuracy_score, mean_squared_error

import numpy as np
import pandas as pd

# Model - Logistic Reg,
# Features -
#     * All Channels - Raw, Abs, Mag (8)
#     * All Windows  - 1, 3, 5, 10, 15
#     * All Indiv    - Stat - Mean, Variance, Spec - PSD 6 bins
#     * All Pairwise -
#            - Synch - Correl, lag-Correl, MI, mimicry
#            - Convr - Sym.Conv, Asym.Conv, Glob.Conv
#     * All GroupFeat-
#            - Aggreagtion - Min, Max, Mean, Mode, Var
#            -
# Evaluation - Acc, Conf.Matrix, AUC, Precision, Recall,

# Variables for baseline
random_seed=44
manifest="indiv"
data_split_per=.30
missing_data_thresh=50.0 #(in percent)
agreeability_thresh=.2
annotators=["Divya", "Nakul"]#, "Swathi"]
only_involved_pairs=True
zero_mean=True

def feature_selection(method="pca"):
    return None

def model_convq_manifestation(train_X, train_y, model="log-reg"):

    if model == "log-reg":
        model = LogisticRegression().fit(train_X, train_y)
    elif model == "lin-reg":
        model = LinearRegression().fit(train_X, train_y)

    return model

def analyse_model_params(model):
    return True

def test_model(test_X, model):
    return model.predict(test_X)

def evaluate_predict(predict_y, test_y, method=[accuracy_score, confusion_matrix, roc_curve]):
    for metric in method:
        print("Eval Metric - " + str(metric))
        score = metric(test_y, predict_y)
        print(score)
    return True


def baseline_experiment_trigger(dataset):

    X, y, ids = data_gen.get_dataset_for_experiment(dataset=dataset,
                                                    manifest=manifest,
                                                    missing_data_thresh=missing_data_thresh,
                                                    agreeability_thresh=agreeability_thresh,
                                                    annotators=annotators, only_involved_pairs=only_involved_pairs, zero_mean=zero_mean)
    print(X.shape)
    # print(X)
    print(len(y))
    # print(y)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=data_split_per, random_state=random_seed)

    model = model_convq_manifestation(train_X, train_y, "lin-reg")
    predict_y = test_model(test_X, model)
    evaluate_predict(predict_y, test_y, [mean_squared_error])


baseline_experiment_trigger(dataset=constants.features_dataset_path_v1)