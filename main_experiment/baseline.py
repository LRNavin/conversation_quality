from modeling import dataset_provider as data_gen

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve

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
manifest="group"
data_split_per=.30

X, y = data_gen.get_dataset_for_experiment(manifest=manifest)
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=data_split_per, random_state=random_seed)

def feature_selection(method="pca"):
    return None

def model_convq_manifestation(model="log-reg"):

    if model == "log-reg":
        model = LogisticRegression().fit(train_X, train_y)
    elif model == "lin-reg":
        model = LinearRegression().fit(train_X, train_y)

    return model

def analyse_model_params(model):
    return True

def test_model(model):
    return model.predict(test_X)

def evaluate_predict(predict_y, method=[confusion_matrix, roc_curve]):
    for metric in method:
        print("Eval Metric - " + str(metric))
        score = metric(test_y, predict_y)
        print(score)
    return True