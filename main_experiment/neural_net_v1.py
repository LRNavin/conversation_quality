#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Baseline Model

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


# In[2]:


import sys  
sys.path.insert(0, '/Users/navinlr/Desktop/Thesis/code_base/conversation_quality')


# In[3]:


from modeling import dataset_provider as data_gen
import constants

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, accuracy_score, mean_squared_error, roc_auc_score, r2_score, explained_variance_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from imblearn import under_sampling 
from imblearn import over_sampling
from imblearn.over_sampling import SMOTE, ADASYN
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
from tqdm import tqdm


# In[4]:


# Variables for baseline
random_seed=20
manifest="indiv"
data_split_per=.30
missing_data_thresh=50.0 #(in percent)
convq_thresh=3.0
agreeability_thresh=.2
annotators=["Divya", "Nakul"]#, "Swathi"]
only_involved_pairs=True
splits = 5
if manifest=="group":
    smote_nn = 2
else:
    smote_nn = 6

label_type = "hard"
model_type = "baseline_nn"
zero_mean  = False

dataset=constants.features_dataset_path_v1


# In[5]:


# Functions 
def over_sample_data(temp_X, temp_y, method="SMOTE", k_neighbors=6):
    if method == "SMOTE":
        temp_X, temp_y = SMOTE(k_neighbors=k_neighbors-1).fit_resample(temp_X, temp_y)
    return temp_X, temp_y

def feature_normalize(temp_X, method="min-max"):
    # Fit on training set only.
    if method == "min-max":
        normaliser = MinMaxScaler().fit(temp_X)
    elif method == "mean-var":
        normaliser = StandardScaler().fit(temp_X)
    return normaliser
    
def feature_selection(temp_X, temp_y, method="anova"):
    top_features = []
    if method == "anova":
        f_values, p_values = f_classif(temp_X, temp_y)
        top_features=np.where(np.array(p_values) <= 0.05)[0]
#         print(top_features)
        print("# Top Features = " + str(len(top_features)))
    return top_features

def select_required_features(temp_X, required_feats):
    temp_X=temp_X[:,required_feats]
#     print("After Feature Selection, Features -> " + str(temp_X.shape))
    return temp_X

def dimension_reduction(temp_X, method="pca"):
    dim_red_model = None
    if method=="pca":
        dim_red_model = PCA(.95).fit(temp_X)
    return dim_red_model
    
def process_convq_labels(y, label_type="soft"):
    print("Data-type of labels - " + str(type(y)))
    if label_type=="soft":
        y=list(np.around(np.array(y),2))
    else:
        y=list(np.where(np.array(y) <= convq_thresh, 1, 0))
        print("ConvQ Classes Distribution : (Total = "+ str(len(y)) +")")
        print("High Quality Conv = " + str(sum(y)))
        print("Low Quality Conv = " + str(len(y)-sum(y)))
    return y

def model_convq_manifestation(temp_X, temp_y, model="log-reg"):

    if model == "log-reg":
        model = LogisticRegression(solver='lbfgs', max_iter=1000).fit(temp_X, temp_y)
    elif model == "lin-reg":
        model = LinearRegression().fit(temp_X, temp_y)
    elif model == "adaboost":
        model = AdaBoostClassifier(n_estimators=100).fit(temp_X, temp_y)
    elif model == "baseline_nn":
        model = Baseline_NN(temp_X.shape[1])
        print(model)
    return model

def analyse_model_params(model):
    return True

def test_model(temp_X, model):
    return model.predict(temp_X)

def evaluate_predict(predict_temp_y, test_temp_y, method=accuracy_score):
    score = method(test_temp_y, predict_temp_y)
    return score

# baseline model
class Baseline_NN(nn.Module):
    # TODO:  Layers ip, op sizes common variable
    def __init__(self, input_dim):
        # ASsuming input dim (# features) is around > 10K
        super(Baseline_NN, self).__init__()
        print("Input Dimension is " + str(input_dim))
        self.inp_lay = nn.Linear(input_dim, int(input_dim/10))
#         self.inp_lay = nn.Linear(input_dim, int(input_dim*2/10)) 
        self.hiddn_1 = nn.Linear(int(input_dim/10), int(input_dim/100))
#         self.hiddn_1 = nn.Linear(int(input_dim*2/10), int(input_dim*2/100))
        self.hiddn_2 = nn.Linear(int(input_dim/100), int(input_dim/1000))
#         self.hiddn_2 = nn.Linear(int(input_dim*2/100), int(input_dim*2/1000))
#         self.hiddn_3 = nn.Linear(int(input_dim/200), int(input_dim/2000))
        self.out_lay = nn.Linear(int(input_dim/1000), 1) 

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm_il = nn.BatchNorm1d(int(input_dim/10))
#         self.batchnorm_il = nn.BatchNorm1d(int(input_dim*2/10))
        self.batchnorm_h1 = nn.BatchNorm1d(int(input_dim/100))
#         self.batchnorm_h1 = nn.BatchNorm1d(int(input_dim*2/100))
        self.batchnorm_h2 = nn.BatchNorm1d(int(input_dim/1000))
#         self.batchnorm_h2 = nn.BatchNorm1d(int(input_dim*2/1000))
#         self.batchnorm_h3 = nn.BatchNorm1d(int(input_dim/2000))

    def forward(self, inputs):
        # Input Layer 
        x = self.relu(self.inp_lay(inputs))
        x = self.batchnorm_il(x)
        
        x = self.dropout(x)
        
        # Hidden Layer 1
        x = self.relu(self.hiddn_1(x))
        x = self.batchnorm_h1(x)
        
        x = self.dropout(x)
        
        # Hidden Layer 2
        x = self.relu(self.hiddn_2(x))
        x = self.batchnorm_h2(x)
        
        x = self.dropout(x)
        
        # Hidden Layer 3
#         x = self.relu(self.hiddn_3(x))
#         x = self.batchnorm_h3(x)
        
        x = self.dropout(x)

        # Output Layer
        x = self.out_lay(x)
        
        return x
    
## train data formatter
class load_train_data(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)

## test data formatter   
class load_test_data(Dataset):
    def __init__(self, X_data):
        self.X_data = X_data
        
    def __getitem__(self, index):
        return self.X_data[index]
        
    def __len__ (self):
        return len(self.X_data)
        
def nn_dataloader(train_X, train_y, test_X, batch_size):
    # Format Dataset
    train_data = load_train_data(torch.FloatTensor(train_X), torch.FloatTensor(train_y))
    test_data  = load_test_data(torch.FloatTensor(test_X))

    #Convert formatted data to data loader
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(dataset=test_data, batch_size=1)
    return train_loader, test_loader
    
def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)    
    return acc
    
def train_nn_model(model, train_loader, weight, num_epochs):
    print("Model in Training Mode")
    model.train()
    
    # Class weights - Formula = Max(Class-Distburion)/Current_class-Distbution
    class_weight = torch.FloatTensor(weight)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for e in tqdm(range(1, num_epochs+1)):
        epoch_loss = 0
        epoch_acc = 0
#         print(train_loader)
        for data in train_loader:
#             print(data)
            X_batch, y_batch = data
            optimizer.zero_grad()
            y_pred = model(X_batch)
            
#             print(y_pred)
#             print(y_batch.unsqueeze(1))

            loss = criterion(y_pred, y_batch.unsqueeze(1))
            acc  = binary_acc(y_pred, y_batch.unsqueeze(1))

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')
    
    return True

def predict_with_nn(test_loader, model):
    y_pred_list = []
    model.eval()
    with torch.no_grad():
        for X_batch in test_loader:
            y_test_pred = model(X_batch)
            y_test_pred = torch.sigmoid(y_test_pred)
            y_pred_tag  = torch.round(y_test_pred)
            y_pred_list.append(y_pred_tag)
#             print(y_pred_tag)

    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
#     print(y_pred_list)
    return y_pred_list


# In[6]:


# Data Read
X, y, ids = data_gen.get_dataset_for_experiment(dataset=dataset,
                                                    manifest=manifest,
                                                    missing_data_thresh=missing_data_thresh,
                                                    agreeability_thresh=agreeability_thresh,
                                                    annotators=annotators,
                                                    only_involved_pairs=only_involved_pairs,
                                                    zero_mean=zero_mean)

# print(y)


# In[7]:


# Label Prep
# Hard/Soft Labels
y = process_convq_labels(y, label_type)


# In[8]:


# Data Prep
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=data_split_per, random_state=random_seed)

final_conf_matrix = [[0,0],[0,0]]
final_auc_score = 0.0
final_r_squared = 0.0
final_expl_vari = 0.0

# Neural Net variables
batch_size=10
num_epochs=100

# skf = StratifiedKFold(n_splits=splits)
# for train_index, test_index in skf.split(X, y):

# # Data Prep
# train_X, test_X  = X[train_index], X[test_index]
# train_y, test_y  = [y[i] for i in train_index], [y[i] for i in test_index]

# Transform Features
normaliser = feature_normalize(train_X, method="mean-var")
# Apply transform to both the training set and the test set.
train_X = normaliser.transform(train_X)
test_X  = normaliser.transform(test_X)

# SAMPLING
# train_X, train_y = over_sample_data(train_X, train_y, method="SMOTE", k_neighbors=smote_nn)

print("Train Data -> Features - " + str(train_X.shape) + " and Labels - " + str(len(train_y)))
print("Test  Data -> Features - " + str(test_X.shape) + " and Labels - " + str(len(test_y)))
print("Number of Positive (1-LowConvq) Sample = " + str(sum(train_y))) 

# NN Data Loader
train_loader, test_loader = nn_dataloader(train_X, train_y, test_X, batch_size)

# Modelling

weight=[len(y)/sum(y)] # Class Imbalance handler weights. 
print('Weights for Positive (1-LowConvq) Class = ' + str(weight))
model = model_convq_manifestation(train_X, train_y, model_type)
_ = train_nn_model(model, train_loader, weight, num_epochs)

#Predict
predict_y = predict_with_nn(test_loader, model)   

# Evaluate
conf_matrix = evaluate_predict(test_y, predict_y, confusion_matrix)
try:
    auc_score = evaluate_predict(test_y, predict_y, roc_auc_score)
except ValueError:
    auc_score = 0.0
    print("Oops! All Predicitons in same class. Bad Fold... Fold not considered. for AUC")
#     r_squared = evaluate_predict(test_y, predict_y, r2_score)
#     expl_vari = evaluate_predict(test_y, predict_y, explained_variance_score)

print("Current Fold Prediciton Eval...")
print(conf_matrix)
print(auc_score)

#Update Cross Validated scores
final_conf_matrix =  conf_matrix
final_auc_score = auc_score
#     final_r_squared = final_r_squared + r_squared
#     final_expl_vari = final_expl_vari + expl_vari

# final_auc_score = final_auc_score/skf.get_n_splits(X, y)
# final_r_squared = final_r_squared/skf.get_n_splits(X, y)
# final_expl_vari = final_expl_vari/skf.get_n_splits(X, y)


# In[ ]:


# Printing Final Score
# print("~~~~~~~~~~~ R^2 Measure ~~~~~~~~~~~")
# print(final_r_squared)
# print("~~~~~~~~~~~ Explained Variance ~~~~~~~~~~~")
# print(final_expl_vari)
print("~~~~~~~~~~~ Confusion Matrix ~~~~~~~~~~~")
print(final_conf_matrix)
print("~~~~~~~~~~~ AUC Score ~~~~~~~~~~~")
print(final_auc_score)


# In[ ]:




