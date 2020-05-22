import numpy as np
from scipy import stats
from distutils.dir_util import copy_tree
import shutil
import constants as const
import pandas as pd
import pickle
import csv

import utilities.data_read_util as reader
import feature_extract.feature_preproc as processor
import feature_extract.statistics_extractor as base_feat_extractor
import feature_extract.synchrony_extractor as sync_extractor
import feature_extract.convergence_extractor as conv_extractor
import feature_extract.features_postproc as post_processor
import feature_extract.group_features_extractor as group_feat_extractor

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

do_missing_stat = False


if do_missing_stat:
    missign_acc_checklist = pd.read_csv(const.missing_acc_stat)

    sns.distplot(missign_acc_checklist.percent_miss, bins=10, kde=False, rug=True).set_title('Percentage of Missing Accelero Data. (#MissingIndiv/#IndivInGroup)')
    plt.show()

    group_size = np.unique(missign_acc_checklist.member_count.values)
    print(group_size)

    f, axes = plt.subplots(3, 2)
    for i, size in enumerate(group_size):
        size_missing_stat = missign_acc_checklist.loc[missign_acc_checklist.member_count == size].percent_miss
        sns.distplot(size_missing_stat, bins=4, kde=False, rug=True, ax=axes[int(i/2)][int(i%2)], axlabel="").set_title("Group Size:"+str(size), fontsize=8)

    f.suptitle("Missing Data Statistics")
    plt.show()

# with open(const.temp_fform_store, 'rb') as data_store:
#     pd.options.display.max_columns = None
#     groups = pickle.load(data_store)
#     print(groups["Day2"][:2])

# TODO :TEST FEATURES

group_id = "1_018"
members = reader.get_members_in_f_form(group_id)

print("~~~~~~~~~~~~~~~~~~~~~~~~~ FOR GROUP - " + str(group_id) + " ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
# Reading Accelerometer Channels - RAW, ABS, MAG
group_acc = processor.get_accelerometer_channels_for(group_id=group_id, channels=["abs", "mag"])

# Extract Base Statistical, Spectral Features
group_acc = base_feat_extractor.get_base_features_for(group_accel_data=group_acc,
                                                                   stat_features=["mean", "var"],
                                                                   spec_features=["psd"],
                                                                   windows=[15], step_size=0.5)
# Extract Synchrony featuress
group_pairwise_sync = sync_extractor.get_synchrony_features_for(group_acc, features=["correl", "lag-correl", "mi", "norm-mi", "mimicry"])
# Extract Convergence features
group_pairwise_conv = conv_extractor.get_convergence_features_for(group_acc, features=["sym-conv", "asym-conv", "global-conv"])


group_pairwise_features = post_processor.concatenate_pairwise_features(group_pairwise_sync, group_pairwise_conv)


print("~~~~~~~~~~ FINAL PAIRWISE FEATUERS ~~~~~~~~~~")
print("# Pairwise Features = " + str(len(group_pairwise_features.keys())))
print("Pairs -> " + str(group_pairwise_features.keys()))

# pd.DataFrame.from_dict(group_pairwise_features).to_csv("features_tester.csv", index=False)

for pair in group_pairwise_features.keys():
    feat_len = len(group_pairwise_features[pair])
    print("# FEATURES per PAIR = " + str(feat_len))
# group_level_features = group_feat_extractor.aggregate_to_group_features(pairwise_features=group_pairwise_features)
# print("# Group-Level Features = " + str(len(group_level_features)))