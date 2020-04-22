import pandas as pd
import utilities.data_read_util as reader
import feature_extract.feature_preproc as processor
import feature_extract.statistics_extractor as base_feat_extractor
import feature_extract.synchrony_extractor as sync_extractor
import feature_extract.convergence_extractor as conv_extractor
import feature_extract.features_postproc as post_processor
import feature_extract.group_features_extractor as group_feat_extractor

import constants
import numpy as np
import json
import os
import shutil

# df = pd.DataFrame([10, 20, 15, 30, 45])
#
# print(df)
# print(df.shift(-2))
#
# for i in range(0,4):
#     print(i)

# reader.get_annotated_fformations(constants.fform_annot_data)

# group_size_count = {}
# data = reader.get_annotated_fformations(constants.fform_annot_data, False)
#
# group_size_count["Total"] = [0] * 10
# group_size_count["Number-Of-Groups"] = 0
# group_size_count["Groups-To-Annotate"] = 0
#
# for key in data.keys():
#     day_data = data[key]
#     group_size_count[key] = [0] * 10
#     for index, group in day_data.iterrows():
#         group_size = len(group["subjects"])
#         group_size_count[key][group_size]       = group_size_count[key][group_size] + 1
#         group_size_count["Total"][group_size]   = group_size_count["Total"][group_size] + 1
#
# group_size_count["Number-Of-Groups"] = sum(group_size_count["Total"])
# group_size_count["Groups-To-Annotate"] = group_size_count["Number-Of-Groups"] - group_size_count["Total"][1]
#
# group_size_count = json.dumps(group_size_count)
# print(group_size_count)
#
# with open(constants.temp_grp_size_store, 'w', encoding='utf-8') as file:
#     file.write(group_size_count)  # use `json.loads` to do the reverse

# if 0:
#     reader.clean_and_store_fform_annotations(constants.fform_annot_data)
# else:
#     directory_path = "/Users/navinlr/Desktop/Annotation_Videos/"
#     sub_directories = [dI for dI in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path,dI))]
#     print("Total Sub-Directories - " + str(len(sub_directories)))
#
#     for directory in sub_directories:
#         directory_to_delete = directory_path + directory + "/1-individual/"
#         if os.path.isdir(directory_to_delete):
#             print("Deleting " + directory_to_delete + " .............")
#             shutil.rmtree(directory_to_delete)
#         # break


from scipy import signal
import json

if False:
    all_groupids = reader.get_all_annotated_groups()["groupid"].values
    errored_groups = []

    for group_id in all_groupids:
        try:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~ FOR GROUP - " + str(group_id) + " ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            group_acc = processor.get_accelerometer_channels_for(group_id=group_id, channels=["abs", "mag"])
            break
        except IndexError as e:
            print("Error Occured")
            print(e)
            errored_groups.append(group_id)
    group_acc = base_feat_extractor.get_base_features_for(group_accel_data=group_acc,
                                                                   stat_features=["mean", "var"],
                                                                   spec_features=["psd"],
                                                                   windows=[1, 3, 5, 10, 15, 20], step_size=0.5)

    print("Number of Group members = " + str(len(group_acc.keys())))
    for member in group_acc.keys():
        print("Number of Windows = " + str(len(group_acc[member].keys())))
        for window in group_acc[member].keys():
            data = group_acc[member][window]
            print("Feature SHAPE (Window->"+ str(window) + ") = " + str(data.shape))
        break

    print("Total Errored Groups (Missing Acc) = " + str(len(errored_groups)))
else:

    all_groupids = reader.get_all_annotated_groups()["groupid"].values

    temp_checklist = {
        "group_ids":all_groupids,
        "member_count":[0] * len(all_groupids),
        "missing_acc":[0] * len(all_groupids)
    }
    group_acc_checklist = pd.DataFrame (temp_checklist, columns = ["group_ids", "member_count", "missing_acc"])

    # group_id = "1_018"
    for group_id in all_groupids:

        members = reader.get_members_in_f_form(group_id)

        print("~~~~~~~~~~~~~~~~~~~~~~~~~ FOR GROUP - " + str(group_id) + " ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        # Reading Accelerometer Channels - RAW, ABS, MAG
        group_acc = processor.get_accelerometer_channels_for(group_id=group_id, channels=["abs", "mag"])

        # Decision/Analysis on Missing Accelerations
        #1. set groups data
        group_acc_checklist.loc[group_acc_checklist["group_ids"] == str(group_id), "member_count"] = len(members)
        group_acc_checklist.loc[group_acc_checklist["group_ids"] == str(group_id), "missing_acc"] = reader.count_missing_accelero(group_acc)


        if False:#group_acc_checklist.loc[str(group_id)]["missing_acc"] ==0: # Continue only when all ACC are available
            # Extract Base Statistical, Spectral Features
            group_acc = base_feat_extractor.get_base_features_for(group_accel_data=group_acc,
                                                                               stat_features=["mean", "var"],
                                                                               spec_features=["psd"],
                                                                               windows=[1, 5, 10, 15], step_size=0.5)
            # Extract Synchrony features
            group_pairwise_sync = sync_extractor.get_synchrony_features_for(group_acc, features=["correl", "lag-correl", "mi", "norm-mi", "mimicry"])
            # Extract Convergence features
            group_pairwise_conv = conv_extractor.get_convergence_features_for(group_acc, features=["sym-conv", "asym-conv", "global-conv"])


            group_pairwise_features = post_processor.concatenate_pairwise_features(group_pairwise_sync, group_pairwise_conv)


            print("~~~~~~~~~~ FINAL PAIRWISE FEATUERS ~~~~~~~~~~")
            print("# Pairwise Features = " + str(len(group_pairwise_features.keys())))
            print("Pairs -> " + str(group_pairwise_features.keys()))

            for pair in group_pairwise_features.keys():
                feat_len = len(group_pairwise_features[pair])
                print("# FEATURES per PAIR = " + str(feat_len))

            group_level_features = group_feat_extractor.aggregate_to_group_features(pairwise_features=group_pairwise_features)
            print("# Group-Level Features = " + str(len(group_level_features)))

    group_acc_checklist.to_csv(constants.missing_acc_stat, index=False)

    # from sklearn import mixture
    #
    # data = list(range(0, 100))
    # print(data)
    # data = np.array(data).reshape(-1, 1)
    # # print(data)
    # clf = mixture.GaussianMixture(n_components=1, covariance_type='full')
    # clf.fit(data)



import matplotlib.pyplot as plt


# x, psd = signal.welch(np.random.rand(1000))
# # hist = np.histogram(psd, 10)
#
# # Use non-equal bin sizes, such that they look equal on log scale.
# logbins = np.logspace(0,8, 8)
# hist, bins, _ = plt.hist(x, bins=logbins)
#
# print(hist)
# print(logbins)
#

from scipy import signal
import matplotlib.pyplot as plt
# Generate a test signal, a 2 Vrms sine wave at 1234 Hz, corrupted by 0.001 V**2/Hz of white noise sampled at 10 kHz.

# fs = 10e3
# N = 1e5
# amp = 2*np.sqrt(2)
# freq = 1234.0
# noise_power = 0.001 * fs / 2
# time = np.arange(N) / fs
# x = amp*np.sin(2*np.pi*freq)
# x += np.random.normal(scale=np.sqrt(noise_power))
# from random import gauss
#
# fs=20
# N=100
# x = [gauss(0.0, 1.0) for i in range(N)]
#
# f, Pxx_den = signal.welch(x)
#
# print(f)
# print(len(f))
# print(Pxx_den)
# print(len(Pxx_den))
#
# # plt.hist(Pxx_den, bins=np.logspace(np.log10(0.1),np.log10(10), 6))
# # pl.gca().set_xscale("log")
# # plt.show()
#
# plt.figure()
# plt.plot(x)
# # plt.xscale('INPUT DATA')
# plt.show()
#
# plt.semilogy(f, Pxx_den)
# plt.ylim([0.5e-3, 1])
# plt.xlabel('frequency [Hz]')
# plt.ylabel('PSD [V**2/Hz]')
# plt.show()