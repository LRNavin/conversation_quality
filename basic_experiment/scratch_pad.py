import pandas as pd
import utilities.data_read_util as reader
import feature_extract.feature_preproc as processor
import feature_extract.statistics_extractor as base_feat_extractor
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

if True:
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