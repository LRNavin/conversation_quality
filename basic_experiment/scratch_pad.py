import pandas as pd
import utilities.data_read_util as reader
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

if 0:
    reader.clean_and_store_fform_annotations(constants.fform_annot_data)
else:
    directory_path = "/Users/navinlr/Desktop/Annotation_Videos/"
    sub_directories = [dI for dI in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path,dI))]
    print("Total Sub-Directories - " + str(len(sub_directories)))

    for directory in sub_directories:
        directory_to_delete = directory_path + directory + "/1-individual/"
        if os.path.isdir(directory_to_delete):
            print("Deleting " + directory_to_delete + " .............")
            shutil.rmtree(directory_to_delete)
        # break

