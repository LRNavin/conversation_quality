import utilities.data_read_util as reader
import constants
import json

group_size_count = {}
data = reader.get_annotated_fformations(constants.fform_annot_data)

group_size_count["Total"] = [0] * 10
group_size_count["Number-Of-Groups"] = 0
group_size_count["Groups-To-Annotate"] = 0

for key in data.keys():
    day_data = data[key]
    group_size_count[key] = [0] * 10
    for index, group in day_data.iterrows():
        group_size = len(group["subjects"])
        group_size_count[key][group_size]       = group_size_count[key][group_size] + 1
        group_size_count["Total"][group_size]   = group_size_count["Total"][group_size] + 1

group_size_count["Number-Of-Groups"] = sum(group_size_count["Total"])
group_size_count["Groups-To-Annotate"] = group_size_count["Number-Of-Groups"] - group_size_count["Total"][1]

group_size_count = json.dumps(group_size_count)
print(group_size_count)

with open(constants.temp_grp_size_store, 'w', encoding='utf-8') as file:
    file.write(group_size_count)  # use `json.loads` to do the reverse