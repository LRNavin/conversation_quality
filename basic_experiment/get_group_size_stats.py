import utilities.data_read_util as reader
import constants
import json

def run_script_fform_size_stats():

    group_size_count = {}

    data = reader.get_annotated_fformations(constants.fform_annot_data)

    group_size_count["Total"] = [0] * 10
    group_size_count["Number-Of-Groups"] = 0
    group_size_count["Groups-To-Annotate"] = 0
    group_size_count["Group-Duration-Lesser30"] = 0
    group_size_count["Group-Duration-Lesser45"] = 0
    group_size_count["Group-Duration-Lesser60"] = 0

    group_size_count["Group-Duration-1Min"] = 0
    group_size_count["Group-Duration-2Min"] = 0
    group_size_count["Group-Duration-3Min"] = 0
    group_size_count["Group-Duration-4Min"] = 0
    group_size_count["Group-Duration-5Min"] = 0
    group_size_count["Group-Duration-Greater-6min"] = 0

    for key in data.keys():
        day_data = data[key]
        group_size_count[key] = [0] * 10
        for index, group in day_data.iterrows():
            group_size = len(group["subjects"])
            group_size_count[key][group_size] = group_size_count[key][group_size] + 1
            group_size_count["Total"][group_size] = group_size_count["Total"][group_size] + 1
            if group_size > 1:
                if group["duration_secs"] <= 30.0:
                    group_size_count["Group-Duration-Lesser30"] = group_size_count["Group-Duration-Lesser30"] + 1
                elif group["duration_secs"] <= 45.0:
                    group_size_count["Group-Duration-Lesser45"] = group_size_count["Group-Duration-Lesser45"] + 1
                elif group["duration_secs"] < 60.0:
                    group_size_count["Group-Duration-Lesser60"] = group_size_count["Group-Duration-Lesser60"] + 1
                elif group["duration_secs"] < 120.0:
                    group_size_count["Group-Duration-1Min"] = group_size_count["Group-Duration-1Min"] + 1
                elif group["duration_secs"] < 180.0:
                    group_size_count["Group-Duration-2Min"] = group_size_count["Group-Duration-2Min"] + 1
                elif group["duration_secs"] < 240.0:
                    group_size_count["Group-Duration-3Min"] = group_size_count["Group-Duration-3Min"] + 1
                elif group["duration_secs"] < 300.0:
                    group_size_count["Group-Duration-4Min"] = group_size_count["Group-Duration-4Min"] + 1
                elif group["duration_secs"] < 360.0:
                    group_size_count["Group-Duration-5Min"] = group_size_count["Group-Duration-5Min"] + 1
                else:
                    group_size_count["Group-Duration-Greater-6min"] = group_size_count["Group-Duration-Greater-6min"] + 1

    group_size_count["Number-Of-Groups"] = sum(group_size_count["Total"])
    group_size_count["Groups-To-Annotate"] = group_size_count["Number-Of-Groups"] - group_size_count["Total"][1]


    group_size_count = json.dumps(group_size_count)
    print(group_size_count)

    with open(constants.temp_grp_size_store, 'w', encoding='utf-8') as file:
        file.write(group_size_count)  # use `json.loads` to do the reverse

    return True

run_script_fform_size_stats()
