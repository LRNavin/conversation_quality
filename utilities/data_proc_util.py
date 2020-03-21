import datetime
import pandas as pd
import numpy as np

def clean_sample_checkpoint(samplepoint_column):
    return samplepoint_column.apply(lambda x: int(x))

def clean_timestamp(time_column):
    #TODO: Some problem with Day3, needs fixes - Problemn with dataset
    return time_column.apply(
        lambda x: ":".join((x.replace("PT", "00:").replace("H", ":").replace("M", ":").replace("S", "")).split(":")[:3]))
            # .replace(":00", "")).replace(".", ":")

# DATASET PROBLEMS:{
def fix_24hrs(x):
    if x>24.0:x=x-24.0
    return x

def fix_timestamp_24hrs_problem(time_column):
    return time_column.apply(lambda x: fix_24hrs(x))
#}


def clean_subjects_array(subject_array):
    for i, subject in enumerate(subject_array):
        subject_array[i] = subject.split('.')[0]
    subject_array = list(filter(None, subject_array))
    return subject_array

def clean_subjects_list(subjects_column):
    split_column = subjects_column.apply(lambda x: str(x).split(" "))
    subjects_column= split_column.apply(lambda x: clean_subjects_array(x))
    return subjects_column

def clean_fformation_data(data):

    for column in data.columns:
        if column == "subjects":
            data[column] = clean_subjects_list(data[column])
        elif column == "timestart":
            data[column] = clean_timestamp(data[column])
            # print(data[column])
            data[column] = data[column].apply(lambda x: to_min_sec(x))
            # data[column] = fix_timestamp_24hrs_problem(data[column])
        elif column == "timeend":
            data[column] = clean_timestamp(data[column])
            data[column] = data[column].apply(lambda x: to_min_sec(x))
            # data[column] = fix_timestamp_24hrs_problem(data[column])
        elif column == "samplestart":
            data[column] = clean_sample_checkpoint(data[column])
        elif column == "sampleend":
            data[column] = clean_sample_checkpoint(data[column])
    return data

def to_min_sec(datetime_str):
    datetime_object = datetime.datetime.strptime(datetime_str, '%H:%M:%S')
    return datetime_object

def add_column_fform_data(data, col_name="time-duration"):
    if col_name == "time-duration":
        buffer_diff = data["timeend"] - data["timestart"]
        data["duration_mins"] = (buffer_diff / np.timedelta64(1, 'm')).round(2)
        data["duration_secs"] = buffer_diff / np.timedelta64(1, 's')
    return data

