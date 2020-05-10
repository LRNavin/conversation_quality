from utilities import data_read_util as data_reader
import constants

import numpy as np


# Particular Feature Extarctor fucntions - Indiv-level and Group-level

# Individual Base
# 1. Conversation Equality:
def get_number_of_turns_for_indiv(indiv_speaking_data):
    number_of_turns=0
    for i, curr_status in enumerate(indiv_speaking_data.values):
        if i==0 and curr_status == 1:
            number_of_turns = number_of_turns + 1
        else:
            if curr_status == 1 and indiv_speaking_data.values[i-1] == 0:
                number_of_turns = number_of_turns + 1
    return number_of_turns

def get_mean_turn_duration_for_indiv(indiv_speaking_data):
    turn_durations=[]
    curr_turn_duration=0
    for i, curr_status in enumerate(indiv_speaking_data.values):
        if curr_status == 1: # Turn Begins/Continues
            curr_turn_duration = curr_turn_duration + 1 # Increase turn duration
        elif curr_status == 0: # Silence Begins/Continues
            turn_durations.extend(curr_turn_duration)
            curr_turn_duration = 0 # Reset turn duration counter
    return np.mean(turn_durations)

def get_percent_talk_for_indiv(indiv_speaking_data):
    total_talk_time = np.sum(indiv_speaking_data.values)
    return total_talk_time/len(indiv_speaking_data.values)

# 2. Conversation Fluency:
def get_mean_silence_duration_for_indiv(indiv_speaking_data):
    silence_durations = []
    curr_silence_duration = 0
    for i, curr_status in enumerate(indiv_speaking_data.values):
        if curr_status == 0:  # Silence Begins/Continues
            curr_silence_duration = curr_silence_duration + 1  # Increase turn duration
        elif curr_status == 1:  # Turn Begins/Continues
            silence_durations.extend(curr_silence_duration)
            curr_silence_duration = 0  # Reset turn duration counter
    return np.mean(silence_durations)

def get_percent_silence_for_indiv(indiv_speaking_data):
    total_silence_time = len(indiv_speaking_data.values) - np.sum(indiv_speaking_data.values)
    return total_silence_time/len(indiv_speaking_data.values)

# TODO: 3. Conversation Synchronisation:

# TODO: 4. Conversation Freedom:


# Group Based
# 1. Conversation Equality:
def get_mean_number_of_turns_for_group(group_speaking_data):
    return 0

def get_mean_turn_duration_for_group(group_speaking_data):
    return 0

def get_conv_equality_score_for_group(group_speaking_data):
    return 0

# 2. Conversation Fluency:
def get_mean_silence_duration_for_group(group_speaking_data):
    return 0

def get_total_silence_duration_for_group(group_speaking_data):
    return 0

# 3. Conversation Synchronisation:
def get_percent_overlap_for_group(group_speaking_data):
    return 0

def get_number_of_successful_interupt_for_group(group_speaking_data):
    return 0


def get_number_of_unsuccessful_interupt_for_group(group_speaking_data):
    return 0

# 3. Conversation Freedom:
def get_conv_freedom_score_for_group(group_speaking_data):
    return 0



# Utlis
#Return - (#mebers, duration) shaped status array
def get_stacked_group_speaking_data(group_members, day, start_time, end_time):
    group_speaking_data=[]
    for indiv_id in group_members:
        indiv_speaking_status = data_reader.load_speaking_annotations(day=day, participant_id=indiv_id,
                                                                      start_time=start_time, end_time=end_time)
        if len(group_members) == 0:
            group_speaking_data = indiv_speaking_status
        else:
            group_speaking_data = np.vstack((group_speaking_data, indiv_speaking_status))
    return group_speaking_data

# Public Functions
# Tester->1_005
def get_group_tt_features_for(group_id, features=["", "mean_turn"]):
    tt_features=[]
    group_members = data_reader.get_members_in_f_form(group_id)
    start_time, end_time  = data_reader.get_temporal_data_in_f_form(group_id=group_id, return_end=False)
    group_speaking_data = get_stacked_group_speaking_data(group_members, group_id.split("_")[0], start_time, end_time)
    for tt_feat in features:
        if tt_feat == "mean_turn":
            tt_features.extend(get_mean_number_of_turns_for_group(group_speaking_data))
    return tt_features

# Tester->1_005_13
def get_indiv_tt_features_for(group_indiv_id, features=["#turns", "mean_turn", "mean_silence", "%talk", "%silence"]):
    tt_features=[]
    split_token = group_indiv_id.split("_")
    day, group_id, indiv_id = split_token[0], split_token[1], split_token[2]
    start_time, end_time  = data_reader.get_temporal_data_in_f_form(group_id=day+"_"+group_id, return_end=False)
    indiv_speaking_status = data_reader.load_speaking_annotations(day=day, participant_id=indiv_id,
                                                                  start_time=start_time, end_time=end_time)
    for tt_feat in features:
        if tt_feat == "#turns":
            tt_features.extend(get_number_of_turns_for_indiv(indiv_speaking_status))
        elif tt_feat == "mean_turn":
            tt_features.extend(get_mean_turn_duration_for_indiv(indiv_speaking_status))
        elif tt_feat == "mean_silence":
            tt_features.extend(get_mean_silence_duration_for_indiv(indiv_speaking_status))
        elif tt_feat == "%talk":
            tt_features.extend(get_percent_talk_for_indiv(indiv_speaking_status))
        elif tt_feat == "%silence":
            tt_features.extend(get_percent_silence_for_indiv(indiv_speaking_status))
    return tt_features

# Testing
from dataset_creation import dataset_creator as data_generator
filtered_dataset, convq_ids, convq_scores = data_generator.filter_dataset(constants.features_dataset_path_v1,
                                                                          50.0, .2,
                                                                          "indiv", ["Nakul", "Divya"], True, False)

print(convq_ids)