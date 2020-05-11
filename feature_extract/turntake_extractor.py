from utilities import data_read_util as data_reader
import constants

import numpy as np
import itertools

# TODO: Skip Conversation Freedom

# VARIABLES
turn_min_length=40
turn_silence_length=10
bc_max_length=40

# Particular Feature Extarctor fucntions - Indiv-level and Group-level

# Individual Base
# 1. Conversation Equality:
def get_number_of_turns_for_indiv(indiv_speaking_data):
    number_of_turns=0
    segments=np.array([[k, len(list(g))] for k, g in itertools.groupby(indiv_speaking_data)])
    # print(" Segment -> ")
    # print(segments)
    for s_i in segments:
        if s_i[0] == 1 and s_i[1] >= turn_min_length:
            number_of_turns = number_of_turns + 1
    return number_of_turns

def get_mean_turn_duration_for_indiv(indiv_speaking_data):
    turn_durations=[]
    segments=np.array([[k, len(list(g))] for k, g in itertools.groupby(indiv_speaking_data)])
    for s_i in segments:
        if s_i[0] == 1 and s_i[1] >= turn_min_length:
            turn_durations.extend([s_i[1]])
    return np.mean(turn_durations)

def get_percent_talk_for_indiv(indiv_speaking_data):
    total_talk_time = np.sum(indiv_speaking_data)
    return total_talk_time/len(indiv_speaking_data)

# 2. Conversation Fluency:
def get_mean_silence_duration_for_indiv(indiv_speaking_data):
    silence_durations = []
    segments = np.array([[k, len(list(g))] for k, g in itertools.groupby(indiv_speaking_data)])
    for s_i in segments:
        if s_i[0] == 0 and s_i[1] >= turn_silence_length:
            silence_durations.extend([s_i[1]])
    return np.mean(silence_durations)

def get_percent_silence_for_indiv(indiv_speaking_data):
    total_silence_time = len(indiv_speaking_data) - np.sum(indiv_speaking_data)
    return total_silence_time/len(indiv_speaking_data)

#bc = back-channels
def get_number_of_bc_for_indiv(indiv_speaking_data):
    number_of_bc=0
    segments=np.array([[k, len(list(g))] for k, g in itertools.groupby(indiv_speaking_data)])
    for s_i in segments:
        if s_i[0] == 1 and s_i[1] < bc_max_length:
            number_of_bc = number_of_bc + 1
    return number_of_bc

# 3. Conversation Synchronisation:
def get_percent_overlap_for_indiv(indiv_speaking_data, rest_group_data):
    total_talk_time = np.sum(indiv_speaking_data)
    total_indiv_ovrlap = np.sum(np.where((np.sum(rest_group_data, axis=0) > 0) & (np.array(indiv_speaking_data) == 1), 1, 0))
    return total_indiv_ovrlap/total_talk_time

def get_number_of_interupt_for_indiv(indiv_speaking_data, rest_group_data):
    success_interupt_cnt=0
    unsuccess_interupt_cnt = 0
    is_turn_overlapping=False
    for curr_speaker_stat, rest_group_stat in zip(indiv_speaking_data, np.sum(rest_group_data, axis=0)):
        if is_turn_overlapping==False: # No ongoing overlaps
            if curr_speaker_stat == 1 and rest_group_stat > 0: # Talk overlap with current indiv
                is_turn_overlapping = True
        else: # Yes Turn ongoing is overlapin
            if curr_speaker_stat == 0 and rest_group_stat > 0:
                # Current Indiv Turn Over ("AFTER Ongoing Overlap"), BUT SOMEONE In GROUP TAKES Over TURN
                success_interupt_cnt = success_interupt_cnt + 1
                is_turn_overlapping = False
            elif curr_speaker_stat == 1 and rest_group_stat == 0:
                # Current Indiv Turn CONTINUES ("AFTER Ongoing Overlap"), BUT GROUP back to SILENCE
                unsuccess_interupt_cnt = unsuccess_interupt_cnt + 1
                is_turn_overlapping = False
    return success_interupt_cnt, unsuccess_interupt_cnt

# TODO: 4. Conversation Freedom: - May not work in dyads
def get_conv_freedom_score_for_indiv(indiv_speaking_data, rest_group_data):
    return 0

# Group Based
# 1. Conversation Equality:
def get_std_number_of_turns_for_group(group_speaking_data):
    members_turns=[]
    for i in range(group_speaking_data.shape[0]):
        members_turns.extend([get_number_of_turns_for_indiv(group_speaking_data[i,:])])
    return np.std(members_turns)

def get_std_turn_duration_for_group(group_speaking_data):
    members_turn_duration = []
    for i in range(group_speaking_data.shape[0]):
        members_turn_duration.extend([get_mean_turn_duration_for_indiv(group_speaking_data[i, :])])
    return np.std(members_turn_duration)

def get_conv_equality_score_for_group(group_speaking_data):
    # REF - C.Lai et al., (2013) - "Modelling Participant Affect in Meetings with Turn-Taking Features"
    def calculate_numerator(T):
        diff=0
        for t_i in T:
            diff = diff + (t_i-np.mean(T))**2
        diff = diff/np.mean(T)
        return diff

    # Current Case - A (Numerator)
    A = calculate_numerator(np.sum(group_speaking_data, axis=1))
    # Worst Case - E (Denominator)
    # Only one person talks - Indiv 0 talks throught
    temp_worst_case = np.zeros(group_speaking_data.shape[0])
    temp_worst_case[0] = group_speaking_data.shape[1]
    E = calculate_numerator(temp_worst_case)
    return A/E

# 2. Conversation Fluency:
def get_mean_silence_duration_for_group(group_speaking_data):
    members_silence_duration = []
    for i in range(group_speaking_data.shape[0]):
        members_silence_duration.extend([get_mean_silence_duration_for_indiv(group_speaking_data[i, :])])
    return np.mean(members_silence_duration)

def get_total_silence_duration_for_group(group_speaking_data):
    overall_silence = np.where(np.sum(group_speaking_data, axis=0) == 0, 1, 0)
    return np.sum(overall_silence)

#bc = back-channels
def get_number_of_bc_for_group(group_speaking_data):
    number_group_bc = []
    for i in range(group_speaking_data.shape[0]):
        number_group_bc.extend([get_number_of_bc_for_indiv(group_speaking_data[i, :])])
    return np.sum(number_group_bc)

# 3. Conversation Synchronisation:
def get_percent_overlap_for_group(group_speaking_data):
    is_overlap = np.where(np.sum(group_speaking_data, axis=0) > 0, 1, 0)
    return np.sum(is_overlap)/len(is_overlap)

def get_number_of_successful_interupt_for_group(group_speaking_data):
    members_suc_interupt=[]
    select_indxs=list(range(group_speaking_data.shape[0]))
    for i in range(group_speaking_data.shape[0]):
        select_indxs.remove(i)
        rest_group_data=np.take(group_speaking_data,select_indxs,axis=0)
        members_suc_interupt.extend([get_number_of_interupt_for_indiv(group_speaking_data[i, :], rest_group_data)[0]])
        select_indxs = list(range(group_speaking_data.shape[0]))
    return np.sum(members_suc_interupt)

def get_number_of_unsuccessful_interupt_for_group(group_speaking_data):
    members_unsuc_interupt=[]
    select_indxs=list(range(group_speaking_data.shape[0]))
    for i in range(group_speaking_data.shape[0]):
        select_indxs.remove(i)
        rest_group_data=np.take(group_speaking_data,select_indxs,axis=0)
        members_unsuc_interupt.extend([get_number_of_interupt_for_indiv(group_speaking_data[i, :], rest_group_data)[1]])
        select_indxs = list(range(group_speaking_data.shape[0]))
    return np.sum(members_unsuc_interupt)

# TODO: 4. Conversation Freedom: - May not work in dyads
def get_conv_freedom_score_for_group(group_speaking_data):
    return 0


# Utlis
#Return - (#mebers, duration) shaped status array
def get_stacked_group_speaking_data(group_members, day, start_time, end_time):
    group_speaking_data=[]
    for indiv_id in group_members:
        indiv_speaking_status = data_reader.load_speaking_annotations(day=day, participant_id=indiv_id,
                                                                      start_time=start_time, end_time=end_time)
        if len(group_speaking_data) == 0:
            group_speaking_data = indiv_speaking_status
        else:
            group_speaking_data = np.vstack((group_speaking_data, indiv_speaking_status))
    return group_speaking_data

# Public Functions
# Tester->1_005
def get_group_tt_features_for(group_id, features=["#turns", "mean_turn", "mean_silence", "%talk", "%silence"]):
    tt_features=[]
    group_members = data_reader.get_members_in_f_form(group_id)
    start_time, end_time  = data_reader.get_temporal_data_in_f_form(group_id=group_id, return_end=True)
    group_speaking_data = get_stacked_group_speaking_data(group_members, group_id.split("_")[0], start_time, end_time)
    for tt_feat in features:
        # Conv Eq ["#turns", "mean_turn", "conv_eq"]
        if tt_feat == "var_#turn":
            tt_features.extend([get_std_number_of_turns_for_group(group_speaking_data)])
        elif tt_feat == "var_dturn":
            tt_features.extend([get_std_turn_duration_for_group(group_speaking_data)])
        elif tt_feat == "conv_eq":
            tt_features.extend([get_conv_equality_score_for_group(group_speaking_data)])
        # Conv Fluency ["mean_silence", "%silence", "#bc"]
        elif tt_feat == "mean_silence":
            tt_features.extend([get_mean_silence_duration_for_group(group_speaking_data)])
        elif tt_feat == "%silence":
            tt_features.extend([get_total_silence_duration_for_group(group_speaking_data)])
        elif tt_feat == "#bc":
            tt_features.extend([get_number_of_bc_for_group(group_speaking_data)])
        # Conv Synch ["%overlap", "#suc_interupt", "#un_interupt"]
        elif tt_feat == "%overlap":
            tt_features.extend([get_percent_overlap_for_group(group_speaking_data)])
        elif tt_feat == "#suc_interupt":
            tt_features.extend([get_number_of_successful_interupt_for_group(group_speaking_data)])
        elif tt_feat == "#un_interupt":
            tt_features.extend([get_number_of_unsuccessful_interupt_for_group(group_speaking_data)])
        # Conv Free ["conv_free"]
        elif tt_feat == "conv_free":
            tt_features.extend([get_conv_freedom_score_for_group(group_speaking_data)])
    return tt_features

# Tester->1_005_13
def get_indiv_tt_features_for(group_indiv_id, features=["#turns", "mean_turn", "mean_silence", "%talk", "%silence"]):
    tt_features=[]
    split_token = group_indiv_id.split("_")
    day, group_id, indiv_id = split_token[0], split_token[1], split_token[2]
    group_members = rest_group_members = data_reader.get_members_in_f_form(day+"_"+group_id)
    start_time, end_time  = data_reader.get_temporal_data_in_f_form(group_id=day+"_"+group_id, return_end=True)
    indiv_speaking_status = data_reader.load_speaking_annotations(day=day, participant_id=indiv_id,
                                                                  start_time=start_time, end_time=end_time)
    rest_group_members.remove(indiv_id)
    rest_group_data = get_stacked_group_speaking_data(rest_group_members, day, start_time, end_time)
    for tt_feat in features:
        # Conv Eq ["#turns", "%talk", "mean_turn"]
        if tt_feat == "#turns":
            tt_features.extend([get_number_of_turns_for_indiv(indiv_speaking_status)])
        elif tt_feat == "%talk":
            tt_features.extend([get_percent_talk_for_indiv(indiv_speaking_status)])
        elif tt_feat == "mean_turn":
            tt_features.extend([get_mean_turn_duration_for_indiv(indiv_speaking_status)])
        # Conv Fluency ["mean_silence", "%silence", "#bc"]
        elif tt_feat == "mean_silence":
            tt_features.extend([get_mean_silence_duration_for_indiv(indiv_speaking_status)])
        elif tt_feat == "%silence":
            tt_features.extend([get_percent_silence_for_indiv(indiv_speaking_status)])
        elif tt_feat == "#bc":
            tt_features.extend([get_number_of_bc_for_indiv(indiv_speaking_status)])
        # Conv Synch ["%overlap", "#suc_interupt", "#un_interupt"]
        elif tt_feat == "%overlap":
            tt_features.extend([get_percent_overlap_for_indiv(indiv_speaking_status, rest_group_data)])
        elif tt_feat == "#suc_interupt":
            tt_features.extend([get_number_of_interupt_for_indiv(indiv_speaking_status, rest_group_data)[0]]) # Can merge this and below, but hold for readability
        elif tt_feat == "#un_interupt":
            tt_features.extend([get_number_of_interupt_for_indiv(indiv_speaking_status, rest_group_data)[1]])
        # Conv Free ["conv_free"]
        elif tt_feat == "conv_free":
            tt_features.extend([get_conv_freedom_score_for_indiv(indiv_speaking_status, rest_group_data)])
    return tt_features

# Testing
from dataset_creation import dataset_creator as data_generator
# filtered_dataset, convq_ids, convq_scores = data_generator.filter_dataset(constants.features_dataset_path_v1,
#                                                                           50.0, .2,
#                                                                           "indiv", ["Nakul", "Divya"], True, False)

test_manifest="group"

if test_manifest=="indiv":
    featues=["#turns", "%talk", "mean_turn", "mean_silence", "%silence", "#bc", "%overlap", "#suc_interupt", "#un_interupt"]
    print(get_indiv_tt_features_for("1_005_13", featues))
    print(featues)
elif test_manifest=="group":
    featues = ["var_#turn", "var_dturn", "conv_eq", "mean_silence", "%silence", "#bc", "%overlap", "#suc_interupt", "#un_interupt"]
    print(get_group_tt_features_for("1_005", featues))
    print(featues)