#!python
#!/usr/bin/env python
# from pandas_ods_reader import read_ods
from scipy.io import loadmat
from scipy import stats
import constants as const
import numpy as np
import pandas as pd
# import ezodf
import datetime
import pickle
import constants

import utilities.data_proc_util as data_processor
from annotations import load_annotations as loader

def open_dataset(filename=const.dataset_name):
    data = loadmat(const.dataset_base_folder+const.dataset_file_path+filename,
                   struct_as_record=False,
                   squeeze_me=True)[const.data_name]
    return data

def get_data_from_day(filename=const.dataset_name, day=1):
    # print("Opening analysis_data - " + filename + " from Day - " + str(day))
    return open_dataset(filename)[day-1]

def get_accel_data_from_participant(filename=const.dataset_name,
                                    day=1, participant_id=1):
    '''
    Participant Data Structure
    node - int
    participant - int
    begin - double
    end - double
    time - n*1 matrix (n = number of samples (30 min = 36000 samples))
    accel - n*3 matrix (n = number of samples (30 min = 36000 samples))
    '''
    day_data = get_data_from_day(filename, day)
    for member in day_data:
        if str(member.participant) == str(participant_id):
            # print("Found Participant's ("+ str(participant_id) +")"+ " Accelero, ")
            return member.accel
    return None

def z_norm_accel_data(accel_data):
    accel_data = stats.zscore(accel_data, axis=0)
    return accel_data

def get_accel_data_from_participant_between(filename=const.dataset_name, day=1, participant_id=1,
                                            start_time=0, duration=10):
    '''
    start_time, duration in sample rates(hz) i.e 1 sec = 20 samples
    returns numpy array - 1*t [t-time duration] t in hertz -> 20 per sec
    '''

    full_member_data  = np.array(get_accel_data_from_participant(filename=filename, day=day,
                                                                 participant_id=participant_id))

    # print("From: " + str(start_time) + ", For: " + str(duration))
    if len(full_member_data) == 0:
        member_data_timed = full_member_data
    else:
        # Z-Normalise - Remove interpersonal differences in movement intensity
        # print("Shape Before Z-Norm -> " + str(full_member_data.shape))
        full_member_data = z_norm_accel_data(full_member_data)
        # print("After Before Z-Norm -> " + str(full_member_data.shape))
        member_data_timed = full_member_data[(start_time):(start_time+duration), :]
    return member_data_timed

def get_all_annotated_groups():
    return get_annotated_fformations(annotation_file=const.fform_gt_data, from_final_annotations=True)

def get_group_data(group_id="1_000"):
    groups_df = get_all_annotated_groups()
    groups_df["subjects"] = data_processor.clean_subjects_list(groups_df["subjects"])
    group = groups_df.loc[groups_df["groupid"] == group_id]
    return group

def get_members_in_f_form(group_id="1_000"):
    return get_group_data(group_id)['subjects'].values[0]

def get_temporal_data_in_f_form(group_id="1_000", return_end=False):
    group = get_group_data(group_id)
    start, end = group['samplestart'].values[0], group['sampleend'].values[0]
    if return_end:
        return start, end
    else:
        return start, end-start

def get_accel_data_from_f_form(filename=const.dataset_name, group_id="1_000"):
    '''
    returns numpy array - n*t [nFound Participant's Accelero-number of participants, t-time duration]
    '''
    day = int(group_id.split('_')[0])
    group_members       = get_members_in_f_form(group_id)
    start_time, duration = get_temporal_data_in_f_form(group_id)
    print("Fetching Data for Group - " + str(group_id) + ", Members - " + str(group_members))

    # Init group accel array - np
    group_accel = {}#np.empty((len(group_members), duration), int)
    for member in group_members:
        member_accel = get_accel_data_from_participant_between(filename=filename, day=day, participant_id=member,
                                                               start_time=start_time,
                                                               duration=duration)
        group_accel[member] = member_accel #np.append(group_accel, member_accel)
    return group_accel

def process_annotation_sheet(sheet):
    df_dict={}
    print(type(sheet))
    for i, row in enumerate(sheet.rows()):
        # row is a list of cells
        # assume the header is on the first row
        if i not in [0,1]:
            if i == 2:
                header  = {}
                # create index for the column headers
                for j, cell in enumerate(row):
                    if cell.value != None:
                        header[j] = cell.value.replace(" ", "")
                        # columns as lists in a dictionary
                        df_dict[header[j]] = []
                continue
            for j, cell in enumerate(row):
                # use header instead of column index
                if j < len(header) and cell.value is not None:
                    df_dict[header[j]].append(cell.value)

    # and convert to a DataFrame
    sheet_data = pd.DataFrame(df_dict)
    return sheet_data

def get_annotated_fformations(annotation_file=const.fform_annot_data, from_final_annotations=False, from_store=True):
    groups={}
    if from_final_annotations:
        groups = pd.read_csv(annotation_file)
    else:
        if from_store:
            with open(const.temp_fform_store, 'rb') as data_store:
                groups = pickle.load(data_store)
        # else:
        #     # load a file
        #     doc = ezodf.opendoc(annotation_file)
        #     print("Spreadsheet contains %d sheet(s)." % len(doc.sheets))
        #     for i, sheet in enumerate(doc.sheets):
        #         print("Sheet - " + str(sheet.name))
        #         groups[sheet.name] = data_processor.add_column_fform_data(
        #                                 data_processor.clean_fformation_data(
        #                                     process_annotation_sheet(sheet)))
        #         groups[sheet.name].to_csv(const.temp_grps_day+str(i+1)+".csv")
    return groups


def load_speaking_annotations(day=1, participant_id=1, start_time=None, end_time=None,
                              annotation_file=constants.features_tt_path, from_store=True):
    # Data-type formatter
    day, participant_id = int(day), int(participant_id)
    # Real whole Dataframe as MultiIndex
    if from_store:
        tt_df = pd.read_pickle(annotation_file)
    else:
        tt_df = loader.load_annotations(constants.labels_annot, constants.lost_annot, constants.participant_annot)

    # Filter required annotations - For Day, Participant and Time Duration
    if start_time and end_time:
        # Return data from start_time to end_time
        ps_speaking = tt_df.loc[start_time:end_time, (day, participant_id,'Speaking')]
    else:
        # Return whole 30-min data
        ps_speaking = tt_df.loc[:, (day, participant_id,'Speaking')]
    # print("Speaking Status for - Participant = " + str(participant_id) + " in Day = " + str(day) + ". From time " + str(start_time) + " to " +  str(end_time) +".")

    # Filter for -1 (missing data)
    ps_speaking = np.where(ps_speaking.values == -1, 0, ps_speaking.values)
    return ps_speaking.tolist()

def clean_and_store_fform_annotations(annotation_file):
    groups = get_annotated_fformations(annotation_file, from_store=False)
    with open(const.temp_fform_store, 'wb') as data_store:
        pickle.dump(groups, data_store, protocol=pickle.HIGHEST_PROTOCOL)
    return True

def count_missing_accelero(group_acc):
    count=0
    for member in group_acc.keys():
        if len(group_acc[member]) == 0:
            count = count + 1
    return count

def save_pickle(data, file_path):
    with open(file_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return True

def load_pickle(file_path):
    with open(file_path, 'rb') as handle:
        data = pickle.load(handle)
    return data