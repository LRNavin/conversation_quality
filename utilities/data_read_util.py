#!python
#!/usr/bin/env python
from pandas_ods_reader import read_ods
from scipy.io import loadmat
import constants as const
import numpy as np
import pandas as pd
import ezodf
import datetime
import pickle

import utilities.data_proc_util as data_processor

def open_dataset(filename=const.dataset_name):
    data = loadmat(const.dataset_base_folder+const.dataset_file_path+filename,
                   struct_as_record=False,
                   squeeze_me=True)[const.data_name]
    return data

def get_data_from_day(filename=const.dataset_name, day=1):
    print("Opening dataset - " + filename + " from Day - " + str(day))
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
            print("Found Participant")
            return member.accel
    return None

def get_accel_data_from_participant_between(filename=const.dataset_name,
                                            day=1, participant_id=1,
                                            start_time=0, duration=10):
    '''
    start_time, duration in sample rates(hz) i.e 1 sec = 20 samples
    returns numpy array - 1*t [t-time duration] t in hertz -> 20 per sec
    '''
    full_member_data  = np.array(get_accel_data_from_participant(filename=filename,
                                                                 day=day, participant_id=participant_id))
    member_data_timed = full_member_data[(start_time):(start_time+duration), :]
    return member_data_timed

def get_members_in_f_form(day=1, group_id=0):
    return get_annotated_fformations()['Day'+str(day)].loc[[group_id]]['subjects'].values[0]

def get_temporal_data_in_f_form(day=1, group_id=0):
    group_data = get_annotated_fformations()['Day'+str(day)].loc[[group_id]]
    start, end = group_data['samplestart'].values[0], group_data['sampleend'].values[0]
    return start, end-start

def get_accel_data_from_f_form(filename=const.dataset_name,
                               day=1, group_id=0):
    '''
    returns numpy array - n*t [n-number of participants, t-time duration]
    '''
    group_members       = get_members_in_f_form(day, group_id)
    start_time, duration = get_temporal_data_in_f_form(day, group_id)
    print("Fetching Data for Group - " + str(group_id) + ", Day-" + str(day) + ", Members - " + str(group_members))
    # Init group accel array - np
    group_accel = {}#np.empty((len(group_members), duration), int)
    for member in group_members:
        member_accel = get_accel_data_from_participant_between(filename=filename, day=day,
                                                               participant_id=member, start_time=start_time,
                                                               duration=duration)
        group_accel[member] = member_accel #np.append(group_accel, member_accel)
    return group_accel

def process_annotation_sheet(sheet):
    df_dict={}
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

def get_annotated_fformations(annotation_file=const.fform_annot_data, from_store=True):
    groups={}
    if from_store:
        with open(const.temp_fform_store, 'rb') as data_store:
            groups = pickle.load(data_store)
    else:
        # load a file
        doc = ezodf.opendoc(annotation_file)
        print("Spreadsheet contains %d sheet(s)." % len(doc.sheets))
        for i, sheet in enumerate(doc.sheets):
            print("Sheet - " + str(sheet.name))
            groups[sheet.name] = data_processor.clean_fformation_data(process_annotation_sheet(sheet))
            # print(groups[sheet.name])
    return groups

def clean_and_store_fform_annotations(annotation_file):
    groups = get_annotated_fformations(annotation_file, from_store=False)
    with open(const.temp_fform_store, 'wb') as data_store:
        pickle.dump(groups, data_store, protocol=pickle.HIGHEST_PROTOCOL)
    return True



