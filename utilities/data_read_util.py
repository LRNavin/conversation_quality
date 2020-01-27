#!python
#!/usr/bin/env python
from scipy.io import loadmat
import constants as const
import numpy as np

def open_dataset(filename=const.dataset_name):
    data = loadmat(const.dataset_base_folder+const.dataset_file_path+filename,
                   struct_as_record=False,
                   squeeze_me=True)[const.data_name]
    return data

def get_data_from_day(filename=const.dataset_name, day=1):
    print("Opening dataset - " + filename + " from Day - " + str(day))
    return open_dataset(filename)[day-1]

def get_accel_data_from_participant(filename=const.dataset_name, day=1, participant_id=1):
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

def get_accel_data_from_participant_between(filename=const.dataset_name, day=1, participant_id=1, start_time=0, duration=10):
    '''
    start_time, duration in seconds
    returns numpy array - 1*t [t-time duration] t in sec/hertz -> 2 per sec
    '''
    full_member_data  = np.array(get_accel_data_from_participant(filename=filename, day=day, participant_id=participant_id))
    member_data_timed = full_member_data[(start_time*2):(start_time*2+duration*2), :]
    return member_data_timed

def get_members_in_f_form(group_id=1):
    return None

def get_temporal_data_in_f_form(group_id=1):
    return None, None

def get_accel_data_from_f_form(filename=const.dataset_name, day=1, group_id=1):
    '''
    returns numpy array - n*t [n-number of participants, t-time duration]
    '''
    group_memebers       = get_members_in_f_form(group_id)
    start_time, duration = get_temporal_data_in_f_form(group_id)

    # Init group accel array - np
    group_accel = np.empty((len(group_memebers), duration), int)
    for member in group_memebers:
        member_accel = get_accel_data_from_participant_between(filename=filename, day=day,
                                                               participant_id=member, start_time=start_time,
                                                               duration=duration)
        group_accel = np.append(group_accel, member_accel)

    return group_accel
