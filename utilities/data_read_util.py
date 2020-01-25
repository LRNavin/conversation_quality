#!python
#!/usr/bin/env python
from scipy.io import loadmat
import constants as const

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

def get_accel_data_from_group(filename=const.dataset_name, day=1, group_id=1):
    
    return None
