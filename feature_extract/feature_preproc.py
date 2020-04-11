import utilities.data_proc_util as data_processor
import utilities.data_read_util as data_reader
import constants as const

import numpy as np


#Change Shape of ACC ARRAY - Transpose required
def invert_shape_of(group_acc):
    for member in group_acc.keys():
        group_acc[member] = np.transpose(group_acc[member])
    return group_acc

# Extract Channels from Raw Accel => Raw x,y,z , Absolute x,y,z and Magnitude
def get_raw_accelerometer_for(group_id):
    group_acc = data_reader.get_accel_data_from_f_form(filename=const.dataset_name, group_id=group_id)
    return group_acc

def add_abs_accelerometer_for(group_acc):
    for member in group_acc.keys():
        raw_member_acc = group_acc[member]
        abs_member_acc = np.absolute(raw_member_acc)
        group_acc[member] = np.concatenate((raw_member_acc, abs_member_acc), axis=1)
    return group_acc

def add_mag_accelerometer_for(group_acc):
    for member in group_acc.keys():
        raw_member_acc = group_acc[member]
        mag_member_acc = np.sqrt(np.square(raw_member_acc[:,0]) + np.square(raw_member_acc[:,1]) + np.square(raw_member_acc[:,2]))
        group_acc[member] = np.concatenate((raw_member_acc, mag_member_acc[:, None]), axis=1)
    return group_acc

# Get all Channels
def get_accelerometer_channels_for(group_id, channels=["abs", "mag"]):
    channled_group_acc = raw_group_acc = get_raw_accelerometer_for(group_id)
    for channel in channels:
        if channel == "abs":
            channled_group_acc = add_abs_accelerometer_for(raw_group_acc)
        if channel == "mag":
            channled_group_acc = add_mag_accelerometer_for(raw_group_acc)
    return invert_shape_of(channled_group_acc)
