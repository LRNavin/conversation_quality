import utilities.data_proc_util as data_processor
import utilities.data_read_util as data_reader
import constants as const

import numpy as np
import pandas as pd
from scipy import signal

# Spectral Feature Extraction - Windowed PSD Calculations
def get_binned_psd_for(data, nfft=64, nbins=6):
    f, psd = signal.periodogram(x=data, nfft=nfft, return_onesided=True) #TODO: CHECK WITH WELCH also.
    # print (psd)
    # bin the PSD
    binned_psd = np.zeros(nbins)
    # if the bins have power of two bounds
    if 2 ** (nbins) == nfft:
        binned_psd[0] = psd[1]
        i = 1
        j = 2
        bin_size = 1
        while bin_size <= nfft / 4:
            for k in range(0, bin_size):
                binned_psd[i] += psd[j]
                j += 1
            i += 1
            bin_size *= 2
    else:
        print("BIN Count not a power of 2")

    # print(binned_psd.shape)
    return binned_psd

# Extract Base-Level features - MEAN, VARIANCE, PSDs -> WINDOWED
'''
For all Functions below, Under Extract
Window Size and Step Size -> in Hz. So, convert to hz before windowed calc

Input Size -> (n,7) 
i.e n=duration in secs*20, and 7=#Channels


ROLLING WINDOW with STEP SIZE - Confusion

[This]
df = df.resample('30s')
df.rolling(..).max() (or whatever function) 
[Or]
ts.rolling(5).max().dropna()[::2]  -> Going with this for now

# Returns only the Extracted Features and not the Actual Channels
'''
def extract_windowed_mean(accel_data, window_size, step_size):
    # print("++ Windowed Mean: Window=" + str(window_size) + ", Step=" + str(step_size) + " ++")
    temp_accel_data = [] #np.array([]) #np.copy(accel_data)
    for axial in range(accel_data.shape[0]):
        temp_df = pd.DataFrame(data=accel_data[axial,:])
        windowed_mean_channel = temp_df.rolling(window_size).mean().dropna()[::step_size].values
        if len(temp_accel_data) != 0:
            temp_accel_data = np.concatenate((temp_accel_data, windowed_mean_channel), axis=1)
        else:
            temp_accel_data = windowed_mean_channel
        # print(temp_accel_data.shape)
    return temp_accel_data

def extract_windowed_variance(accel_data, window_size, step_size):
    # print("++ Windowed Variance: Window=" + str(window_size) + ", Step=" + str(step_size) + " ++")
    temp_accel_data = [] #np.array([]) #np.copy(accel_data)
    for axial in range(accel_data.shape[0]):
        temp_df = pd.DataFrame(data=accel_data[axial,:])
        windowed_var_channel = temp_df.rolling(window_size).var().dropna()[::step_size].values
        if len(temp_accel_data) != 0:
            temp_accel_data = np.concatenate((temp_accel_data, windowed_var_channel), axis=1)
        else:
            temp_accel_data = windowed_var_channel
        # print(temp_accel_data.shape)
    return temp_accel_data

# Reshape a numpy array 'a' of shape (n, x) to form shape((n - window_size), window_size, x))
def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def extract_windowed_psd(accel_data, window_size, step_size):
    # print("++ Windowed PSD: Window=" + str(window_size) + ", Step=" + str(step_size) + " ++")
    spec_accel_data = []
    for axial in range(accel_data.shape[0]):
        strided_acc = rolling_window(accel_data[axial,:], window_size)[::step_size]
        bined_psd = np.apply_along_axis(get_binned_psd_for, 1, strided_acc)
        if len(spec_accel_data) != 0:
            spec_accel_data = np.concatenate((spec_accel_data, bined_psd), axis=1)
        else:
            spec_accel_data = bined_psd
        # print(spec_accel_data.shape)
    return spec_accel_data


# Get all Windowed, All Stats Features - Per Member
# Returns only the Extracted Features and not the Actual Channels

def get_windowed_statistical_features_for(accel_data, stats=["mean", "var"], window=1, step_size=1):
    temp_accel_data = np.copy(accel_data)
    stat_accel_data = []
    for stat in stats:
        if stat == "mean": # Windowed Mean
            windowed_stat_acc = extract_windowed_mean(temp_accel_data, window, step_size)
        else: # Windowed Variance
            windowed_stat_acc = extract_windowed_variance(temp_accel_data, window, step_size)
        if len(stat_accel_data) != 0:
            stat_accel_data = np.concatenate((stat_accel_data, windowed_stat_acc), axis=1)
        else:
            stat_accel_data = windowed_stat_acc
    # print(stat_accel_data.shape)
    return stat_accel_data

def get_windowed_spectral_features_for(accel_data, stats=["psd"], window=1, step_size=1):
    temp_accel_data = np.copy(accel_data)
    spec_accel_data = None
    for stat in stats:
        if stat == "psd": # Windowed PSD
            windowed_spec_acc = extract_windowed_psd(temp_accel_data, window, step_size)
        if spec_accel_data:
            spec_accel_data = np.concatenate((spec_accel_data, windowed_spec_acc), axis=1)
        else:
            spec_accel_data = windowed_spec_acc
    # print(spec_accel_data.shape)
    return spec_accel_data

# To Organise Differnet Window size is same straucture (Dict), Can later be combined when actual features are calculated
def get_windowed_base_features_for(group_data, stat_features=["mean", "var"],
                                   spec_features=["psd"], windows=[1, 3, 5, 10, 15], step_size=1):
    group_windowed_data = {}
    for window in windows:
        # print("~~ FOR WINDOW - " + str(window) + " ~~")
        stat_accel_data = get_windowed_statistical_features_for(group_data, stat_features, window, step_size)
        spec_accel_data = get_windowed_spectral_features_for(group_data, spec_features, window, step_size)
        group_windowed_data[str(window)] = np.concatenate((stat_accel_data, spec_accel_data), axis=1)
        # print("AFTER BASE FEATURE EXTRACTION, SHAPE - " + str(group_windowed_data[str(window)].shape))

    return group_windowed_data


# Get all Windowed, All Stats Features - Per Group
def get_base_features_for(group_accel_data, stat_features=["mean", "var"],
                                   spec_features=["psd"], windows=[1, 3, 5, 10, 15], step_size=1):
    step_size = int(step_size * 20) #Convert to samples . 1sec = 20Hz
    windows = [w * 20 for w in windows] #Convert to samples . 1sec = 20Hz
    for member in group_accel_data.keys():
        print("===Extracting BASE Features: Member "+ str(member) + " ===" )
        # print(group_accel_data[member].shape)
        if len(group_accel_data[member]) != 0:
            group_accel_data[member] = get_windowed_base_features_for(group_accel_data[member],
                                                                      stat_features, spec_features, windows, step_size)
    return group_accel_data