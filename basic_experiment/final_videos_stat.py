#!python
#!/usr/bin/env python
import constants as const
import pandas as pd
import utilities.data_proc_util as data_processor
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
import numpy as np
from scipy import stats
from statistics import mode

plot_data=False
save_data=True

if save_data:
    print("Reading Updated Datasheet.....")
    data = pd.read_csv(const.final_grps_raw_data)
    sheet_data = data_processor.add_column_fform_data(data,"final")
    sheet_data.to_csv(const.final_grps_data)
else:
    sheet_data = pd.read_csv(const.final_grps_data)

all_durations = sheet_data["duration_secs"]
if plot_data:
    print("Total Groups - " + str(len(all_durations)))
    dur_plt = sns.distplot(all_durations, kde=False, rug=True)
    dur_plt.set(xlabel='Conversation Duration (minutes)', ylabel='# Groups')
    plt.show()

print("MIN Duration - " + str(min(all_durations)))
print("MAX Duration - " + str(max(all_durations)))
print("MEAN Duration - " + str(np.mean(all_durations)))
print("MEDIAN Duration - " + str(np.median(all_durations)))
print("MODE Duration - " + str(mode(all_durations)))
print("STD Duration - " + str(np.std(all_durations)))



