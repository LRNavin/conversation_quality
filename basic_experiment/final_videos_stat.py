#!python
#!/usr/bin/env python
import constants as const
import pandas as pd
import utilities.data_proc_util as data_processor
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)

plot_data=True
save_data=True

if save_data:
    print("Reading Updated Datasheet.....")
    data = pd.read_csv(const.final_grps_raw_data)
    sheet_data = data_processor.add_column_fform_data(data,"final")
    sheet_data.to_csv(const.final_grps_data)
else:
    sheet_data = pd.read_csv(const.final_grps_data)

if plot_data:
    all_durations = sheet_data["duration_mins"]
    print("Total Groups - " + str(len(all_durations)))
    dur_plt = sns.distplot(all_durations, kde=False, rug=True)
    dur_plt.set(xlabel='Conversation Duration (minutes)', ylabel='# Groups')
    plt.show()
