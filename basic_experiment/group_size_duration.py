import utilities.data_read_util as reader
import constants
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)


data = reader.get_annotated_fformations(constants.fform_annot_data)
all_durations = []
valid_durations = []

for key in data.keys():
    day_data = data[key]
    for index, group in day_data.iterrows():
        group_size = len(group["subjects"])
        if group_size > 1:
            group_duration = group["duration_mins"]
            valid_durations.append(group_duration)
    print(key + " - " + str(len(valid_durations)))
    all_durations.extend(valid_durations)
    valid_durations = []

print("Total Groups - " + str(len(all_durations)))
dur_plt = sns.distplot(all_durations, bins=10, kde=False, rug=True)
dur_plt.set(xlabel='Conversation Duration (minutes)', ylabel='# Groups')
plt.show()