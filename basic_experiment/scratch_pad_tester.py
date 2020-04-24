import numpy as np
from scipy import stats
from distutils.dir_util import copy_tree
import shutil
import constants as const
import pandas as pd
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

do_missing_stat = False


if do_missing_stat:
    missign_acc_checklist = pd.read_csv(const.missing_acc_stat)

    sns.distplot(missign_acc_checklist.percent_miss, bins=10, kde=False, rug=True).set_title('Percentage of Missing Accelero Data. (#MissingIndiv/#IndivInGroup)')
    plt.show()

    group_size = np.unique(missign_acc_checklist.member_count.values)
    print(group_size)

    f, axes = plt.subplots(3, 2)
    for i, size in enumerate(group_size):
        size_missing_stat = missign_acc_checklist.loc[missign_acc_checklist.member_count == size].percent_miss
        sns.distplot(size_missing_stat, bins=4, kde=False, rug=True, ax=axes[int(i/2)][int(i%2)], axlabel="").set_title("Group Size:"+str(size), fontsize=8)

    f.suptitle("Missing Data Statistics")
    plt.show()

# with open(const.temp_fform_store, 'rb') as data_store:
#     pd.options.display.max_columns = None
#     groups = pickle.load(data_store)
#     print(groups["Day2"][:2])