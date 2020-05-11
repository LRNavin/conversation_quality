import itertools

from annotations import load_annotations as loader

import pandas as pd
import constants


def load_speaking_annotations(day=1, participant_id=1, start_time=None, end_time=None,
                              annotation_file=constants.features_tt_path, from_store=True):
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
    print(ps_speaking)
    return ps_speaking

# ('1', '13', 'Speaking')
# print(load_speaking_annotations(day=1, participant_id=13).values)
# ps = "".join(load_speaking_annotations(day=1, participant_id=1).values.astype(str))

# import itertools.groupby as groupby
# import numpy as np
# c=np.array([[k, len(list(g))] for k, g in itertools.groupby(ps)])
# print(c)