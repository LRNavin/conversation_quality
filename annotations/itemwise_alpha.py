import annotations.annotation_cleaner as annot_cleaner
import constants
import pingouin as pg
import numpy as np
import pandas as pd

manifestation = "indiv"
annotation_file = constants.indiv_conq_annot_data
annotators=["Nakul", "Swathi", "Divya"]


def get_cronbach_alpha(itemscores):
    # cols are items, rows are observations
    calpha = pg.cronbach_alpha(data=pd.DataFrame(itemscores))[0]
    print("Current Alpha: " + str(calpha))
    return calpha

cleaned_annotations = annot_cleaner.derive_convq_scores_for_reponses(annotators, annotation_file, manifestation, calculate_convq=False, reverse_scale=False, zero_mean=True)
col_list = list(cleaned_annotations)
if manifestation == "group":
    column_omit = ["Annotator Name", "Group ID"]
    sort_list = ['Group ID']
    drop_list = ['Annotator Name','Group ID']
else:
    column_omit = ["Annotator Name", "Group ID", "Individual ID"]
    sort_list = ['Group ID', 'Individual ID']
    drop_list = ['Annotator Name','Group ID',"Individual ID"]

for col in column_omit:
    col_list.remove(col)

annotator1_resp = cleaned_annotations.loc[cleaned_annotations['Annotator Name'] == annotators[0]].drop_duplicates(drop_list, 'last').sort_values(by=sort_list)
annotator2_resp = cleaned_annotations.loc[cleaned_annotations['Annotator Name'] == annotators[1]].drop_duplicates(drop_list, 'last').sort_values(by=sort_list)
annotator3_resp = cleaned_annotations.loc[cleaned_annotations['Annotator Name'] == annotators[2]].drop_duplicates(drop_list, 'last').sort_values(by=sort_list)


for ques in col_list:
    print("Cronbach Alpha for Question - " + ques)
    question_responses = cleaned_annotations[ques]

    question_responses_stacked = np.vstack((annotator1_resp[ques].values,
                                            annotator2_resp[ques].values,
                                            annotator3_resp[ques].values)).transpose()

    score = get_cronbach_alpha(question_responses_stacked)