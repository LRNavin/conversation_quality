import utilities.data_read_util as reader
from sklearn import preprocessing
from sklearn.preprocessing import scale

import constants
import pandas as pd
import numpy as np

# best = {1-5}
normal_conversion_dict = {
"Strongly disagree" : 1,
"Disagree" : 2,
"Neutral" : 3,
"Agree" : 4,
"Strongly Agree" : 5,
"Strongly agree" : 5
}

reverse_conversion_dict = {

"Strongly Agree" : 1,
"Strongly agree" : 1,
"Agree" : 2,
"Neutral" : 3,
"Disagree" : 4,
"Strongly disagree" : 5
}


# Group Questions - Not Used - Used only for reverse scale

# gq1="The interaction within the group was smooth, natural and relaxed."
# gq2="The group members looked to have enjoyed the interaction."
gq3="The interaction within the group was forced, awkward, and strained."
# gq4="The group members accepted and respected each other in the interaction."
# gq5="The group members received equal opportunity to participate freely in the interaction."
# gq6="The interaction involves equal participation from all group members."
# gq7="The group members seemed to have gotten along with each other pretty well."
# gq8="The group members were paying attention to their partners throughout the interaction."
# gq9="The group members attempted to get “in sync” with their partners."
# gq10="The group members used their partner’s behavior as a guide for their own behavior."

iq3="The individual’s interaction seemed to be forced, awkward, and strained."
iq5="The individual looked uncomfortable during the interaction."
iq10="The individual looked to be self-conscious during the interaction."

all_groups = reader.get_all_annotated_groups()["groupid"].values.tolist()
no_of_groups = len(all_groups)
print("Total Groups = " + str(no_of_groups))
no_of_indivs = 340


def are_all_members_annotated_in_group(cleaned_annotation):
    incomplete_groups = []
    for group_id in all_groups:
        members_count = len(reader.get_members_in_f_form(group_id))
        annotat_count = len(cleaned_annotation.loc[cleaned_annotation['Group ID'] == str(group_id)])
        if members_count != annotat_count:
            print("MISSING INDIV ANNOTATIONS - " + str(group_id) + " --- " + str(members_count-annotat_count))
            incomplete_groups.append(group_id)
    return incomplete_groups

def sanity_check_responses(annotator_annotations, manifestation):
    cleaned_annotation = annotator_annotations.copy(deep=False)
    unique_groupids = np.unique(cleaned_annotation['Group ID'].values)
    # Completion Checker
    # print("MISSING GROUP - " + str([x for x in all_groups if x not in unique_groupids]))
    # print("Unique groups - " + str(len(unique_groupids)))
    # print("All response groups - " + str(len(cleaned_annotation['Group ID'].values)))
    if len(unique_groupids) != no_of_groups:
        return False, cleaned_annotation
    if len(unique_groupids) != len(cleaned_annotation['Group ID'].values):
        # Yes, Duplciates present - keep first for now
        # print("~~~~~~~~~~~~~~~~~~ Removing DUPLICATESS ~~~~~~~~~~~~~~~~~~~~ "+ manifestation)
        if manifestation == "group":
            cleaned_annotation = cleaned_annotation.drop_duplicates(['Annotator Name', 'Group ID'], 'first')
        else:
            cleaned_annotation = cleaned_annotation.drop_duplicates(['Annotator Name', 'Group ID', "Individual ID"], 'last')

    if manifestation == "indiv":
        # print("# Indiv Responses After Clean - " + str(len(cleaned_annotation)))
        #1. Check NUmber of members per group
        incomplete_groups = are_all_members_annotated_in_group(cleaned_annotation)
        if len(incomplete_groups) == 0:
            return True, cleaned_annotation
        else:
            print("INCOMPLTE GROUPS - " + str(incomplete_groups))
            return False, cleaned_annotation

    return True, cleaned_annotation


def convert_normal_annotations_to_int(annotation):
    int_value = normal_conversion_dict[annotation]
    return int_value

def convert_reverse_annotations_to_int(annotation):
    int_value = reverse_conversion_dict[annotation]
    return int_value

def clean_annotations(raw_annotations, manifestation, reverse_scale):
    if manifestation == "group":
        column_omit = ["Annotator Name", "Group ID"]
        reverse_sca = [gq3]
    else:
        column_omit = ["Annotator Name", "Group ID", "Individual ID"]
        reverse_sca = [iq3, iq5, iq10]

    cleaned_annotation = raw_annotations.copy(deep=False)
    for column in cleaned_annotation:
        if column not in column_omit:
            if reverse_scale:
                if column in reverse_sca: #Reverse Scale
                    # print("~~~~~~~~ Reversing Scale ~~~~~~~~~")
                    cleaned_annotation[column] = cleaned_annotation[column].apply(convert_reverse_annotations_to_int)
                else:
                    cleaned_annotation[column] = cleaned_annotation[column].apply(convert_normal_annotations_to_int)
            else:
                cleaned_annotation[column] = cleaned_annotation[column].apply(convert_normal_annotations_to_int)
    return cleaned_annotation

def normalize_annotator_bias_for(group_annotations):
    normalized_annota = preprocessing.MinMaxScaler().fit_transform(group_annotations.reshape(-1, 1))
    return normalized_annota

def calcualte_convq_score_for(cleaned_annotation, manifestation="group"):
    scored_annotations = cleaned_annotation.copy(deep=False)
    col_list = list(scored_annotations)
    col_list.remove("Group ID")
    col_list.remove("Annotator Name")
    if manifestation == "group":
        scored_annotations["group_convq"] = scored_annotations[col_list].mean(axis=1)
        # scored_annotations["group_convq"] = normalize_annotator_bias_for(scored_annotations["group_convq"].values)
    else:
        col_list.remove('Individual ID')
        scored_annotations["indiv_convq"] = scored_annotations[col_list].mean(axis=1)
        # scored_annotations["indiv_convq"] = normalize_annotator_bias_for(scored_annotations["indiv_convq"].values)

    return scored_annotations

def derive_convq_scores_for_reponses(annotators, annotation_file=constants.group_conq_annot_data, manifestation="group", calculate_convq=True, reverse_scale=False, zero_mean=False):

    raw_annotations = pd.read_csv(annotation_file).drop('Timestamp', 1)
    raw_annotations = raw_annotations.loc[raw_annotations["Group ID"].isin(all_groups)]
    cleaned_annotation = clean_annotations(raw_annotations, manifestation, reverse_scale)
    print("ZERO-MEAN Technique ? - " + str(zero_mean))
    if zero_mean:
        skip_column=['Annotator Name', 'Group ID', 'Individual ID']
        for i, annotator in enumerate(annotators):
            # annotator_mean = int(np.mean(np.mean(cleaned_annotation.loc[cleaned_annotation['Annotator Name'] == annotator])[2:]))
            # print("Anntoator MEan")
            # print(annotator_mean)
            for column in cleaned_annotation.columns:
                if column not in skip_column:
                    annotator_ques_responses = cleaned_annotation.loc[cleaned_annotation['Annotator Name'] == annotator][column]
                    # print(annotator_ques_responses.values)
                    # print(np.mean(annotator_ques_responses.values))
                    # print(annotator_ques_responses.values)
                    # print(np.mean(annotator_ques_responses))
                    # print("Mean = " + str(int(np.mean(annotator_ques_responses))))
                    response_scaled = annotator_ques_responses - int(np.mean(annotator_ques_responses)) #scale(annotator_ques_responses, with_mean=True, with_std=False)
                    # print(response_scaled.values)
                    cleaned_annotation.loc[cleaned_annotation['Annotator Name'] == annotator, column]=response_scaled
                    # print(cleaned_annotation.loc[cleaned_annotation['Annotator Name'] == annotator][column].values)
    if calculate_convq:
        scored_annotations = calcualte_convq_score_for(cleaned_annotation, manifestation)
        return scored_annotations
    return cleaned_annotation


def get_annotator_wise_responses(annotation_file=constants.group_conq_annot_data, manifestation="group", annotators=["Nakul", "Swathi", "Divya"], zero_mean=False):
    convq_scores_dict={}
    # TODO: Do ZERO-MEAN here - Under "derive_convq_scores_for_reponses" - Common function for both kappa calc and convq calc
    # TODO: OR CAN also do below under annotator-wise for loop (But need two fucntion edits)
    scored_annotations = derive_convq_scores_for_reponses(annotators, annotation_file, manifestation, reverse_scale=True, zero_mean=zero_mean)
    for i, annotator in enumerate(annotators):
        annotator_responses = scored_annotations.loc[scored_annotations['Annotator Name'] == annotator]
        sanity, annotator_responses = sanity_check_responses(annotator_responses, manifestation)
        if not sanity:
            print("!!!!!!!!!! WARNING - MISSING RESPONSES !!!!!!!!!!")
        if manifestation == "group":
            annotator_responses = annotator_responses.drop('Annotator Name', 1).sort_values(by=['Group ID'])[["Group ID", "group_convq"]]
        elif manifestation == "indiv":
            annotator_responses = annotator_responses.drop('Annotator Name', 1).sort_values(by=['Group ID','Individual ID'])[["Group ID",'Individual ID', "indiv_convq"]]
        # print("Number of responses afer clean for " + annotator + " = " + str(len(annotator_responses)))
        # annotator_responses.to_csv(manifestation+"-annotator"+str(i)+".csv", index=False)
        convq_scores_dict["annotator"+str(i)]=annotator_responses
    return convq_scores_dict

# ANNOTATOR EXTRACTOR and CLEANER
def get_groupwise_annotator_responses(annotation_file=constants.group_conq_annot_data, manifestation="group", annotators=["Nakul", "Swathi", "Divya"], zero_mean=False):
    convq_scores_dict={}
    # TODO: Do ZERO-MEAN here - Under "derive_convq_scores_for_reponses" - Common function for both kappa calc and convq calc
    raw_annotations = derive_convq_scores_for_reponses(annotators, annotation_file, manifestation, calculate_convq=False, reverse_scale=False, zero_mean=zero_mean)
    for i, annotator in enumerate(annotators):
        annotator_responses = raw_annotations.loc[raw_annotations['Annotator Name'] == annotator]
        sanity, annotator_responses = sanity_check_responses(annotator_responses, manifestation)
        if not sanity:
            print("!!!!!!!!!! WARNING - MISSING RESPONSES !!!!!!!!!!")
        if manifestation == "group":
            annotator_responses = annotator_responses.drop('Annotator Name', 1).sort_values(by=['Group ID'])
        elif manifestation == "indiv":
            annotator_responses = annotator_responses.drop('Annotator Name', 1).sort_values(by=['Group ID', 'Individual ID'])
        # print("Number of responses afer clean for " + annotator + " = " + str(len(annotator_responses)))
        convq_scores_dict["annotator"+str(i)]=annotator_responses
    return convq_scores_dict

