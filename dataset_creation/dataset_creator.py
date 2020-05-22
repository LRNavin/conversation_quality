import utilities.data_read_util as reader
import feature_extract.feature_preproc as processor
import feature_extract.statistics_extractor as base_feat_extractor
import feature_extract.synchrony_extractor as sync_extractor
import feature_extract.convergence_extractor as conv_extractor
import feature_extract.features_postproc as post_processor
import annotations.analysis as annotation_reader
import utilities.data_read_util as data_util
import constants as const

from tqdm import tqdm
import pandas as pd
import numpy as np


def generate_indiv_dataset_from_all_groups(channels,
                                           acc_norm,
                                           stat_features, spec_features, windows, step_size,
                                           sync_feats, conver_feats):
    all_groupids = reader.get_all_annotated_groups()["groupid"].values
    groupwise_indiv_acc = {}
    for group_id in tqdm(all_groupids):
        group_acc = processor.get_accelerometer_channels_for(group_id=group_id, acc_norm=acc_norm, channels=channels)

        # Extract Base Statistical, Spectral Features
        group_acc = base_feat_extractor.get_base_features_for(group_accel_data=group_acc,
                                                              stat_features=stat_features,
                                                              spec_features=spec_features,
                                                              windows=windows, step_size=step_size)
        print("Base Features Extracted")

        # Extract Synchrony features
        group_pairwise_sync = sync_extractor.get_synchrony_features_for(group_acc, features=sync_feats)
        print("Synchrony Features Extracted")

        # Extract Convergence features
        group_pairwise_conv = conv_extractor.get_convergence_features_for(group_acc, features=conver_feats)
        print("Convergence Features Extracted")

        group_pairwise_features = post_processor.concatenate_pairwise_features(group_pairwise_sync, group_pairwise_conv)

        groupwise_indiv_acc[group_id] = group_pairwise_features
    return groupwise_indiv_acc


def extract_and_save_dataset(dest_file, acc_norm, channels, stat_features, spec_features, windows, step_size, sync_feats, conver_feats):
    features_data = generate_indiv_dataset_from_all_groups(channels = channels,
                                                           acc_norm = acc_norm,
                                                           stat_features=stat_features, spec_features=spec_features,
                                                           windows=windows, step_size=step_size,
                                                           sync_feats=sync_feats, conver_feats=conver_feats)

    data_util.save_pickle(features_data, dest_file)
    print("~~~~~~~~~~~~~~~~~~~~ data saved ~~~~~~~~~~~~~~~~~~~~~")
    return True, features_data

#BASED ON MISS INDIV ACC and LOW ANNOTATOR RELIABILITY
# Groups and Indiividuals - Data selection based on agreeability scores:
# USed for analysis on reliable convq scores - e.g. statistical tests
def get_annotation_realiable_labels(agreeability_thresh, manifest, annotators, zero_mean):
    if manifest == "group":
        file = const.group_conq_annot_data
    else:
        file = const.indiv_conq_annot_data

    score_convq, score_kappa, final_average_convq, final_average_kappa, groups_label = annotation_reader.get_final_convq_score_for(annotation_file=file,
                                                                                                    manifestation=manifest,
                                                                                                    annotators=annotators, zero_mean=zero_mean)
    # print(groups_label)
    # print(final_average_convq)
    # print(final_average_kappa)
    filtered_ind    = np.where(np.array(final_average_kappa) >= agreeability_thresh)
    reliable_ids    = np.array(groups_label)[filtered_ind[0]].tolist()
    reliable_convqs = np.array(final_average_convq)[filtered_ind[0]].tolist()
    reliable_kappas = np.array(final_average_kappa)[filtered_ind[0]].tolist()

    return reliable_ids, reliable_convqs, reliable_kappas

def is_individual_accelro_present_in(missing_filtered_data, group_id, indv_id):
    individual_acc_available = False
    for pair in missing_filtered_data[group_id].keys():
        if indv_id in pair:
            individual_acc_available = True
            break
    return individual_acc_available

def get_pairs_involved_by_indiv(missing_filtered_data, grp_id, ind_id, indiv_pairs_data):
    group_pairs_list = list(missing_filtered_data[grp_id].keys())
    # print("Group ID - " + str(grp_id))
    # print("All - Pairs - " + str(group_pairs_list))
    # print("INdiv ID - " + str(ind_id))

    # Get Only inlvolved pairs
    for pair in missing_filtered_data[grp_id].keys():
        if ind_id not in pair:  # Take only pairwise features related to INDIV - ind_id for inidivudal ConvQ modelling
            group_pairs_list.remove(pair)

    # Store Only inlvolved pairs
    for i, required_pair in enumerate(group_pairs_list):
        if ind_id == required_pair.split("_")[0]:
            # print("Involved - Pairs - " + str(i) + ", " + str(required_pair))
            indiv_pairs_data[required_pair] = missing_filtered_data[grp_id][required_pair]
    return indiv_pairs_data


# THE ONLY DIRTY CODE - WROTE IN A BAD MOOD :(
def filter_dataset(features_data_path, missing_data_thresh, agreeability_thresh, manifest, annotators, only_involved_pairs, zero_mean):
    features_data = reader.load_pickle(features_data_path)
    missing_filtered_data = {}

    missing_acc_checklist = pd.read_csv(const.missing_acc_stat)
    filtered_group_ids = missing_acc_checklist.loc[missing_acc_checklist['percent_miss'] < missing_data_thresh]["group_ids"].values.tolist()
    # Remove Incomplete Acc Data
    # While preparing Group_level and Indiv modelling dataset
    for group_id in features_data.keys():
        if group_id in filtered_group_ids: # If not many missing accelero as per threshold set above
            # print("For Group - " + str(group_id))
            group_data = features_data[group_id]
            # print("All True pairs - " + str(group_data.keys()))
            filtered_group_data = {}
            for pairs in group_data.keys():
                if len(group_data[pairs]) != 0: # If current pair data not empty
                    filtered_group_data[pairs]=group_data[pairs]
            # print("All avaialble pairs - " + str(filtered_group_data.keys()))
            missing_filtered_data[group_id] = filtered_group_data
    print("Number of Groups (After removing missing data) - " + str(len(missing_filtered_data.keys())))

    agreeability_filtered_data = {}
    # Remove LOW ANNOTATOR RELIABILITY
    reliable_ids, reliable_convqs, reliable_kappas = get_annotation_realiable_labels(agreeability_thresh, manifest, annotators, zero_mean)
    final_item_ids = []
    final_convq_scores = []
    # print("Total Reliable Data-points - " + str(len(reliable_ids)))
    if manifest == "group": # While preparing Group_level modelling dataset
        for i, group_id in enumerate(reliable_ids): # for every reliable group
            if group_id in missing_filtered_data: # If statement required, Coz All relaiable groups may not have required amount of accelro data
                agreeability_filtered_data[group_id] = missing_filtered_data[group_id]
                final_item_ids.append(reliable_ids[i])
                final_convq_scores.append(reliable_convqs[i])
    else: # While preparing Indiv-level modelling dataset
        for i, indiv_id in enumerate(reliable_ids):
            grp_id, ind_id = indiv_id.split("_")[0]+"_"+indiv_id.split("_")[1], indiv_id.split("_")[2]
            indiv_pairs_data = {}
            # Condition - 1 : USING ONLY DATA FROM COMPLETE GROUPs - Coz FEATURES ARE PAIRWISE
            # Condition - 2 : Though Indiv Label reliable, Accelerometer Missing :(
            if grp_id in missing_filtered_data and is_individual_accelro_present_in(missing_filtered_data, grp_id, ind_id):
                if only_involved_pairs: # Use only pairs the individual is involved in.
                    indiv_pairs_data = get_pairs_involved_by_indiv(missing_filtered_data, grp_id, ind_id, indiv_pairs_data)
                else:
                    indiv_pairs_data = missing_filtered_data[grp_id]
                agreeability_filtered_data[grp_id + "_" +ind_id] = indiv_pairs_data
                final_item_ids.append(reliable_ids[i])
                final_convq_scores.append(reliable_convqs[i])

    print("Number of Final Data-points (After removing unreliable annotation data) - " + str(len(agreeability_filtered_data.keys())))

    return agreeability_filtered_data, final_item_ids, final_convq_scores

# TODO: funcitons can be converted to more efficient lambda / vectorised methods
def get_all_group_sizes_for_group(group_ids):
    group_sizes=[]
    for id in group_ids:
        group_sizes.append(len(reader.get_members_in_f_form(group_id=id)))
    return group_sizes

def extract_grp_ids_from_indiv_ids(indiv_ids):
    group_ids = []
    for curr_indiv_id in indiv_ids:
        group_ids.append(curr_indiv_id.split("_")[0] + "_" + curr_indiv_id.split("_")[1])
    return group_ids

def get_group_sizes_for_ids(ids, manifest):
    if manifest == "group":
        return get_all_group_sizes_for_group(ids)
    else:
        return get_all_group_sizes_for_group(extract_grp_ids_from_indiv_ids(ids))