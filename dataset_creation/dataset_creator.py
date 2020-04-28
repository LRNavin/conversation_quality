import utilities.data_read_util as reader
import feature_extract.feature_preproc as processor
import feature_extract.statistics_extractor as base_feat_extractor
import feature_extract.synchrony_extractor as sync_extractor
import feature_extract.convergence_extractor as conv_extractor
import feature_extract.features_postproc as post_processor
import feature_extract.group_features_extractor as group_feat_extractor
import annotations.analysis as annotation_reader
import utilities.data_read_util as data_util
import constants as const

from tqdm import tqdm
import pandas as pd
import numpy as np


def generate_indiv_dataset_from_all_groups(channels,
                                           stat_features, spec_features, windows, step_size,
                                           sync_feats, conver_feats):
    all_groupids = reader.get_all_annotated_groups()["groupid"].values
    groupwise_indiv_acc = {}
    for group_id in tqdm(all_groupids):
        group_acc = processor.get_accelerometer_channels_for(group_id=group_id, channels=channels)

        # Extract Base Statistical, Spectral Features
        group_acc = base_feat_extractor.get_base_features_for(group_accel_data=group_acc,
                                                              stat_features=stat_features,
                                                              spec_features=spec_features,
                                                              windows=windows, step_size=step_size)
        # Extract Synchrony features
        group_pairwise_sync = sync_extractor.get_synchrony_features_for(group_acc, features=sync_feats)
        # Extract Convergence features
        group_pairwise_conv = conv_extractor.get_convergence_features_for(group_acc, features=conver_feats)
        group_pairwise_features = post_processor.concatenate_pairwise_features(group_pairwise_sync, group_pairwise_conv)

        groupwise_indiv_acc[group_id] = group_pairwise_features
    return groupwise_indiv_acc


def extract_and_save_dataset(channels, stat_features, spec_features, windows, step_size, sync_feats, conver_feats):
    features_data = generate_indiv_dataset_from_all_groups(channels = channels,
                                                         stat_features=stat_features, spec_features=spec_features,
                                                         windows=windows, step_size=step_size,
                                                         sync_feats=sync_feats, conver_feats=conver_feats)

    data_util.save_pickle(features_data, const.features_dataset_path)
    print("~~~~~~~~~~~~~~~~~~~~ data saved ~~~~~~~~~~~~~~~~~~~~~")
    return True, features_data

#BASED ON MISS INDIV ACC and LOW ANNOTATOR RELIABILITY
# Groups and Indiividuals - Data selection based on agreeability scores:
def get_annotation_realiable_labels(agreeability_thresh, manifest, annotators):
    if manifest == "group":
        file = const.group_conq_annot_data
    else:
        file = const.indiv_conq_annot_data
    score, final_average_score_all, groups_label = annotation_reader.get_final_convq_score_for(annotation_file=file,
                                                                                                    manifestation=manifest,
                                                                                                    annotators=annotators)
    filtered_ind = np.where(np.array(final_average_score_all) >= agreeability_thresh)
    reliable_ids = np.array(groups_label)[filtered_ind[0]].tolist()
    reliable_scores = np.array(final_average_score_all)[filtered_ind[0]].tolist()
    return reliable_ids, reliable_scores

# THE ONLY DIRTY CODE - WROTE IN A BAD MOOD :(
# TODO: CHECK/DEBUG once whether IDs and Data are in same Order
# TODO: Debug on Reliability filter
def filter_dataset(features_data_path, missing_data_thresh, agreeability_thresh, manifest, annotators, only_involved_pairs):
    features_data = reader.load_pickle(features_data_path)
    missing_filtered_data = {}

    missign_acc_checklist = pd.read_csv(const.missing_acc_stat)
    filtered_group_ids = missign_acc_checklist.loc[missign_acc_checklist['percent_miss'] < missing_data_thresh]["group_ids"].values.tolist()
    # print(filtered_group_ids)
    # Remove Incomplete Acc Data
    # While preparing Group_level and Indiv modelling dataset
    for group_id in features_data.keys():
        if group_id in filtered_group_ids: # If not many missing accelero as per threshold set above
            group_data = features_data[group_id]
            filtered_group_data = {}
            for pairs in group_data.keys():
                if len(group_data[pairs]) != 0: # If current pair data not empty
                    filtered_group_data[pairs]=group_data[pairs]
            missing_filtered_data[group_id] = filtered_group_data

    agreeability_filtered_data = {}

    # Remove LOW ANNOTATOR RELIABILITY
    reliable_ids, reliable_scores = get_annotation_realiable_labels(agreeability_thresh, manifest, annotators)
    if manifest == "group": # While preparing Group_level modelling dataset
        for group_id in reliable_ids: # for every reliable group
            if group_id in missing_filtered_data: # If statement required, Coz All relaiable groups may not have required amount of accelro data
                agreeability_filtered_data[group_id] = missing_filtered_data[group_id]
    else: # While preparing Indiv-level modelling dataset
        for indiv_id in reliable_ids:
            grp_id, ind_id = indiv_id.split("_")[0]+"_"+indiv_id.split("_")[1], indiv_id.split("_")[2]
            indiv_pairs_data = {}
            if grp_id in missing_filtered_data:# and ind_id in missing_filtered_data[grp_id]: # If statement required, Coz All relaiable groups may not have required amount of accelro data
                if only_involved_pairs: # Use only pairs the individual is involved in.
                    group_pairs_list = list(missing_filtered_data[grp_id].keys())
                    # Get Only inlvolved pairs
                    for pair in missing_filtered_data[grp_id].keys():
                        if ind_id not in pair: # Take only pairwise features related to INDIV - ind_id for inidivudal ConvQ modelling
                            group_pairs_list.remove(pair)
                    # Store Only inlvolved pairs
                    for required_pairs in group_pairs_list:
                        indiv_pairs_data[required_pairs] = missing_filtered_data[grp_id][required_pairs]
                else:
                    indiv_pairs_data = missing_filtered_data[grp_id]
                agreeability_filtered_data[grp_id + "_" +ind_id] = indiv_pairs_data
    return agreeability_filtered_data, reliable_ids, reliable_scores