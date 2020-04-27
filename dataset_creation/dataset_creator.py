import utilities.data_read_util as reader
import feature_extract.feature_preproc as processor
import feature_extract.statistics_extractor as base_feat_extractor
import feature_extract.synchrony_extractor as sync_extractor
import feature_extract.convergence_extractor as conv_extractor
import feature_extract.features_postproc as post_processor
import utilities.data_read_util as data_util
import constants as const

from tqdm import tqdm


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

#TODO: DATA FILTERATION - BASED ON MISS INDIV ACC and LOW ANNOTATOR RELIABILITY
def filter_dataset(features_data, rules=["missing_acc", "low_reliaiblity"]):
    for rule in rules:
        continue
    return features_data


features_data = extract_and_save_dataset(channels=["abs", "mag"],
                                         stat_features=["mean", "var"], spec_features=["psd"], windows=[1, 5, 10, 15], step_size=0.5,
                                         sync_feats=["correl", "lag-correl", "mi", "norm-mi", "mimicry"], conver_feats=["sym-conv", "asym-conv", "global-conv"])