sub_channels_dict = {
    "raw" : ["x_raw", "y_raw", "z_raw"],
    "abs" : ["x_abs", "y_abs", "z_abs"],
    "mag" : ["mag"]
}

sub_features_dict= {
    "correl" : ["corr"], #1
    "lagcorr" : ["min_lagcorr", "max_lagcorr", "argmin_lagcorr", "argmax_lagcorr"], #4
    "mi" : ["mi"], #1
    "mimicry" : ["min_lead_mimicry", "max_lead_mimicry", "mean_lead_mimicry", "var_lead_mimicry",
                 "min_lag_mimicry", "max_lag_mimicry", "mean_lag_mimicry", "var_lag_mimicry"], #8
    "coherence": ["min_coherence", "max_coherence"], #2
    "granger": ["granger"], #1
    "symconv" : ["symconv"], #1
    "asymconv" : ["lead_asymconv", "lag_asymconv"], #2
    "globconv" : ["globconv"] #1
}


def name_sync_features_for_dataframe(channels, sync_feats, conver_feats, grp_feats):
    feature_names=[]
    # Synchrony is done first, before Convergence
    behav_feats = sync_feats + conver_feats
    for grp_feat in grp_feats:
        for behav_feat in behav_feats:
            for sub_behav_feat in sub_features_dict[behav_feat]:
                for channel in channels:
                    for sub_channel in sub_channels_dict[channel]:
                        curr_feat_name = sub_channel + "-" + sub_behav_feat + "--" + grp_feat
                        feature_names.append(curr_feat_name)
    print("Total #Features = " + str(len(feature_names)))
    # print(feature_names) - #TODO:Cross-Check after GreenLight. But in First check, it seems to be right.
    return feature_names

#Hard-Coded per dataset :( - Bad one Navin
def get_feature_names_for_dataset_7(manifest):
    return name_sync_features_for_dataframe(channels=["raw", "abs", "mag"],
                                             sync_feats=["correl", "lagcorr", "mi", "mimicry"],
                                             conver_feats=["symconv", "asymconv", "globconv"],
                                             grp_feats=["min", "max", "mean", "var", "median", "mode"])

def get_feature_names_for_dataset_13(manifest):
    return name_sync_features_for_dataframe(channels=["raw", "abs", "mag"],
                                             sync_feats=["coherence", "granger"],
                                             conver_feats=[],
                                             grp_feats=["min", "max", "mean", "var", "median", "mode"])
def get_feature_names_for_tt(manifest):
    if manifest == "indiv":
        return ["tt-conv_eq", "tt-#turns", "tt-%talk", "tt-mean_turn",
                "tt-mean_silence", "tt-%silence", "tt-#bc", "tt-%overlap",
                "tt-#suc_interupt", "tt-#un_interupt"]
    else:
        return ["tt-var_#turn", "tt-var_dturn", "tt-conv_eq", "tt-mean_silence",
                "tt-%silence", "tt-#bc", "tt-%overlap", "tt-#suc_interupt", "tt-#un_interupt"]


#Testing
# name_sync_features_for_dataframe(channels=["raw", "abs", "mag"],
#                                  sync_feats=["correl", "lagcorr", "mi", "mimicry"],
#                                  conver_feats=["symconv", "asymconv", "globconv"],
#                                  grp_feats=["min", "max", "mean", "var", "median", "mode"])