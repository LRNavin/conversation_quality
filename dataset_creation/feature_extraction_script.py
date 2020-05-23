import sys
sys.path.insert(0, '/Users/navinlr/Desktop/Thesis/code_base/conversation_quality')

import dataset_creation.dataset_creator as data_gen
import constants

# features_dataset.pickle
# _, data = data_gen.extract_and_save_dataset(dest_file=constants.features_dataset_path_v1,
#                                             channels=["abs", "mag"],
#                                             stat_features=["mean", "var"], spec_features=["psd"],
#                                             windows=[1, 5, 10, 15], step_size=0.5,
#                                             sync_feats=["correl", "lag-correl", "mi", "norm-mi", "mimicry"],
#                                             conver_feats=["sym-conv", "asym-conv", "global-conv"])

# features_dataset_v1.pickle
# _, data = data_gen.extract_and_save_dataset(dest_file=constants.features_dataset_path_v1,
#                                             channels=["abs", "mag"],
#                                             stat_features=["mean", "var"], spec_features=["psd"],
#                                             windows=[1, 3, 5, 10, 15], step_size=0.5,
#                                             sync_feats=["correl", "lag-correl", "mi", "norm-mi", "mimicry"],
#                                             conver_feats=["sym-conv", "asym-conv", "global-conv"])

# features_dataset_v2.pickle [Noramlized]
# _, data = data_gen.extract_and_save_dataset(dest_file=constants.features_dataset_path_v2,
#                                             acc_norm=True,
#                                             channels=["abs", "mag"],
#                                             stat_features=["mean", "var"], spec_features=["psd"],
#                                             windows=[1, 3, 5, 10, 15], step_size=0.5,
#                                             sync_feats=["correl", "lag-correl", "mi", "norm-mi", "mimicry"],
#                                             conver_feats=["sym-conv", "asym-conv", "global-conv"])


# features_dataset_v3.pickle [Modified Lag-Correl, MI & Noramlized & Includes Raw values]
# _, data = data_gen.extract_and_save_dataset(dest_file=constants.features_dataset_path_v3,
#                                             acc_norm=True,
#                                             channels=["abs", "mag"],
#                                             stat_features=["mean", "var"], spec_features=["psd"],
#                                             windows=[1, 3, 5, 10, 15, 0], step_size=0.5,
#                                             sync_feats=["correl", "lag-correl", "mi", "norm-mi", "mimicry"],
#                                             conver_feats=["sym-conv", "asym-conv", "global-conv"])
#
#
# # features_dataset_v4.pickle [Modified Lag-Correl, MI & Non-Noramlized & Includes Raw values]
# _, data = data_gen.extract_and_save_dataset(dest_file=constants.features_dataset_path_v4,
#                                             acc_norm=False,
#                                             channels=["abs", "mag"],
#                                             stat_features=["mean", "var"], spec_features=["psd"],
#                                             windows=[0, 1, 3, 5, 10, 15], step_size=0.5,
#                                             sync_feats=["correl", "lag-correl", "mi", "norm-mi", "mimicry"],
#                                             conver_feats=["sym-conv", "asym-conv", "global-conv"])


# features_dataset_v5.pickle [Noramlized & Only Raw values]
# _, data = data_gen.extract_and_save_dataset(dest_file=constants.features_dataset_path_v5,
#                                             acc_norm=True,
#                                             channels=["abs", "mag"],
#                                             stat_features=["mean", "var"], spec_features=["psd"],
#                                             windows=[0], step_size=0.5,
#                                             sync_feats=["correl", "lag-correl", "mi", "norm-mi", "mimicry"],
#                                             conver_feats=["sym-conv", "asym-conv", "global-conv"])

# features_dataset_v6.pickle [Modified Lag-Correl, MI & Noramlized & Only Raw values]
_, data = data_gen.extract_and_save_dataset(dest_file=constants.features_dataset_path_v7,
                                            acc_norm=True,
                                            channels=["abs", "mag"],
                                            stat_features=["mean", "var"], spec_features=["psd"],
                                            windows=[0], step_size=0.5,
                                            sync_feats=["correl", "lag-correl", "mi", "norm-mi", "mimicry"],
                                            conver_feats=["sym-conv", "asym-conv", "global-conv"])

# features_dataset_v6.pickle [Modified Lag-Correl, MI & Noramlized & Only Window 5]
_, data = data_gen.extract_and_save_dataset(dest_file=constants.features_dataset_path_v8,
                                            acc_norm=True,
                                            channels=["abs", "mag"],
                                            stat_features=["mean", "var"], spec_features=["psd"],
                                            windows=[5], step_size=0.5,
                                            sync_feats=["correl", "lag-correl", "mi", "norm-mi", "mimicry"],
                                            conver_feats=["sym-conv", "asym-conv", "global-conv"])