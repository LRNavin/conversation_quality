import dataset_creation.dataset_creator as data_gen
import constants

_, data = data_gen.extract_and_save_dataset(dest_file=constants.features_dataset_path_v1,
                                            channels=["abs", "mag"],
                                            stat_features=["mean", "var"], spec_features=["psd"],
                                            windows=[1, 3, 5, 10, 15], step_size=0.5,
                                            sync_feats=["correl", "lag-correl", "mi", "norm-mi", "mimicry"],
                                            conver_feats=["sym-conv", "asym-conv", "global-conv"])