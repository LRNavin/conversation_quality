
# Groups and Indiividuals - Data selection based on agreeability scores:
def get_valid_indiv_ids(cutoff=0.0):
    selected_indivs = []
    selected_indivs_group = []
    return selected_indivs, selected_indivs_group

def get_valid_group_ids(cutoff=0.0):
    selected_groups = []
    return selected_groups

# Indiv-Level Dataset Generation
def get_indiv_level_labels():
    selected_indivs, respective_group = get_valid_indiv_ids()
    y = []
    return y

def get_indiv_level_features():
    selected_indivs, respective_group = get_valid_indiv_ids()
    X = []
    return X

# Group-Level Dataset Generation
def get_group_level_labels():
    selected_groups = get_valid_group_ids()
    y = []
    return y

def get_group_level_features():
    selected_groups = get_valid_group_ids()
    X = []
    return X

def get_group_level_dataset():
    X, y = get_group_level_features(), get_group_level_labels()
    return X, y

def get_indiv_level_dataset():
    X, y = get_indiv_level_features(), get_indiv_level_labels()
    return X, y


# Public Function - Receive external requests
def get_dataset_for_experiment(manifest="group"):
    if manifest == "group":
        return get_group_level_dataset()
    else:
        return get_indiv_level_dataset()