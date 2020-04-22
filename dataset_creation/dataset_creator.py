import utilities.data_read_util as reader

def generate_dataset_from_all_groups():
    all_groupids = reader.get_all_annotated_groups()["groupid"].values

    return []