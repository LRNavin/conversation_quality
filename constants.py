# Dataset Constants
dataset_base_folder = "/Users/navinlr/Desktop/MatchMakersStudentEdition/"
dataset_file_path   = "Acceleration/"
annotations_path    = "annotations/"

#Temporary Data Storage Path
temp_storage_path   = "/Users/navinlr/Desktop/Thesis/code_base/conversation_quality/temporary_data_storage/"
temp_dataset_store  = temp_storage_path + "dataset/"
temp_fform_store    = temp_dataset_store + "groups.pickle"
temp_grps_day       = temp_dataset_store + "groups_day"

final_grps_data     = temp_dataset_store + "all_groups_final.csv"
final_grps_raw_data = temp_dataset_store + "final_groups_raw.csv"

data_stat = temp_dataset_store + "initial_groups_raw.csv"
missing_acc_stat = temp_dataset_store + "missing_acc_details.csv"

temp_grp_size_store = temp_dataset_store + "group_size.json"
temp_windowed_group_data = temp_dataset_store + "temp_group_data.txt"

dataset_name        = "Mingle_30minSeg.mat"
data_name           = "Mingle_30min"

# F-Formation Details data Constants
fform_annot_data    = dataset_base_folder + annotations_path + "groups_GT.ods"

# Conversation Quality Annotations
group_conq_annot_data = dataset_base_folder + annotations_path + "GroupConvQ-final.csv"
indiv_conq_annot_data = dataset_base_folder + annotations_path + "IndivConvQ-final.csv"

# Final-Annotations -> F-Form and ConvQ Labels
# #TODO:ASSUMPTION HERE -> ANNOTATIONS DONE, BUT HAVE TO DO ANNOTATION to FINAL F-FORM + LABELS
fform_gt_data       = temp_dataset_store + "all_groups_final.csv"