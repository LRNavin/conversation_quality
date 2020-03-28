import os

# Create directory
dir_name = "/Users/navinlr/Desktop/Thesis/Annotations/Annotation_Videos/"

day=1
start_count=0
end_count=25

for i in range(start_count, end_count):
    folder_name = dir_name + str((day*1000)+i)
    sub_folder_grp = folder_name + "/group"
    sub_folder_ind = folder_name + "/individual"
    try:
        # Create target Directory
        os.mkdir(folder_name)
        print("Main Directory ", folder_name, " Created ")
        os.mkdir(sub_folder_grp)
        print("Sub Directory ", folder_name, " Created ")
        os.mkdir(sub_folder_ind)
        print("Sub Directory ", folder_name, " Created ")
    except FileExistsError:
        print("Directory ", folder_name, " already exists")