from distutils.dir_util import copy_tree
import shutil

rem_videos = ["1_077",
"2_062",
"2_068",
"3_063",
"1_044",
"1_079",
"3_062",
"2_060",
"2_020",
"2_061",
"3_072",
"3_070",
"1_090",
"3_061",
"2_067",
"3_046",
"3_004",
"3_073",
"3_028",
"3_029",
"3_033",
"3_043",
"3_050",
"3_013",
"2_022",
"1_030",
"1_067",
"1_019",
"1_086",
"3_068",
"2_043",
"1_059",
"3_066",
"1_085",
"3_041",
"1_068",
"3_010",
"1_080",
"1_062",
"3_024"]

fromDirectory = "/Users/navinlr/Desktop/Thesis/Annotations/Annotation_Videos/"
toDirectory = "/Users/navinlr/Desktop/RemainingVideos/"

for video_folder in rem_videos:
    print(video_folder)
    from_folder = fromDirectory + str(video_folder)
    to_folder   = toDirectory + str(video_folder)
    shutil.copytree(from_folder, to_folder)

