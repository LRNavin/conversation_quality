# from distutils.dir_util import copy_tree
# import shutil
#
# rem_videos = ["1_077",
# "2_062",
# "2_068",
# "3_063",
# "1_044",
# "1_079",
# "3_062",
# "2_060",
# "2_020",
# "2_061",
# "3_072",
# "3_070",
# "1_090",
# "3_061",
# "2_067",
# "3_046",
# "3_004",
# "3_073",
# "3_028",
# "3_029",
# "3_033",
# "3_043",
# "3_050",
# "3_013",
# "2_022",
# "1_030",
# "1_067",
# "1_019",
# "1_086",
# "3_068",
# "2_043",
# "1_059",
# "3_066",
# "1_085",
# "3_041",
# "1_068",
# "3_010",
# "1_080",
# "1_062",
# "3_024"]
#
# fromDirectory = "/Users/navinlr/Desktop/Thesis/Annotations/Annotation_Videos/"
# toDirectory = "/Users/navinlr/Desktop/RemainingVideos/"
#
# for video_folder in rem_videos:
#     print(video_folder)
#     from_folder = fromDirectory + str(video_folder)
#     to_folder   = toDirectory + str(video_folder)
#     shutil.copytree(from_folder, to_folder)
#


# Create signals
# n = 20
# x = pd.DataFrame(1.0 * np.random.rand(n, 1), range(0, n))
# y = pd.DataFrame(1.0 * np.random.rand(n, 1), range(0, n))
#

# x = pd.read_csv('/Users/navinlr/Desktop/sampl1.csv', sep=',', header=None)
# y = pd.read_csv('/Users/navinlr/Desktop/sampl2.csv', sep=',', header=None)


import pingouin as pg
import pandas as pd
import numpy as np

arr= [[-0.31791908,  0.69364162, -0.69364162, -0.2716763,   0.23699422, -0.10404624, -0.13583815,  0.67630058, -0.28323699,  0.26589595],
      [ 0.39117647,  0.31470588, -0.35588235,  0.51764706, -0.27647059,  0.47941176, 0.32058824,  0.35294118,  0.21176471 , 1.73823529]]

arr = [[-0.31791908, -0.30635838,  0.30635838,  0.7283237 , -0.76300578  ,0.89595376 ,-0.13583815 , 0.67630058 , 0.71676301 , 2.26589595],
        [ 1.39117647 , 1.31470588, -1.35588235 , 0.51764706, -1.27647059,  1.47941176, 0.32058824 , 1.35294118 , 1.21176471, -1.26176471]]

df = pd.DataFrame(np.array(arr))
print(pg.cronbach_alpha(data=df))

# Current Alpha: 0.7131577882585368
