from __future__ import print_function
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import time

from sklearn.manifold import TSNE

from modeling import dataset_provider as data_gen
import constants

# Variables for baseline
random_seed=20
manifest="indiv"
data_split_per=.40
missing_data_thresh=50.0 #(in percent)
convq_thresh=3.0
agreeability_thresh=.2
annotators=["Divya", "Nakul"]#, "Swathi"]
only_involved_pairs=True

label_type = "hard"
model_type = "rand-for"
zero_mean  = False

dataset=constants.features_dataset_path_v1

def process_convq_labels(y, label_type="soft"):
    print("Data-type of labels - " + str(type(y)))
    if label_type=="soft":
        y=list(np.around(np.array(y),2))
    else:
        y=list(np.where(np.array(y) <= convq_thresh, 0, 1))
        print("ConvQ Classes Distribution : (Total = "+ str(len(y)) +")")
        print("High Quality Conv = " + str(sum(y)))
        print("Low Quality Conv = " + str(len(y)-sum(y)))
    return y

# Data Read
X, y, ids = data_gen.get_dataset_for_experiment(dataset=dataset,
                                                    manifest=manifest,
                                                    missing_data_thresh=missing_data_thresh,
                                                    agreeability_thresh=agreeability_thresh,
                                                    annotators=annotators,
                                                    only_involved_pairs=only_involved_pairs,
                                                    zero_mean=zero_mean)



# Label Prep
# Hard/Soft Labels
y = process_convq_labels(y, label_type)

time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=10000)
tsne_results = tsne.fit_transform(X)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

data = {'tsne-2d-one':tsne_results[:,0], 'tsne-2d-two':tsne_results[:,1], 'y':y}
df = pd.DataFrame.from_dict(data)

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", 2),
    data=df,
    legend="full",
    alpha=0.5
)
plt.show()