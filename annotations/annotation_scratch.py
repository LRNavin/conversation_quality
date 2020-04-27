import pandas as pd
import numpy as np
from scipy import stats
import os

manifest="group"

for i in range(3):
    annotator=str(i)
    annotation_file = os.path.abspath(manifest +"-annotator" + annotator + ".csv")
    print("Annotator: " + annotator)
    raw_annotations = pd.read_csv(annotation_file)
    convq_col =raw_annotations.columns[-1]

    print("MINIMUM  - " + str(min(raw_annotations[convq_col])))
    print("MAXIMUM  - " + str(max(raw_annotations[convq_col])))
    print("AVERAGE  - " + str(np.mean(raw_annotations[convq_col])))
    print("VARIANCE - " + str(np.var(raw_annotations[convq_col])))
    print("MODE     - " + str(stats.mode(raw_annotations[convq_col])[0][0]))