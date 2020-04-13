import numpy as np
from scipy import stats

a = np.array([[6, 8, 3, 0],
               [3, 2, 1, 7],
               [8, 1, 8, 4],
               [5, 3, 0, 5],
               [4, 7, 5, 9]])
m=stats.mode(a)
print(m.mode[0])