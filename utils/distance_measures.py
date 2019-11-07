import numpy as np
import dtw

def dtw_distance(traj1, traj2):
    diffs = []
    for d in range(traj1.shape[1]):
        diffs.append(dtw.dtw(traj1[:,d], traj2[:,d], dist=lambda x,y: (x-y)**2)[0])
    return np.mean(diffs)
