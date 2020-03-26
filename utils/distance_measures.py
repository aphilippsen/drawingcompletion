import chainer
import numpy as np
import dtw

def distance_measure(target_traj, generated_traj, method = 'mse'):
    if method == 'mse':
        return chainer.functions.mean_squared_error(target_traj[1:,:], generated_traj[:-1,:]).data.tolist()
    elif method == 'dtw':
        euclidean_distance = lambda x, y: (x-y)**2
        val = 0
        for d in range(target_traj.shape[1]):
            val += dtw.dtw(target_traj[1:,d], generated_traj[:-1,d], dist=euclidean_distance)[0]
        return val

