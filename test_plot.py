import os
import numpy as np
import matplotlib.pyplot as plt

data_set_name = '2019-11-all'
eval_dir = "./results/evaluation/" + data_set_name + "/"


inners001 = np.load('./results/evaluation/' + data_set_name + '/inference-0.01/inner_dist_mean.npy')
inners01 = np.load('./results/evaluation/' + data_set_name + '/inference-0.1/inner_dist_mean.npy')
inners1 = np.load('./results/evaluation/' + data_set_name + '/inference-1/inner_dist_mean.npy')
inners10 = np.load('./results/evaluation/' + data_set_name + '/inference-10/inner_dist_mean.npy')
inners100 = np.load('./results/evaluation/' + data_set_name + '/inference-100/inner_dist_mean.npy')

""" This is variability between patterns which is not so interesting here
inners100var = np.load('./results/evaluation/' + data_set_name + '/inference-100/inner_dist_var.npy')
inners10var = np.load('./results/evaluation/' + data_set_name + '/inference-10/inner_dist_var.npy')
inners1var = np.load('./results/evaluation/' + data_set_name + '/inference-1/inner_dist_var.npy')
inners001var = np.load('./results/evaluation/' + data_set_name + '/inference-0.01/inner_dist_var.npy')
inners01var = np.load('./results/evaluation/' + data_set_name + '/inference-0.1/inner_dist_var.npy')
"""

plt.figure()

for i in range(inners001.shape[0]):
    plt.plot(np.arange(90), inners001[i,:], color='blue')
    plt.plot(np.arange(90), inners01[i,:], color='purple')
    plt.plot(np.arange(90), inners1[i,:], color='green')
    plt.plot(np.arange(90), inners10[i,:], color='orange')
    plt.plot(np.arange(90), inners100[i,:], color='red')

plt.errorbar(np.arange(90), np.mean(inners001,axis=0), yerr=np.sqrt(np.var(inners001,axis=0)), color='blue')
plt.errorbar(np.arange(90), np.mean(inners01,axis=0), yerr=np.sqrt(np.var(inners01,axis=0)), color='purple')
plt.errorbar(np.arange(90), np.mean(inners1,axis=0), yerr=np.sqrt(np.var(inners1,axis=0)), color='green')
plt.errorbar(np.arange(90), np.mean(inners10,axis=0), yerr=np.sqrt(np.var(inners10,axis=0)), color='orange')
plt.errorbar(np.arange(90), np.mean(inners100,axis=0), yerr=np.sqrt(np.var(inners100,axis=0)), color='red')

plt.savefig(os.path.join(eval_dir, "inner-distance-per-H.png"))


# export separately all 0.01... etc per time scale... no better do that already in the eval-repr script


# check, are they all the same scale due to scaling??? should be at least m=0 and v=1




