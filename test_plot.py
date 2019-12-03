import os
import numpy as np
import matplotlib.pyplot as plt

"""
data_set_name = '2019-11-all'
eval_dir = "./results/evaluation/" + data_set_name + "/"

inners0001 = np.load('./results/evaluation/' + data_set_name + '/inference-0.001/inner_dist_mean.npy')
inners001 = np.load('./results/evaluation/' + data_set_name + '/inference-0.01/inner_dist_mean.npy')
inners01 = np.load('./results/evaluation/' + data_set_name + '/inference-0.1/inner_dist_mean.npy')
inners1 = np.load('./results/evaluation/' + data_set_name + '/inference-1/inner_dist_mean.npy')
inners10 = np.load('./results/evaluation/' + data_set_name + '/inference-10/inner_dist_mean.npy')
inners100 = np.load('./results/evaluation/' + data_set_name + '/inference-100/inner_dist_mean.npy')
inners1000 = np.load('./results/evaluation/' + data_set_name + '/inference-1000/inner_dist_mean.npy')

% This is variability between patterns which is not so interesting here
%inners100var = np.load('./results/evaluation/' + data_set_name + '/inference-100/inner_dist_var.npy')
%inners10var = np.load('./results/evaluation/' + data_set_name + '/inference-10/inner_dist_var.npy')
%inners1var = np.load('./results/evaluation/' + data_set_name + '/inference-1/inner_dist_var.npy')
%inners001var = np.load('./results/evaluation/' + data_set_name + '/inference-0.01/inner_dist_var.npy')
%inners01var = np.load('./results/evaluation/' + data_set_name + '/inference-0.1/inner_dist_var.npy')

colors = ['blue', 'purple', 'turquoise', 'green', 'red', 'orange', 'pink']

plt.figure()

for i in range(inners001.shape[0]):
    plt.plot(np.arange(90), inners0001[i,:], color='blue')
    plt.plot(np.arange(90), inners001[i,:], color='purple')
    plt.plot(np.arange(90), inners01[i,:], color='turquoise')
    plt.plot(np.arange(90), inners1[i,:], color='green')
    plt.plot(np.arange(90), inners10[i,:], color='orange')
    plt.plot(np.arange(90), inners100[i,:], color='red')
    plt.plot(np.arange(90), inners1000[i,:], color='pink')

plt.errorbar(np.arange(90), np.mean(inners001,axis=0), yerr=np.sqrt(np.var(inners001,axis=0)), color='blue')

plt.errorbar(np.arange(90), np.mean(inners001,axis=0), yerr=np.sqrt(np.var(inners001,axis=0)), color='blue')
plt.errorbar(np.arange(90), np.mean(inners01,axis=0), yerr=np.sqrt(np.var(inners01,axis=0)), color='purple')
plt.errorbar(np.arange(90), np.mean(inners1,axis=0), yerr=np.sqrt(np.var(inners1,axis=0)), color='green')
plt.errorbar(np.arange(90), np.mean(inners10,axis=0), yerr=np.sqrt(np.var(inners10,axis=0)), color='orange')
plt.errorbar(np.arange(90), np.mean(inners100,axis=0), yerr=np.sqrt(np.var(inners100,axis=0)), color='red')

plt.savefig(os.path.join(eval_dir, "inner-distance-per-H.png"))


# export separately all 0.01... etc per time scale... no better do that already in the eval-repr script


# check, are they all the same scale due to scaling??? should be at least m=0 and v=1

"""





# PLOT TRAINING DATA

train_data = np.load('data/drawing-data-sets/drawings-191105-6-drawings.npy')
num_timesteps = 90
input_dim = 3
given_part = 0
num_classes = 6
# 3rd dim == 1 means that this point is connected to the previous one, dim == 0 that not!

fig = plt.figure('Original training data', figsize=(30, 21.0))
plt.rcParams.update({'font.size': 55, 'legend.fontsize': 30})


names = ['FACE', 'HOUSE', 'CAR', 'FLOWER', 'HUMAN', 'ROCKET']
for pattern in range(num_classes):
    ax = fig.add_subplot(231+pattern)
    ax.set_title(names[pattern])
    ax.set_xlabel('$x_0$')
    ax.set_ylabel('$x_1$')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    for i in range(pattern, train_data.shape[0], num_classes):
        for t in range(1, num_timesteps):
            traj = train_data[i,:].reshape((num_timesteps, input_dim))
            if int(np.round(traj[t,2])) == 1:
                if t < given_part:
                    ax.plot(traj[t-1:t+1,0], traj[t-1:t+1,1], 'g', linewidth = 0.3)
                else:
                    ax.plot(traj[t-1:t+1,0], traj[t-1:t+1,1], 'k', linewidth = 0.3)
            else:
                if t < given_part:
                    ax.plot(traj[t-1:t+1,0], traj[t-1:t+1,1], '#11ff11', linewidth = 0.3)
                else:
                    ax.plot(traj[t-1:t+1,0], traj[t-1:t+1,1], 'lightgray', linewidth = 0.3)
    current_i=6+pattern
    for t in range(1, num_timesteps):
        traj = train_data[current_i,:].reshape((num_timesteps, input_dim))
        if int(np.round(traj[t,2])) == 1:
            if t < given_part:
                ax.plot(traj[t-1:t+1,0], traj[t-1:t+1,1], 'g', linewidth = 5)
            else:
                ax.plot(traj[t-1:t+1,0], traj[t-1:t+1,1], 'k', linewidth = 5)
        else:
            if t < given_part:
                ax.plot(traj[t-1:t+1,0], traj[t-1:t+1,1], '#11ff11', linewidth = 5)
            else:
                ax.plot(traj[t-1:t+1,0], traj[t-1:t+1,1], 'lightgray', linewidth = 5)

    if pattern == 1 or pattern == 2 or pattern == 4 or pattern == 5:
        plt.yticks([], [])
        plt.ylabel('')
    if pattern <= 2:
        plt.xticks([], [])
        plt.xlabel('')


plt.tight_layout()
plt.savefig('original-training-data.pdf')
plt.close()


