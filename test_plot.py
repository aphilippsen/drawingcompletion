import os
import numpy as np
import matplotlib.pyplot as plt


data_set_name = '2019-11-all-new'
setting = "pca-only-one-norm"
eval_dir = "./results/evaluation/" + data_set_name + "/"


"""
# Inner distances of neuron activations

inners0001 = np.load('./results/evaluation/' + data_set_name + '/inference-0.001/' + setting + '_inner_dist_mean.npy')
inners001 = np.load('./results/evaluation/' + data_set_name + '/inference-0.01/' + setting + '_inner_dist_mean.npy')
inners01 = np.load('./results/evaluation/' + data_set_name + '/inference-0.1/' + setting + '_inner_dist_mean.npy')
inners1 = np.load('./results/evaluation/' + data_set_name + '/inference-1/' + setting + '_inner_dist_mean.npy')
inners10 = np.load('./results/evaluation/' + data_set_name + '/inference-10/' + setting + '_inner_dist_mean.npy')
inners100 = np.load('./results/evaluation/' + data_set_name + '/inference-100/' + setting + '_inner_dist_mean.npy')
inners1000 = np.load('./results/evaluation/' + data_set_name + '/inference-1000/' + setting + '_inner_dist_mean.npy')

# This is variability between patterns which is not so interesting here
#inners100var = np.load('./results/evaluation/' + data_set_name + '/inference-100/' + setting = "pca-only-one" + '_inner_dist_var.npy')
#inners10var = np.load('./results/evaluation/' + data_set_name + '/inference-10/' + setting = "pca-only-one" + '_inner_dist_var.npy')
#inners1var = np.load('./results/evaluation/' + data_set_name + '/inference-1/' + setting = "pca-only-one" + '_inner_dist_var.npy')
#inners001var = np.load('./results/evaluation/' + data_set_name + '/inference-0.01/' + setting = "pca-only-one" + '_inner_dist_var.npy')
#inners01var = np.load('./results/evaluation/' + data_set_name + '/inference-0.1/' + setting = "pca-only-one" + '_inner_dist_var.npy')

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

plt.errorbar(np.arange(90), np.mean(inners0001,axis=0), yerr=np.sqrt(np.var(inners0001,axis=0)), color='blue')
plt.errorbar(np.arange(90), np.mean(inners001,axis=0), yerr=np.sqrt(np.var(inners001,axis=0)), color='purple')
plt.errorbar(np.arange(90), np.mean(inners01,axis=0), yerr=np.sqrt(np.var(inners01,axis=0)), color='turquoise')
plt.errorbar(np.arange(90), np.mean(inners1,axis=0), yerr=np.sqrt(np.var(inners1,axis=0)), color='green')
plt.errorbar(np.arange(90), np.mean(inners10,axis=0), yerr=np.sqrt(np.var(inners10,axis=0)), color='orange')
plt.errorbar(np.arange(90), np.mean(inners100,axis=0), yerr=np.sqrt(np.var(inners100,axis=0)), color='red')
plt.errorbar(np.arange(90), np.mean(inners1000,axis=0), yerr=np.sqrt(np.var(inners1000,axis=0)), color='pink')

plt.savefig(os.path.join(eval_dir, "inner-distance-per-H.png"))


# export separately all 0.01... etc per time scale... no better do that already in the eval-repr script


# check, are they all the same scale due to scaling??? should be at least m=0 and v=1
"""



"""
# PLOT TRAINING DATA

train_data = np.load('data/drawing-data-sets/drawings-191105-6-drawings.npy')
num_timesteps = 90
input_dim = 3
draw_only = 30
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
                if t < draw_only:
                    ax.plot(traj[t-1:t+1,0], traj[t-1:t+1,1], 'k', linewidth = 0.3)
            else:
                if t < draw_only:
                    ax.plot(traj[t-1:t+1,0], traj[t-1:t+1,1], 'lightgray', linewidth = 0.3)
    current_i=6+pattern
    for t in range(1, num_timesteps):
        traj = train_data[current_i,:].reshape((num_timesteps, input_dim))
        if int(np.round(traj[t,2])) == 1:
            if t < draw_only:
                ax.plot(traj[t-1:t+1,0], traj[t-1:t+1,1], 'k', linewidth = 5)
        else:
            if t < draw_only:
                ax.plot(traj[t-1:t+1,0], traj[t-1:t+1,1], 'lightgray', linewidth = 5)

    if pattern == 1 or pattern == 2 or pattern == 4 or pattern == 5:
        plt.yticks([], [])
        plt.ylabel('')
    if pattern <= 2:
        plt.xticks([], [])
        plt.xlabel('')


plt.tight_layout()
#plt.savefig('original-training-data.pdf')
plt.savefig('to-complete-training-data.pdf')
plt.close()

"""

"""
# plot the inferred initial states and the training initial states
colors = ['red', 'orange', 'green', 'blue', 'gray', 'black']
pattern_category = ['FACE', 'HOUSE', 'CAR', 'FLOWER', 'HUMAN', 'ROCKET']
all_runs = ["2019-11-05_14-34_0970831", "2019-11-05_14-35_0321645", "2019-11-05_14-35_0421490", "2019-11-05_14-35_0786428", "2019-11-05_14-36_0715535", "2019-11-08_15-35_0902631", "2019-11-08_15-36_0061932", "2019-11-08_15-36_0110189", "2019-11-08_15-36_0712878", "2019-11-08_15-36_0818585"]
setting = "pca-only-one-norm"
num_inferences = 10
num_patterns = 6
#which_H = ['0.001', '0.01', '1', '100', '1000']
which_H = ['0.001', '1', '1000']
patterns_to_plot = [0, 1, 3]
# from mpl_toolkits.mplot3d import Axes3D

for plot_run in all_runs:
    which_run = plot_run

    fig = plt.figure('Trained and inferred IS', figsize=(10*(len(which_H)+1), 12.0))
    plt.rcParams.update({'font.size': 40, 'legend.fontsize': 40})

    for h in which_H:

        # ax = fig.add_subplot(101 + len(which_H)*10 + which_H.index(h), projection='3d')
        ax = fig.add_subplot(101 + len(which_H)*10 + which_H.index(h))
        ax.set_title('H='+h)
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')

        trained_is = np.load(os.path.join(eval_dir, 'inference-' + h + '/' + setting + '_training-is_' + which_run + '_H-' + h + '.npy'))
        ax.scatter(trained_is[:,0], trained_is[:,1], color= colors, s=1000,marker='*')
        # ax.scatter(trained_is[:,0], trained_is[:,1], trained_is[:,2], color= colors, s=1000,marker='*')

        for inf in range(num_inferences):
            inferred_is = np.load(os.path.join(eval_dir, 'inference-' + h + '/' + setting + '_inferred-is_' + which_run + '-' + str(inf) + '_H-' + h + '.npy'))
            ax.scatter(inferred_is[:,0], inferred_is[:,1], color= colors, s=200,marker='o')
            # ax.scatter(inferred_is[:,0], inferred_is[:,1], inferred_is[:,2], color= colors, s=200,marker='o')

    #fig.legend(loc=(0.855, 0.47))
    fig.tight_layout()
    fig.show()
    # import pdb; pdb.set_trace()
    plt.savefig(os.path.join(eval_dir, 'inferred-is-plot-' + str(len(which_H)) + '-' + which_run + '.pdf'))
    plt.close()



    # plot the neuron activations
    fig = plt.figure('Neural activation trajectories', figsize=(10*(len(which_H)+1), 24.0))
    plt.rcParams.update({'font.size': 40, 'legend.fontsize': 40})

    for h in which_H:
        # above row: 0:30 timesteps
        ax = fig.add_subplot(201 + len(which_H)*10 + which_H.index(h))
        ax.set_title('H='+h)
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')

        for inf in range(num_inferences):
            for pat in patterns_to_plot:
                uh_history = np.load(os.path.join(eval_dir, 'inference-' + h + '/' + setting + '_uh-history_' + which_run + '-' + str(inf) + '_pattern-' + str(pat) + '_H-' + h + '.npy'))
                ax.scatter(uh_history[1:30,0], uh_history[1:30,1], color=colors[pat], s = 50, marker = 'o')
                ax.scatter(uh_history[0,0], uh_history[0,1], color=colors[pat], s = 2000, marker = '*')
                ax.scatter(uh_history[29,0], uh_history[29,1], color=colors[pat], s = 1000, marker = 's')
        ax.set_xlim([-1,1])
        ax.set_ylim([-1,1])
        ax.set_xticks([-1,-0.5,0,0.5,1])
        ax.set_yticks([-1,-0.5,0,0.5,1])

        # lower row: 30:90 time_steps
        ax = fig.add_subplot(201 + len(which_H)*10 + which_H.index(h) + 3)
        ax.set_title('H='+h)
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')


        for inf in range(num_inferences):
            for pat in patterns_to_plot:
                uh_history = np.load(os.path.join(eval_dir, 'inference-' + h + '/' + setting + '_uh-history_' + which_run + '-' + str(inf) + '_pattern-' + str(pat) + '_H-' + h + '.npy'))
                ax.scatter(uh_history[30:,0], uh_history[30:,1], color=colors[pat], s = 50, marker = 'o')
                ax.scatter(uh_history[30,0], uh_history[30,1], color=colors[pat], s = 1000, marker = 's')
        ax.set_xlim([-1,1])
        ax.set_ylim([-1,1])
        ax.set_xticks([-1,-0.5,0,0.5,1])
        ax.set_yticks([-1,-0.5,0,0.5,1])

    fig.tight_layout()
    plt.savefig(os.path.join(eval_dir, 'uh-history-' + str(len(which_H)) + "-" + which_run + '.pdf'))
    plt.close()



    # plot initial state emergence
    fig = plt.figure('Neural activation trajectories', figsize=(10*(len(which_H)+1), 12.0))
    plt.rcParams.update({'font.size': 40, 'legend.fontsize': 40})

    for h in which_H:
        # above row: 0:30 timesteps
        ax = fig.add_subplot(101 + len(which_H)*10 + which_H.index(h))
        ax.set_title('H='+h)
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')

        for inf in range(num_inferences):
            for pat in patterns_to_plot:
                is_history = np.load(os.path.join(eval_dir, 'inference-' + h + '/' + setting + '_is-history_' + which_run + '-' + str(inf) + '_pattern-' + str(pat) + '_H-' + h + '.npy'))
                ax.scatter(is_history[1:30,0], is_history[1:30,1], color=colors[pat], s = 50, marker = 'o')
                ax.scatter(is_history[0,0], is_history[0,1], color=colors[pat], s = 2000, marker = '*')
                ax.scatter(is_history[is_history.shape[0]-1,0], is_history[is_history.shape[0]-1,1], color=colors[pat], s = 1000, marker = 's')

    fig.tight_layout()
    plt.savefig(os.path.join(eval_dir, 'is-history-' + str(len(which_H)) + "-" + which_run + '.pdf'))
    plt.close()
"""


"""
disttotrain0001 = np.load(eval_dir + '/inference-0.001/' + setting + '_dist_to_corresponding_training_is.npy')
disttotrain001 = np.load(eval_dir + '/inference-0.01/' + setting + '_dist_to_corresponding_training_is.npy')
disttotrain01 = np.load(eval_dir + '/inference-0.1/' + setting + '_dist_to_corresponding_training_is.npy')
disttotrain1 = np.load(eval_dir + '/inference-1/' + setting + '_dist_to_corresponding_training_is.npy')
disttotrain10 = np.load(eval_dir + '/inference-10/' + setting + '_dist_to_corresponding_training_is.npy')
disttotrain100 = np.load(eval_dir + '/inference-100/' + setting + '_dist_to_corresponding_training_is.npy')
disttotrain1000 = np.load(eval_dir + '/inference-1000/' + setting + '_dist_to_corresponding_training_is.npy')

distbettrain0001 = np.load(eval_dir + '/inference-0.001/' + setting + '_dist_between_training_is.npy')
distbettrain001 = np.load(eval_dir + '/inference-0.01/' + setting + '_dist_between_training_is.npy')
distbettrain01 = np.load(eval_dir + '/inference-0.1/' + setting + '_dist_between_training_is.npy')
distbettrain1 = np.load(eval_dir + '/inference-1/' + setting + '_dist_between_training_is.npy')
distbettrain10 = np.load(eval_dir + '/inference-10/' + setting + '_dist_between_training_is.npy')
distbettrain100 = np.load(eval_dir + '/inference-100/' + setting + '_dist_between_training_is.npy')
distbettrain1000 = np.load(eval_dir + '/inference-1000/' + setting + '_dist_between_training_is.npy')

plt.figure()

for i in range(disttotrain0001.shape[0]):
    for j in range(len(disttotrain0001[0])):
        plt.plot(1, disttotrain0001[i][j], color='blue', marker='*')
        plt.plot(2, disttotrain001[i][j], color='purple', marker='*')
        plt.plot(3, disttotrain01[i][j], color='turquoise', marker='*')
        plt.plot(4, disttotrain1[i][j], color='green', marker='*')
        plt.plot(5, disttotrain10[i][j], color='orange', marker='*')
        plt.plot(6, disttotrain100[i][j], color='red', marker='*')
        plt.plot(7, disttotrain1000[i][j], color='pink', marker='*')
plt.savefig(os.path.join(eval_dir, "disttotrain.png"))

plt.figure()

for i in range(distbettrain0001.shape[0]):
    plt.plot(1, distbettrain0001[i], color='blue', marker='*')
    plt.plot(2, distbettrain001[i], color='purple', marker='*')
    plt.plot(3, distbettrain01[i], color='turquoise', marker='*')
    plt.plot(4, distbettrain1[i], color='green', marker='*')
    plt.plot(5, distbettrain10[i], color='orange', marker='*')
    plt.plot(6, distbettrain100[i], color='red', marker='*')
    plt.plot(7, distbettrain1000[i], color='pink', marker='*')
plt.savefig(os.path.join(eval_dir, "distbettrain.png"))


#plt.errorbar(np.arange(90), np.mean(outers0001,axis=0), yerr=np.mean(outers0001v,axis=0), color='blue')
#plt.errorbar(np.arange(90), np.mean(outers001,axis=0), yerr=np.mean(outers001v,axis=0), color='purple')
#plt.errorbar(np.arange(90), np.mean(outers01,axis=0), yerr=np.mean(outers01v,axis=0), color='turquoise')
#plt.errorbar(np.arange(90), np.mean(outers1,axis=0), yerr=np.mean(outers1v,axis=0), color='green')
#plt.errorbar(np.arange(90), np.mean(outers10,axis=0), yerr=np.mean(outers10v,axis=0), color='orange')
#plt.errorbar(np.arange(90), np.mean(outers100,axis=0), yerr=np.mean(outers100v,axis=0), color='red')
#plt.errorbar(np.arange(90), np.mean(outers1000,axis=0), yerr=np.mean(outers1000v,axis=0), color='pink')

"""







"""
import os
import numpy as np
import matplotlib.pyplot as plt


data_set_name = '2019-11-all'
setting = "pca-only-one-norm"
eval_dir = "./results/evaluation/" + data_set_name + "/"

inners0001 = np.load(eval_dir + '/inference-0.001/' + setting + '_inner_dist_mean.npy')
inners001 = np.load(eval_dir + '/inference-0.01/' + setting + '_inner_dist_mean.npy')
inners01 = np.load(eval_dir + '/inference-0.1/' + setting + '_inner_dist_mean.npy')
inners1 = np.load(eval_dir + '/inference-1/' + setting + '_inner_dist_mean.npy')
inners10 = np.load(eval_dir + '/inference-10/' + setting + '_inner_dist_mean.npy')
inners100 = np.load(eval_dir + '/inference-100/' + setting + '_inner_dist_mean.npy')
inners1000 = np.load(eval_dir + '/inference-1000/' + setting + '_inner_dist_mean.npy')

inners0001v = np.load(eval_dir + '/inference-0.001/' + setting + '_inner_dist_var.npy')
inners001v = np.load(eval_dir + '/inference-0.01/' + setting + '_inner_dist_var.npy')
inners01v = np.load(eval_dir + '/inference-0.1/' + setting + '_inner_dist_var.npy')
inners1v = np.load(eval_dir + '/inference-1/' + setting + '_inner_dist_var.npy')
inners10v = np.load(eval_dir + '/inference-10/' + setting + '_inner_dist_var.npy')
inners100v = np.load(eval_dir + '/inference-100/' + setting + '_inner_dist_var.npy')
inners1000v = np.load(eval_dir + '/inference-1000/' + setting + '_inner_dist_var.npy')


outers0001 = np.load(eval_dir + '/inference-0.001/' + setting + '_outer_dist_mean.npy')
outers001 = np.load(eval_dir + '/inference-0.01/' + setting + '_outer_dist_mean.npy')
outers01 = np.load(eval_dir + '/inference-0.1/' + setting + '_outer_dist_mean.npy')
outers1 = np.load(eval_dir + '/inference-1/' + setting + '_outer_dist_mean.npy')
outers10 = np.load(eval_dir + '/inference-10/' + setting + '_outer_dist_mean.npy')
outers100 = np.load(eval_dir + '/inference-100/' + setting + '_outer_dist_mean.npy')
outers1000 = np.load(eval_dir + '/inference-1000/' + setting + '_outer_dist_mean.npy')

outers0001v = np.load(eval_dir + '/inference-0.001/' + setting + '_outer_dist_var.npy')
outers001v = np.load(eval_dir + '/inference-0.01/' + setting + '_outer_dist_var.npy')
outers01v = np.load(eval_dir + '/inference-0.1/' + setting + '_outer_dist_var.npy')
outers1v = np.load(eval_dir + '/inference-1/' + setting + '_outer_dist_var.npy')
outers10v = np.load(eval_dir + '/inference-10/' + setting + '_outer_dist_var.npy')
outers100v = np.load(eval_dir + '/inference-100/' + setting + '_outer_dist_var.npy')
outers1000v = np.load(eval_dir + '/inference-1000/' + setting + '_outer_dist_var.npy')

# This is variability between patterns which is not so interesting here
#outers100var = np.load('./results/evaluation/' + data_set_name + '/inference-100/' + setting = "pca-only-one" + '_outer_dist_var.npy')
#outers10var = np.load('./results/evaluation/' + data_set_name + '/inference-10/' + setting = "pca-only-one" + '_outer_dist_var.npy')
#outers1var = np.load('./results/evaluation/' + data_set_name + '/inference-1/' + setting = "pca-only-one" + '_outer_dist_var.npy')
#outers001var = np.load('./results/evaluation/' + data_set_name + '/inference-0.01/' + setting = "pca-only-one" + '_outer_dist_var.npy')
#outers01var = np.load('./results/evaluation/' + data_set_name + '/inference-0.1/' + setting = "pca-only-one" + '_outer_dist_var.npy')

colors = ['blue', 'purple', 'turquoise', 'green', 'red', 'orange', 'pink']

plt.figure()

for i in range(outers001.shape[0]):
    plt.plot(np.arange(90), outers0001[i,:], color='blue')
    plt.plot(np.arange(90), outers001[i,:], color='purple')
    plt.plot(np.arange(90), outers01[i,:], color='turquoise')
    plt.plot(np.arange(90), outers1[i,:], color='green')
    plt.plot(np.arange(90), outers10[i,:], color='orange')
    plt.plot(np.arange(90), outers100[i,:], color='red')
    plt.plot(np.arange(90), outers1000[i,:], color='pink')

#plt.errorbar(np.arange(90), np.mean(outers0001,axis=0), yerr=np.mean(outers0001v,axis=0), color='blue')
#plt.errorbar(np.arange(90), np.mean(outers001,axis=0), yerr=np.mean(outers001v,axis=0), color='purple')
#plt.errorbar(np.arange(90), np.mean(outers01,axis=0), yerr=np.mean(outers01v,axis=0), color='turquoise')
#plt.errorbar(np.arange(90), np.mean(outers1,axis=0), yerr=np.mean(outers1v,axis=0), color='green')
#plt.errorbar(np.arange(90), np.mean(outers10,axis=0), yerr=np.mean(outers10v,axis=0), color='orange')
#plt.errorbar(np.arange(90), np.mean(outers100,axis=0), yerr=np.mean(outers100v,axis=0), color='red')
#plt.errorbar(np.arange(90), np.mean(outers1000,axis=0), yerr=np.mean(outers1000v,axis=0), color='pink')

plt.savefig(os.path.join(eval_dir, "outer-distance-per-H.png"))


# export separately all 0.01... etc per time scale... no better do that already in the eval-repr script


# check, are they all the same scale due to scaling??? should be at least m=0 and v=1



disttotrain0001 = np.load(eval_dir + '/inference-0.001/' + setting + '_dist_to_corresponding_training_is.npy')
disttotrain001 = np.load(eval_dir + '/inference-0.01/' + setting + '_dist_to_corresponding_training_is.npy')
disttotrain01 = np.load(eval_dir + '/inference-0.1/' + setting + '_dist_to_corresponding_training_is.npy')
disttotrain1 = np.load(eval_dir + '/inference-1/' + setting + '_dist_to_corresponding_training_is.npy')
disttotrain10 = np.load(eval_dir + '/inference-10/' + setting + '_dist_to_corresponding_training_is.npy')
disttotrain100 = np.load(eval_dir + '/inference-100/' + setting + '_dist_to_corresponding_training_is.npy')
disttotrain1000 = np.load(eval_dir + '/inference-1000/' + setting + '_dist_to_corresponding_training_is.npy')

distbettrain0001 = np.load(eval_dir + '/inference-0.001/' + setting + '_dist_between_training_is.npy')
distbettrain001 = np.load(eval_dir + '/inference-0.01/' + setting + '_dist_between_training_is.npy')
distbettrain01 = np.load(eval_dir + '/inference-0.1/' + setting + '_dist_between_training_is.npy')
distbettrain1 = np.load(eval_dir + '/inference-1/' + setting + '_dist_between_training_is.npy')
distbettrain10 = np.load(eval_dir + '/inference-10/' + setting + '_dist_between_training_is.npy')
distbettrain100 = np.load(eval_dir + '/inference-100/' + setting + '_dist_between_training_is.npy')
distbettrain1000 = np.load(eval_dir + '/inference-1000/' + setting + '_dist_between_training_is.npy')

plt.figure()

for i in range(disttotrain0001.shape[0]):
    for j in range(len(disttotrain0001[0])):
        plt.plot(1, disttotrain0001[i][j]/distbettrain0001[i], color='blue', marker='*')
        plt.plot(2, disttotrain001[i][j]/distbettrain001[i], color='purple', marker='*')
        plt.plot(3, disttotrain01[i][j]/distbettrain01[i], color='turquoise', marker='*')
        plt.plot(4, disttotrain1[i][j]/distbettrain1[i], color='green', marker='*')
        plt.plot(5, disttotrain10[i][j]/distbettrain10[i], color='orange', marker='*')
        plt.plot(6, disttotrain100[i][j]/distbettrain100[i], color='red', marker='*')
        plt.plot(7, disttotrain1000[i][j]/distbettrain1000[i], color='pink', marker='*')

#plt.errorbar(np.arange(90), np.mean(outers0001,axis=0), yerr=np.mean(outers0001v,axis=0), color='blue')
#plt.errorbar(np.arange(90), np.mean(outers001,axis=0), yerr=np.mean(outers001v,axis=0), color='purple')
#plt.errorbar(np.arange(90), np.mean(outers01,axis=0), yerr=np.mean(outers01v,axis=0), color='turquoise')
#plt.errorbar(np.arange(90), np.mean(outers1,axis=0), yerr=np.mean(outers1v,axis=0), color='green')
#plt.errorbar(np.arange(90), np.mean(outers10,axis=0), yerr=np.mean(outers10v,axis=0), color='orange')
#plt.errorbar(np.arange(90), np.mean(outers100,axis=0), yerr=np.mean(outers100v,axis=0), color='red')
#plt.errorbar(np.arange(90), np.mean(outers1000,axis=0), yerr=np.mean(outers1000v,axis=0), color='pink')

plt.savefig(os.path.join(eval_dir, "disttotrain.png"))


# PLOT TRAINING DATA

train_data = np.load('data/drawing-data-sets/drawings-191105-6-drawings.npy')
num_timesteps = 90
input_dim = 3
draw_only = 30
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
                if t < draw_only:
                    ax.plot(traj[t-1:t+1,0], traj[t-1:t+1,1], 'k', linewidth = 0.3)
            else:
                if t < draw_only:
                    ax.plot(traj[t-1:t+1,0], traj[t-1:t+1,1], 'lightgray', linewidth = 0.3)
    current_i=6+pattern
    for t in range(1, num_timesteps):
        traj = train_data[current_i,:].reshape((num_timesteps, input_dim))
        if int(np.round(traj[t,2])) == 1:
            if t < draw_only:
                ax.plot(traj[t-1:t+1,0], traj[t-1:t+1,1], 'k', linewidth = 5)
        else:
            if t < draw_only:
                ax.plot(traj[t-1:t+1,0], traj[t-1:t+1,1], 'lightgray', linewidth = 5)

    if pattern == 1 or pattern == 2 or pattern == 4 or pattern == 5:
        plt.yticks([], [])
        plt.ylabel('')
    if pattern <= 2:
        plt.xticks([], [])
        plt.xlabel('')


plt.tight_layout()
#plt.savefig('original-training-data.pdf')
plt.savefig('to-complete-training-data.pdf')
plt.close()

"""


