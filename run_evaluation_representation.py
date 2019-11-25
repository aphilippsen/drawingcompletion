# do it first for the "best" completion: then I can be sure it is working as expected!

import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pathlib

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from nets import load_network

# Plotting the internal representations of the trained networks

training_data_file = "data/drawing-data-sets/drawings-191105-6-drawings.npy"
data_set_name = '2019-11-all'
# data_set_name = '2019-11-08'
mode = 'inference'
training_hyp = '100'

num_timesteps = 90
num_neurons = 250


# this code is only for one reduced_time_steps at a time! (index reduced=0)
reduced = 0

# test across all testing conditions? this determines which inference data are used for learning the PCA mapping
test_hyp_all = ['0.001', '0.01', '0.1', '1', '10', '100', '1000']
# or only for the corresponding one
#test_hyp_all = etest_hyp_all.index(training_hyp)

# which to include in the plots (per indices of test_hyp_all)
#plotting_test_hyps = [0, 1, 2, 3, 4]
plotting_test_hyps = [test_hyp_all.index(training_hyp)]

eval_head_dir = './results/completion/' + data_set_name

result_dir = "./results/evaluation/" + data_set_name + "/" + mode + '-' + training_hyp
pathlib.Path(result_dir).mkdir(parents=True, exist_ok=True)

num_runs = 10 # how often the experiment was independently conducted
num_inferences = 10 # how many test inferences have been performed in each run
num_patterns = 6 # number of different training sample patterns

# to iterate across these:
# for each of the trained networks (=runs) separately

mean_inner_dist_per_timestep = np.zeros((num_runs, num_timesteps))
var_inner_dist_per_timestep = np.zeros((num_runs, num_timesteps))

dir_list = next(os.walk(eval_head_dir))[1]
for curr_r in range(len(dir_list)):
    # do analysis for a determined training hyp_prior and one of the num_runs trained networks
    current_run = os.path.join(eval_head_dir, dir_list[curr_r])
    print("Evaluate for " + str(current_run))

    # get training initial state (which is the same regardless of testing hyp_prior)
    corresponding_train_dir = os.path.join('./results/training/' + data_set_name, dir_list[curr_r])
    params, train_model = load_network(corresponding_train_dir + "/" + training_hyp, model_filename='network-epoch-best.npz')

    trained_is = train_model.initial_states.W.array

    uh_history = np.empty((len(test_hyp_all), 1), dtype=object)
    inferred_is = np.empty((len(test_hyp_all), 1), dtype=object)

    test_hyp_idx = -1
    for test_hyp in test_hyp_all:
        test_hyp_idx += 1
        eval_dir = current_run+'/'+training_hyp+'/'+mode+'/test-'+test_hyp
        print("Load data from " + eval_dir)

        # load generation neuron activation
        uh_history[test_hyp_idx,0] = np.empty((num_inferences, num_patterns), dtype=object)
        
        # load the current history
        uh = np.load(eval_dir + "/final-uh-history-"+test_hyp+"_mode-"+mode+".npy")
        # that's how the loaded uh should look like:
        #   uh[cl,red][inf].reshape((num_timesteps, num_neurons))
        # transfer info and do the reshape:
        for i in range(uh.shape[0]): # for all pattern classes
            for j in range(len(uh[i,0])): # for all inferences for this pattern
                uh_history[test_hyp_idx,0][j,i] = uh[i,0][j].reshape((num_timesteps, num_neurons))

        inferred_is[test_hyp_idx,0] = np.empty((num_inferences, 1), dtype=object)


        # get the trained initial states and the inferred initial states from all the ten inferences
        inf_nets_path = os.path.join(eval_dir, "inference_networks")
        inf_nets_list = next(os.walk(inf_nets_path))[1]
        for curr_inf in range(len(inf_nets_list)):
            # load each and extract and store the initial states
            params, model = load_network(os.path.join(inf_nets_path, inf_nets_list[curr_inf]), model_filename='network-final')
            inferred_is[test_hyp_idx,0][curr_inf,0] = model.initial_states.W.array

    # reshape everything to (-1, 250) for preparing for PCA
    inferred_is_temp = np.empty((len(test_hyp_all),1),dtype=object)
    for i in range(len(test_hyp_all)):
        inferred_is_temp[i,0] = np.concatenate(inferred_is[i,0][:,0],axis=0)
    all_inferred_is = np.concatenate(inferred_is_temp[:,0],axis=0)

    uh_history_temp = np.empty((len(test_hyp_all),1),dtype=object)
    for i in range(len(test_hyp_all)):
        uh_history_temp[i,0] = np.concatenate(uh_history[i,0].reshape((-1,1))[:,0],axis=0)
    all_uh_history = np.concatenate(uh_history_temp[:,0],axis=0)

    # all neuron activations collected into one large (-1, 250) array
    all_neuron_activations = np.concatenate((trained_is, all_inferred_is, all_uh_history))
    all_initial_states = np.concatenate((trained_is, all_inferred_is))

    # Can do PCA either on the whole trajectory of activations or only on the initial states
    data_for_pca_transform = all_neuron_activations
    # data_for_pca_transform = all_initial_states

    # scaling data to achieve mean=0 and var=1
    scaler = StandardScaler().fit(data_for_pca_transform)
    data_for_pca_transform_scaled = scaler.transform(data_for_pca_transform)

    # create PCA mapping
    pca = PCA(n_components=250)
    pca.fit(data_for_pca_transform_scaled)

    # apply PCA mapping to map initial states ...
    pca_trained_is = pca.transform(scaler.transform(trained_is))

    pca_inferred_is = np.empty((len(test_hyp_all), 1), dtype=object)
    for i in range(len(test_hyp_all)): # per test_hyp
        pca_inferred_is[i,0] = np.empty((num_inferences, 1), dtype=object)
        for j in range(num_inferences): # per inference
            pca_inferred_is[i,0][j,0] = pca.transform(scaler.transform(inferred_is[i,0][j,0]))

    # ... and neuron activations
    pca_uh_history = np.empty((len(test_hyp_all), 1), dtype=object)
    for i in range(len(test_hyp_all)): # per test_hyp
        pca_uh_history[i,0] = np.empty((num_inferences, num_patterns), dtype=object)
        for j in range(num_inferences): # per inference
            for k in range(num_patterns):
                pca_uh_history[i,0][j,k] = pca.transform(scaler.transform(uh_history[i,0][j,k]))


    # PLOTTING
    
    colors = ['red', 'orange', 'green', 'blue', 'gray', 'black']
    pattern_category = ['FACE', 'HOUSE', 'CAR', 'FLOWER', 'HUMAN', 'ROCKET']

    """
    # plot the trained and inferred_is
    # 3d
    fig = plt.figure('Trained and inferred initial states (3d)', figsize=(30,30)) #figsize=(10, 11.0))
    plt.rcParams.update({'font.size': 50, 'legend.fontsize': 30})
    ax = fig.add_subplot(111, projection='3d')

    for th in plotting_test_hyps:
        for i in range(num_patterns):
            for j in range(num_inferences):
                ax.scatter(pca_inferred_is[th,0][j][0][i,0], pca_inferred_is[th,0][j][0][i,1], pca_inferred_is[th,0][j][0][i,2], color=colors[i], marker='o',s=100)

    for i in range(num_patterns):
        ax.scatter(pca_trained_is[i,0], pca_trained_is[i,1], pca_trained_is[i,2], color=colors[i], marker='*',s=800, label=pattern_category[i])

    plt.legend()
    plt.savefig(os.path.join(result_dir, 'inferred-is-3d_run-' + dir_list[curr_r] + '.pdf'))
    plt.close()
    """

    # 2d
    fig = plt.figure('Trained and inferred initial states (2d)', figsize=(30,30)) #figsize=(10, 11.0))
    plt.rcParams.update({'font.size': 50, 'legend.fontsize': 30})
    ax = fig.add_subplot(111)

    for th in plotting_test_hyps:
        for i in range(num_patterns):
            for j in range(num_inferences):
                ax.scatter(pca_inferred_is[th,0][j][0][i,0], pca_inferred_is[th,0][j][0][i,1], color=colors[i], marker='o',s=200)    

    for i in range(num_patterns):
        ax.scatter(pca_trained_is[i,0], pca_trained_is[i,1], color=colors[i], marker='*',s=1300, label=pattern_category[i])

    plt.legend()
    plt.savefig(os.path.join(result_dir, 'inferred-is-2d_run-' + dir_list[curr_r] + '.pdf'))
    plt.close()

    # TODO plot after 30 timesteps where the activations are (did they converge according to class?)
    
    
    # From 0 to 30
    # 3d
    """
    fig = plt.figure('Neuron activations during first 30 timesteps (2d)', figsize=(30,30)) #figsize=(10, 11.0))
    plt.rcParams.update({'font.size': 50, 'legend.fontsize': 30})
    ax = fig.add_subplot(111, projection='3d')

    for th in plotting_test_hyps:
        for i in range(num_patterns):
            for j in range(num_inferences):
                for t in range(30):
                    if t==29:
                        ax.scatter(pca_uh_history[th,0][j,i][t,0], pca_uh_history[th,0][j,i][t,1], pca_uh_history[th,0][j,i][t,2], color=colors[i], marker='s',s=1000)
                    else:
                        ax.scatter(pca_uh_history[th,0][j,i][t,0], pca_uh_history[th,0][j,i][t,1], pca_uh_history[th,0][j,i][t,2], color=colors[i], marker='o',s=200)

    for ii in np.arange(10,360,45):
        for jj in np.arange(10,360,45):
            ax.view_init(elev=ii, azim=jj)
            plt.savefig(os.path.join(result_dir, 'uh-3d_run-'+ dir_list[curr_r] + "_view-" + str(ii) + "-" + str(jj) + '.png'))

    #plt.legend()
    #plt.savefig(os.path.join(result_dir, 'uh-3d_run-' + dir_list[curr_r] + '.pdf'))
    plt.close()
    """

    
    # 2d
    fig = plt.figure('Neuron activations after 30 timesteps (2d)', figsize=(30,30)) #figsize=(10, 11.0))
    plt.rcParams.update({'font.size': 50, 'legend.fontsize': 30})
    ax = fig.add_subplot(111)

    for th in plotting_test_hyps:
        for i in range(num_patterns):
            for j in range(num_inferences):
                for t in range(30):
                    if t==1:
                        ax.scatter(pca_uh_history[th,0][j,i][t,0], pca_uh_history[th,0][j,i][t,1], color=colors[i], marker='*',s=1500)
                    elif t==29:
                        ax.scatter(pca_uh_history[th,0][j,i][t,0], pca_uh_history[th,0][j,i][t,1], color=colors[i], marker='s',s=1000)
                    else:
                        ax.scatter(pca_uh_history[th,0][j,i][t,0], pca_uh_history[th,0][j,i][t,1], color=colors[i], marker='o',s=200)

    #plt.legend()
    plt.savefig(os.path.join(result_dir, 'uh-2d_run-' + dir_list[curr_r] + '.png'))
    plt.close()


    
    # From 30 to end
    # 3d

    """
    fig = plt.figure('Neuron activations during first 30 timesteps (2d)', figsize=(30,30)) #figsize=(10, 11.0))
    plt.rcParams.update({'font.size': 50, 'legend.fontsize': 30})
    ax = fig.add_subplot(111, projection='3d')

    for th in plotting_test_hyps:
        for i in range(num_patterns):
            for j in range(num_inferences):
                for t in np.arange(30,90):
                    ax.scatter(pca_uh_history[th,0][j,i][t,0], pca_uh_history[th,0][j,i][t,1], pca_uh_history[th,0][j,i][t,2], color=colors[i], marker='o',s=200)    

    for ii in np.arange(10,360,45):
        for jj in np.arange(10,360,45):
            ax.view_init(elev=ii, azim=jj)
            plt.savefig(os.path.join(result_dir, 'uh-30-end_3d_run-'+ dir_list[curr_r] + "_view-" + str(ii) + "-" + str(jj) + '.png'))

    #plt.legend()
    #plt.savefig(os.path.join(result_dir, 'uh-3d_run-' + dir_list[curr_r] + '.pdf'))
    plt.close()
    """

    
    # 2d
    fig = plt.figure('Neuron activations after 30 timesteps (2d)', figsize=(30,30)) #figsize=(10, 11.0))
    plt.rcParams.update({'font.size': 50, 'legend.fontsize': 30})
    ax = fig.add_subplot(111)

    for th in plotting_test_hyps:
        for i in range(num_patterns):
            for j in range(num_inferences):
                for t in np.arange(30,90):
                    if t==30:
                        ax.scatter(pca_uh_history[th,0][j,i][t,0], pca_uh_history[th,0][j,i][t,1], color=colors[i], marker='s',s=1000)
                    else:
                        ax.scatter(pca_uh_history[th,0][j,i][t,0], pca_uh_history[th,0][j,i][t,1], color=colors[i], marker='o',s=200)

    #plt.legend()
    plt.savefig(os.path.join(result_dir, 'uh-30-to-end_2d_run-' + dir_list[curr_r] + '.png'))
    plt.close()
    




    # Analysis
    from scipy.spatial.distance import pdist, squareform

    # compare the clustering of the inferred initial states:
    # in the original space of neuron activations

    # collect all initial states for one pattern
    all_is_per_cluster = np.empty((num_patterns,1), dtype=object)
    inner_distances_is_clusters = []
    for pat in range(num_patterns):
        all_is_per_cluster[pat,0] = np.zeros((len(plotting_test_hyps) * num_inferences, num_neurons))
        for i in range(len(plotting_test_hyps)): #range(len(test_hyp_all)):
            for j in range(num_inferences):
                all_is_per_cluster[pat,0][i*num_inferences+j,:] = pca_inferred_is[i,0][j,0][pat,:]
        
        inner_distances_is_clusters.append(np.mean(pdist(all_is_per_cluster[pat,0])))

    



    # per time step, how close are the trajectories together?
#    all_uh_per_timestep_per_cluster = np.empty((num_timesteps, num_patterns),dtype=object)


    for t in range(num_timesteps):

        inner_distances = []
        inner_dist_var = []
        for pat in range(num_patterns):        
            current_matrix = np.zeros((len(plotting_test_hyps) * num_inferences, num_neurons))

            for i in range(len(plotting_test_hyps)): #range(len(test_hyp_all)):
                for j in range(num_inferences):
                    current_matrix[i*num_inferences+j,:] = pca_uh_history[i,0][j,pat][t,:]

            inner_distances.append(np.mean(pdist(current_matrix)))
            inner_dist_var.append(np.var(pdist(current_matrix)))

        # take the mean of measured distances for this timestep for each timestep
        mean_inner_dist_per_timestep[curr_r,t] = np.mean(inner_distances)
        # take the mean over variance of patterns to get the variance
        var_inner_dist_per_timestep[curr_r,t] = np.mean(inner_dist_var)

np.save(os.path.join(result_dir, 'inner_dist_mean.npy'), mean_inner_dist_per_timestep)
np.save(os.path.join(result_dir, 'inner_dist_var.npy'), var_inner_dist_per_timestep)
np.save(os.path.join(result_dir, 'inner_dist_is_clusters.npy'), inner_distances_is_clusters)


# mean distance score of within-class-differences of neuron activations per time step, averaged across all classes, for each network separately, and mean and variance among networks
with open(os.path.join(result_dir, "neuron-act-inner-distances-per-time_H-" + str(training_hyp) + ".txt"), 'w') as f:
    f.write("t\t")
    for net in range(num_runs):
        f.write("net" + str(net) + "\t")
    f.write("mean\tstd\n")
    for t in range(num_timesteps):
        f.write(str(t) + "\t")
        for net in range(num_runs):
            f.write(str(mean_inner_dist_per_timestep[net,t]) + "\t")
        f.write(str(np.mean(mean_inner_dist_per_timestep[:,t])) + "\t" + str(np.sqrt(np.var(mean_inner_dist_per_timestep[:,t]))) + "\n")






# PCA, and plot them

# measure the distance of the inferred initial states to the correct trained initial states

# => as the inference part probably is not related to the problem, we do not see a difference there


# then project all the 10 generations of one network via PCA

# with a weak prior, there should be more variation according to the input data

# with a strong prior, there should be less variation according to the input data



