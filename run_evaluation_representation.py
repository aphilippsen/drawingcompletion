# do it first for the "best" completion: then I can be sure it is working as expected!

import numpy as np
import scipy
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
training_hyp = '0.001'

num_timesteps = 90
num_neurons = 250

graphics_extension = ".png"

# this code is only for one reduced_time_steps at a time! (reduced=0 is the array index that should be used)
reduced = 0

# test across all testing conditions? this determines which inference data are used for learning the PCA mapping
#test_hyp_all = ['0.001', '0.01', '0.1', '1', '10', '100', '1000']
#eval_setting = "pca-all"
# or only for the corresponding one
eval_setting = "pca-only-one"
test_hyp_all = [training_hyp]

normalize_for_statistics = True
if normalize_for_statistics:
    eval_setting += "-norm"

# which of these inferred initial states should be included in the plots? (define per indices of test_hyp_all)
#plotting_test_hyps = [0, 1, 2, 3, 4, 5, 6]
plotting_test_hyps = [test_hyp_all.index(training_hyp)]

eval_head_dir = './results/completion/' + data_set_name

result_dir = "./results/evaluation/" + data_set_name + "/" + mode + '-' + training_hyp
pathlib.Path(result_dir).mkdir(parents=True, exist_ok=True)

num_runs = 10 # how often the experiment was independently conducted
num_inferences = 10 # how many inferences have been performed in each run to test the network
num_patterns = 6 # number of different training sample patterns

# data structures for collecting the distances between different training samples in the neural activation space in each time step
mean_inner_dist_per_timestep = np.zeros((num_runs, num_timesteps))
var_inner_dist_per_timestep = np.zeros((num_runs, num_timesteps))

# Analysis is performed separately for each of the trained networks (=runs) with the above defined H_train
dir_list = next(os.walk(eval_head_dir))[1]
for curr_r in range(len(dir_list)):
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
#    if len(data_for_pca_transform)

    # scaling data to achieve mean=0 and var=1
    scaler = StandardScaler().fit(data_for_pca_transform)
    data_for_pca_transform_scaled = scaler.transform(data_for_pca_transform)

    # create PCA mapping
    pca = PCA(n_components=250)
    pca.fit(data_for_pca_transform_scaled)
    all_pca_data = pca.transform(data_for_pca_transform_scaled)

    # compute factors for normalizing everything to [-1, 1]
    if normalize_for_statistics:
        from utils.normalize import normalize, range2norm, norm2range
        all_pca_data_normalized, norm_offset, norm_range, minmax = normalize(all_pca_data)

    # apply PCA mapping to map initial states ...
    pca_trained_is = pca.transform(scaler.transform(trained_is))
    if normalize_for_statistics:
        pca_trained_is = range2norm(pca_trained_is, norm_offset, norm_range, minmax)

    pca_inferred_is = np.empty((len(test_hyp_all), 1), dtype=object)
    for i in range(len(test_hyp_all)): # per test_hyp
        pca_inferred_is[i,0] = np.empty((num_inferences, 1), dtype=object)
        for j in range(num_inferences): # per inference
            pca_inferred_is[i,0][j,0] = pca.transform(scaler.transform(inferred_is[i,0][j,0]))
            if normalize_for_statistics:
                pca_inferred_is[i,0][j,0] = range2norm(pca_inferred_is[i,0][j,0], norm_offset, norm_range, minmax)

    # ... and neuron activations
    pca_uh_history = np.empty((len(test_hyp_all), 1), dtype=object)
    for i in range(len(test_hyp_all)): # per test_hyp
        pca_uh_history[i,0] = np.empty((num_inferences, num_patterns), dtype=object)
        for j in range(num_inferences): # per inference
            for k in range(num_patterns):
                pca_uh_history[i,0][j,k] = pca.transform(scaler.transform(uh_history[i,0][j,k]))
                if normalize_for_statistics:
                    pca_uh_history[i,0][j,k] = range2norm(pca_uh_history[i,0][j,k], norm_offset, norm_range, minmax)

    # store all the results as text files
    with open(os.path.join(result_dir, eval_setting + "_training-is_" + dir_list[curr_r] + "_H-" + str(training_hyp) + ".txt"), "w") as f:
        f.write('class')
        for i in range(250):
            f.write('\tdim' + str(i))
        f.write('\n')
        for pa in range(num_patterns):
            f.write(str(pa))
            for i in range(250):
                f.write('\t' + str(pca_trained_is[pa,i]))
            f.write('\n')
    np.save(os.path.join(result_dir, eval_setting + "_training-is_" + dir_list[curr_r] + "_H-" + str(training_hyp) + ".npy"), pca_trained_is)

    for h in range(len(test_hyp_all)): # per test_hyp
        for inf in range(num_inferences): # per inference
            with open(os.path.join(result_dir, eval_setting + "_inferred-is_" + dir_list[curr_r] + '-' + str(inf) + '_H-' + test_hyp_all[h] + ".txt"), "w") as f:
                f.write('class')
                for i in range(250):
                    f.write('\tdim' + str(i))
                f.write('\n')
                for pa in range(num_patterns):
                    f.write(str(pa))
                    for i in range(250):
                        f.write('\t' + str(pca_inferred_is[h,0][inf,0][pa,i]))
                    f.write('\n')
            np.save(os.path.join(result_dir, eval_setting + "_inferred-is_" + dir_list[curr_r] + '-' + str(inf) + '_H-' + test_hyp_all[h] + ".npy"), pca_inferred_is[h,0][inf,0])

    for h in range(len(test_hyp_all)): # per test_hyp
        for inf in range(num_inferences): # per inference
            for pa in range(num_patterns): # per pattern class
                with open(os.path.join(result_dir, eval_setting + "_uh-history_" + dir_list[curr_r] + '-' + str(inf) + "_pattern-" + str(pa) + '_H-' + test_hyp_all[h] + ".txt"), "w") as f:
                    f.write('t')
                    for i in range(250):
                        f.write('\tdim' + str(i))
                    f.write('\n')
                    for t in range(num_timesteps):
                        f.write(str(pa))
                        for i in range(250):
                            f.write('\t' + str(pca_uh_history[h,0][inf,pa][t,i]))
                        f.write('\n')
                np.save(os.path.join(result_dir, eval_setting + "_uh-history_" + dir_list[curr_r] + '-' + str(inf) + "_pattern-" + str(pa) + '_H-' + test_hyp_all[h] + ".npy"), pca_uh_history[h,0][inf,pa])


    # STATISTICS

    pca_inferred_is_temp = np.empty((len(test_hyp_all),1),dtype=object)
    for i in range(len(test_hyp_all)):
        pca_inferred_is_temp[i,0] = np.concatenate(pca_inferred_is[i,0][:,0],axis=0)
    all_pca_inferred_is = np.concatenate(pca_inferred_is_temp[:,0],axis=0)

    for hyp in plotting_test_hyps:
        pairwise_distances_inferred_is = np.zeros((num_patterns, int(np.round(num_inferences * num_inferences - num_inferences) / 2)))

        inferred_is_this_hyp = all_pca_inferred_is[hyp*num_inferences*num_patterns:(hyp+1)*num_inferences*num_patterns,:]

        # collect all for one pattern to draw
        for pa in range(num_patterns):
            inferred_is_this_pattern = inferred_is_this_hyp[pa:num_inferences*num_patterns:num_patterns]

            # get pairwise distance
            # returns an array with all pairwise distances
            # of size: (num_inference * num_inference - num_inferences) / 2
            pairwise_distances_inferred_is[pa,:] = scipy.spatial.distance.pdist(inferred_is_this_pattern)

            # and get distance to the training initial state!
            dist_to_training_pattern = np.sqrt(np.sum((inferred_is_this_pattern - np.tile(pca_trained_is[pa,:],(10,1)))**2,axis=1))

        if curr_r == 0:
            with open(os.path.join(result_dir, eval_setting + "_pairwise-dist-inferred-is_H-" + str(training_hyp) + ".txt"), "w") as f:
                f.write('htrain')
                for pa in range(num_patterns):
                    f.write('\tpa' + str(pa))
                f.write('\n')

                for i in range(pairwise_distances_inferred_is.shape[1]):
                    f.write(str(training_hyp))
                    for pa in range(num_patterns):
                        f.write('\t' + str(pairwise_distances_inferred_is[pa, i]))
                    f.write('\n')

            with open(os.path.join(result_dir, eval_setting + "_dist-to-training-is_H-" + str(training_hyp) + ".txt"), "w") as f:
                f.write('htrain\tdist\n')
                for i in range(len(dist_to_training_pattern)):
                    f.write(str(training_hyp) + "\t" + str(dist_to_training_pattern[i]) + "\n")

        else:
            with open(os.path.join(result_dir, eval_setting + "_pairwise-dist-inferred-is_H-" + str(training_hyp) + ".txt"), "a") as f:
                for i in range(pairwise_distances_inferred_is.shape[1]):
                    f.write(str(training_hyp))
                    for pa in range(num_patterns):
                        f.write('\t' + str(pairwise_distances_inferred_is[pa, i]))
                    f.write('\n')
            with open(os.path.join(result_dir, eval_setting + "_dist-to-training-is_H-" + str(training_hyp) + ".txt"), "a") as f:
                for i in range(len(dist_to_training_pattern)):
                    f.write(str(training_hyp) + "\t" + str(dist_to_training_pattern[i]) + "\n")

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
    plt.savefig(os.path.join(result_dir, eval_setting + '_inferred-is-3d_run-' + dir_list[curr_r] + graphics_extension))
    plt.close()
    """

    # 2d
    fig = plt.figure('Trained and inferred initial states (2d)', figsize=(30,30)) #figsize=(10, 11.0))
    plt.rcParams.update({'font.size': 50, 'legend.fontsize': 30})
    ax = fig.add_subplot(111)

    for th in plotting_test_hyps:
        for i in range(num_patterns):
            for j in range(num_inferences):
                ax.scatter(pca_inferred_is[th,0][j][0][i,0], pca_inferred_is[th,0][j][0][i,1], color=colors[i], marker='o',s=2000)    

    for i in range(num_patterns):
        ax.scatter(pca_trained_is[i,0], pca_trained_is[i,1], color=colors[i], marker='*',s=5000, label=pattern_category[i])

    plt.legend()
    plt.savefig(os.path.join(result_dir, eval_setting + '_inferred-is-2d_run-' + dir_list[curr_r] + graphics_extension))
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
            plt.savefig(os.path.join(result_dir, eval_setting + '_uh-3d_run-'+ dir_list[curr_r] + "_view-" + str(ii) + "-" + str(jj) + graphics_extension))

    #plt.legend()
    #plt.savefig(os.path.join(result_dir, eval_setting + '_uh-3d_run-' + dir_list[curr_r] + graphics_extension))
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
    plt.savefig(os.path.join(result_dir, eval_setting + '_uh-2d_run-' + dir_list[curr_r] + graphics_extension))
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
            plt.savefig(os.path.join(result_dir, eval_setting + '_uh-30-end_3d_run-'+ dir_list[curr_r] + "_view-" + str(ii) + "-" + str(jj) + graphics_extension))

    #plt.legend()
    #plt.savefig(os.path.join(result_dir, eval_setting + '_uh-3d_run-' + dir_list[curr_r] + graphics_extension))
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
    plt.savefig(os.path.join(result_dir, eval_setting + '_uh-30-to-end_2d_run-' + dir_list[curr_r] + graphics_extension))
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

np.save(os.path.join(result_dir, eval_setting + '_inner_dist_mean.npy'), mean_inner_dist_per_timestep)
np.save(os.path.join(result_dir, eval_setting + '_inner_dist_var.npy'), var_inner_dist_per_timestep)
np.save(os.path.join(result_dir, eval_setting + '_inner_dist_is_clusters.npy'), inner_distances_is_clusters)


# mean distance score of within-class-differences of neuron activations per time step, averaged across all classes, for each network separately, and mean and variance among networks
with open(os.path.join(result_dir, eval_setting + "_neuron-act-inner-distances-per-time_H-" + str(training_hyp) + ".txt"), 'w') as f:
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



