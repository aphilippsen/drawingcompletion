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

num_inferences = 3 # how many inferences have been performed in each run to test the network
num_patterns = 6 # number of different training sample patterns

training_data_file = "data_generation/drawing-data-sets/drawings-191105-6x3-test.npy"
#data_set_name = 'example'
data_set_name = "final_0.01-100_6x7" #"training-2020-02-test-set"
#data_set_name = "2019-11-all-test-set"
mode = 'inference'
inf_epochs = np.concatenate((np.arange(1,2001,100), [2000]))

training_hyp_all = ['0.001', '0.01', '0.1', '1', '10', '100', '1000']

num_timesteps = 90
num_neurons = 100
graphics_extension = ".pdf"

# this code is only for one reduced_time_steps at a time! (reduced=0 is the array index that should be used)
reduced = 0

# for counting correct class inferences
correct_class = np.empty((len(training_hyp_all),), dtype=object)

training_hyp_idx = -1
for training_hyp in training_hyp_all:
    training_hyp_idx += 1
    correct_class[training_hyp_idx] = np.zeros((num_patterns))

    # Perform the PCA on inference data of all testing conditions?
    #test_hyp_all = ['0.001', '0.01', '0.1', '1', '10', '100', '1000']
    #eval_setting = "pca-all"
    # ... or only on inference data of the corresponding testing condition?
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

    dir_list = next(os.walk(eval_head_dir))[1]
    num_runs = len(dir_list)

    # data structures for collecting the distances between different training samples in the neural activation space in each time step
    mean_inner_dist_per_timestep = np.zeros((num_runs, num_timesteps))
    var_inner_dist_per_timestep = np.zeros((num_runs, num_timesteps))
    mean_outer_dist_per_timestep = np.zeros((num_runs, num_timesteps))
    var_outer_dist_per_timestep = np.zeros((num_runs, num_timesteps))
    mean_dist_reactive_per_timestep = np.zeros((num_runs*num_inferences*num_patterns, num_timesteps))
    var_dist_reactive_per_timestep = np.zeros((num_runs*num_inferences*num_patterns, num_timesteps))

    mean_inner_dist_is_epochs_per_timestep = np.zeros((num_runs, len(inf_epochs)))
    var_inner_dist_is_epochs_per_timestep = np.zeros((num_runs, len(inf_epochs)))
    mean_outer_dist_is_epochs_per_timestep = np.zeros((num_runs, len(inf_epochs)))
    var_outer_dist_is_epochs_per_timestep = np.zeros((num_runs, len(inf_epochs)))
    mean_dist_training_is_epochs = np.zeros((num_runs, len(inf_epochs)))

    # Analysis is performed separately for each of the trained networks (=runs) with the above defined H_train

    dist_to_corresponding_training_is_all = np.empty((len(dir_list),),dtype=object)
    dist_between_training_is_all = np.empty((len(dir_list),),dtype=object)
    between_dist = np.empty((len(test_hyp_all), len(dir_list)), dtype=object)
    within_dist = np.empty((len(test_hyp_all), len(dir_list)), dtype=object)
    
    for curr_r in range(len(dir_list)):
        current_run = os.path.join(eval_head_dir, dir_list[curr_r])
        print("Evaluate for " + str(current_run))

        # get training initial state (which is the same regardless of testing hyp_prior)
        corresponding_train_dir = os.path.join('./results/training/' + data_set_name, dir_list[curr_r])
        params, train_model = load_network(corresponding_train_dir + "/" + training_hyp, model_filename='network-epoch-best.npz')

        trained_is = train_model.initial_states.W.array

        # get the uh_history during reactive generation of the trained network
        x_train = np.float32(np.load(training_data_file))
        res, resv, resm, pe, wpe, uh_history_reactive, respos = train_model.generate(train_model.initial_states.W.array, num_timesteps, external_input = x_train[:6,:], add_variance_to_output = 0, additional_output='activations', external_signal_variance = train_model.external_signal_variance)


        # get the results from the inferences
        uh_history = np.empty((len(test_hyp_all), 1), dtype=object)
        inferred_is = np.empty((len(test_hyp_all), 1), dtype=object)
        is_history = np.empty((len(test_hyp_all), 1), dtype=object)
        
        test_hyp_idx = -1
        for test_hyp in test_hyp_all:
            test_hyp_idx += 1
            eval_dir = current_run+'/'+training_hyp+'/'+mode+'/test-'+test_hyp
            print("Load data from " + eval_dir)

            # load generation neuron activation
            uh_history[test_hyp_idx,0] = np.empty((num_inferences, num_patterns), dtype=object)
            inferred_is[test_hyp_idx,0] = np.empty((num_inferences, 1), dtype=object)

            is_history[test_hyp_idx,0] = np.empty((num_inferences, num_patterns), dtype=object)
            # load the current history
            uh = np.load(eval_dir + "/final-uh-history-"+test_hyp+"_mode-"+mode+".npy")

            final_is = np.load(eval_dir + "/final-inferred-is-"+test_hyp+"_mode-"+mode+".npy")

            # compute the similarity of the initial states inferred for different patterns            
            between_dist[test_hyp_idx, curr_r] = []
            within_dist[test_hyp_idx, curr_r] = []
            for pat1 in range(num_patterns):
                for pat2 in range(num_patterns):
                    for inf1 in range(num_inferences):
                        for inf2 in range(num_inferences):
                            if inf1 == inf2:
                                continue
                            # get the mean distance to same-pattern IS
                            if pat1 == pat2:
                                within_dist[test_hyp_idx, curr_r].append(scipy.spatial.distance.euclidean(final_is[0][inf1][pat1], final_is[0][inf2][pat2]))
                            # and get the mean distance to other-pattern IS
                            else:
                                between_dist[test_hyp_idx, curr_r].append(scipy.spatial.distance.euclidean(final_is[0][inf1][pat1], final_is[0][inf2][pat2]))
            within_dist[test_hyp_idx, curr_r] = np.mean(within_dist[test_hyp_idx, curr_r])
            between_dist[test_hyp_idx, curr_r] = np.mean(between_dist[test_hyp_idx, curr_r])

            # that's how the loaded uh should look like:
            #   uh[cl,red][inf].reshape((num_timesteps, num_neurons))
            # transfer info and do the reshape:
            for i in range(num_patterns): # for all pattern classes

                for j in range(num_inferences): # for all inferences for this pattern
                    uh_history[test_hyp_idx,0][j,i] = uh[i,0][j].reshape((num_timesteps, num_neurons))
                    inferred_is[test_hyp_idx,0][j,0] = final_is[0][j]

                    # evaluate whether inferred IS is correct or not
                    recogn_is = np.argmin(np.sqrt(np.sum((trained_is - np.tile(inferred_is[test_hyp_idx,0][j,0][i,:],(num_patterns,1)))**2,axis=1)))
                    if recogn_is == i:
                        correct_class[training_hyp_idx][i] += 1


            # get the trained initial states and the inferred initial states from all the ten inferences
            inf_nets_path = os.path.join(eval_dir, "inference_networks")
            inf_nets_list = next(os.walk(inf_nets_path))[1]
            no_cupy = False
            try:
                for curr_inf in range(len(inf_nets_list)):
                    # not required any more, as data is stored separately (loaded as final_is)
                    # # load each and extract and store the initial states
                    # # params, model = load_network(os.path.join(inf_nets_path, inf_nets_list[curr_inf]), model_filename='network-final')
                    # # inferred_is[test_hyp_idx,0][curr_inf,0] = model.initial_states.W.array

                    try:
                        for pat in range(num_patterns):
                            is_history[test_hyp_idx,0][curr_inf,pat] = np.zeros((len(inf_epochs),num_neurons))
                        ep_counter=0
                        for ep in inf_epochs:
                            params, model = load_network(os.path.join(inf_nets_path, inf_nets_list[curr_inf]), model_filename='network-epoch-'+str(ep).zfill(4))
                            for pat in range(num_patterns):
                                is_history[test_hyp_idx,0][curr_inf,pat][ep_counter,:] = model.initial_states.W.array[pat,:]
                            ep_counter += 1
                    except:
                        no_cupy = True
            except:
                print("Cannot load networks due to absence of cupy!!!")



        # reshape everything to (-1, num_neurons) for preparing for PCA
        inferred_is_temp = np.empty((len(test_hyp_all),1),dtype=object)
        for i in range(len(test_hyp_all)):
            inferred_is_temp[i,0] = np.concatenate(inferred_is[i,0][:,0],axis=0)
        all_inferred_is = np.concatenate(inferred_is_temp[:,0],axis=0)

        uh_history_temp = np.empty((len(test_hyp_all),1),dtype=object)
        for i in range(len(test_hyp_all)):
            uh_history_temp[i,0] = np.concatenate(uh_history[i,0].reshape((-1,1))[:,0],axis=0)
        all_uh_history = np.concatenate(uh_history_temp[:,0],axis=0)

        # all neuron activations collected into one large (-1, num_neurons) array
        all_neuron_activations = np.concatenate((trained_is, all_inferred_is, all_uh_history))
        all_initial_states = np.concatenate((trained_is, all_inferred_is))

        # Can do PCA either on the whole trajectory of activations or only on the initial states
        data_for_pca_transform = all_neuron_activations
        # data_for_pca_transform = all_initial_states

        # scaling data to achieve mean=0 and var=1
        scaler = StandardScaler().fit(data_for_pca_transform)
        data_for_pca_transform_scaled = scaler.transform(data_for_pca_transform)

        # create PCA mapping
        pca = PCA(n_components=num_neurons)
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

        # and training/reactive neuron activations
        pca_uh_history_reactive = np.empty((num_patterns,), dtype=object)
        for k in range(num_patterns):
            pca_uh_history_reactive[k] = pca.transform(scaler.transform(uh_history_reactive[k,:].reshape((num_timesteps,-1))))
            if normalize_for_statistics:
                pca_uh_history_reactive[k] = range2norm(pca_uh_history_reactive[k], norm_offset, norm_range, minmax)

        if not no_cupy:
            # and IS history
            pca_is_history = np.empty((len(test_hyp_all), 1), dtype=object)
            for i in range(len(test_hyp_all)): # per test_hyp
                pca_is_history[i,0] = np.empty((num_inferences, num_patterns), dtype=object)
                for j in range(num_inferences): # per inference
                    for k in range(num_patterns):
                        pca_is_history[i,0][j,k] = pca.transform(scaler.transform(is_history[i,0][j,k]))
                        if normalize_for_statistics:
                            pca_is_history[i,0][j,k] = range2norm(pca_is_history[i,0][j,k], norm_offset, norm_range, minmax)

        # store all the results as text files
        with open(os.path.join(result_dir, eval_setting + "_training-is_" + dir_list[curr_r] + "_H-" + str(training_hyp) + ".txt"), "w") as f:
            f.write('class')
            for i in range(num_neurons):
                f.write('\tdim' + str(i))
            f.write('\n')
            for pa in range(num_patterns):
                f.write(str(pa))
                for i in range(num_neurons):
                    f.write('\t' + str(pca_trained_is[pa,i]))
                f.write('\n')
        np.save(os.path.join(result_dir, eval_setting + "_training-is_" + dir_list[curr_r] + "_H-" + str(training_hyp) + ".npy"), pca_trained_is)

        for h in range(len(test_hyp_all)): # per test_hyp
            for inf in range(num_inferences): # per inference
                with open(os.path.join(result_dir, eval_setting + "_inferred-is_" + dir_list[curr_r] + '-' + str(inf) + '_H-' + test_hyp_all[h] + ".txt"), "w") as f:
                    f.write('class')
                    for i in range(num_neurons):
                        f.write('\tdim' + str(i))
                    f.write('\n')
                    for pa in range(num_patterns):
                        f.write(str(pa))
                        for i in range(num_neurons):
                            f.write('\t' + str(pca_inferred_is[h,0][inf,0][pa,i]))
                        f.write('\n')
                np.save(os.path.join(result_dir, eval_setting + "_inferred-is_" + dir_list[curr_r] + '-' + str(inf) + '_H-' + test_hyp_all[h] + ".npy"), pca_inferred_is[h,0][inf,0])

        for h in range(len(test_hyp_all)): # per test_hyp
            for inf in range(num_inferences): # per inference
                for pa in range(num_patterns): # per pattern class
                    with open(os.path.join(result_dir, eval_setting + "_uh-history_" + dir_list[curr_r] + '-' + str(inf) + "_pattern-" + str(pa) + '_H-' + test_hyp_all[h] + ".txt"), "w") as f:
                        f.write('t')
                        for i in range(num_neurons):
                            f.write('\tdim' + str(i))
                        f.write('\n')
                        for t in range(num_timesteps):
                            f.write(str(pa))
                            for i in range(num_neurons):
                                f.write('\t' + str(pca_uh_history[h,0][inf,pa][t,i]))
                            f.write('\n')
                    np.save(os.path.join(result_dir, eval_setting + "_uh-history_" + dir_list[curr_r] + '-' + str(inf) + "_pattern-" + str(pa) + '_H-' + test_hyp_all[h] + ".npy"), pca_uh_history[h,0][inf,pa])
                    if not no_cupy:
                        np.save(os.path.join(result_dir, eval_setting + "_is-history_" + dir_list[curr_r] + '-' + str(inf) + "_pattern-" + str(pa) + '_H-' + test_hyp_all[h] + ".npy"), pca_is_history[h,0][inf,pa])

        # PLOTTING

        colors = ['red', 'orange', 'green', 'blue', 'gray', 'black']
        pattern_category = ['FACE', 'HOUSE', 'CAR', 'FLOWER', 'HUMAN', 'ROCKET']
        to_plot_indices = [0, 1, 3] # [0, 1, 2, 3, 4, 5] # only subset of patterns to make plots clearer

        # 2d
        fig = plt.figure('Trained and inferred initial states (2d)', figsize=(20,20))
        plt.rcParams.update({'font.size': 50, 'legend.fontsize': 30})
        ax = fig.add_subplot(111)

        for th in plotting_test_hyps:
            for i in to_plot_indices:
                for j in range(num_inferences):
                    ax.scatter(pca_inferred_is[th,0][j][0][i,0], pca_inferred_is[th,0][j][0][i,1], color=colors[i], marker='o',s=1000)

        for i in to_plot_indices:
            ax.scatter(pca_trained_is[i,0], pca_trained_is[i,1], color=colors[i], marker='*',s=3000, label=pattern_category[i])

        plt.legend()
        plt.savefig(os.path.join(result_dir, eval_setting + '_inferred-is-2d_run-' + dir_list[curr_r] + graphics_extension))
        plt.close()

        # From 0 to 30

        # 2d
        fig = plt.figure('Neuron activations up to 30 timesteps (2d)', figsize=(10,10)) #figsize=(10, 11.0))
        plt.rcParams.update({'font.size': 30, 'legend.fontsize': 30})
        ax = fig.add_subplot(111)

        for th in plotting_test_hyps:
            for i in to_plot_indices:
                for j in range(num_inferences):
                    for t in range(30):
                        if t==1:
                            ax.scatter(pca_uh_history[th,0][j,i][t,0], pca_uh_history[th,0][j,i][t,1], color=colors[i], marker='*',s=1500)
                        elif t==29:
                            ax.scatter(pca_uh_history[th,0][j,i][t,0], pca_uh_history[th,0][j,i][t,1], color=colors[i], marker='s',s=1000)
                        else:
                            ax.scatter(pca_uh_history[th,0][j,i][t,0], pca_uh_history[th,0][j,i][t,1], color=colors[i], marker='o',s=200)

        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        #plt.legend()
        plt.savefig(os.path.join(result_dir, eval_setting + '_uh-2d_run-' + dir_list[curr_r] + graphics_extension))
        plt.close()



        # From 30 to end

        # 2d
        fig = plt.figure('Neuron activations after 30 timesteps (2d)', figsize=(10,10)) #figsize=(10, 11.0))
        plt.rcParams.update({'font.size': 30, 'legend.fontsize': 30})
        ax = fig.add_subplot(111)

        for th in plotting_test_hyps:
            for i in to_plot_indices:
                for j in range(num_inferences):
                    for t in np.arange(30,90):
                        if t==30:
                            ax.scatter(pca_uh_history[th,0][j,i][t,0], pca_uh_history[th,0][j,i][t,1], color=colors[i], marker='s',s=1000)
                        else:
                            ax.scatter(pca_uh_history[th,0][j,i][t,0], pca_uh_history[th,0][j,i][t,1], color=colors[i], marker='o',s=200)

        #plt.legend()
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        plt.savefig(os.path.join(result_dir, eval_setting + '_uh-30-to-end_2d_run-' + dir_list[curr_r] + graphics_extension))
        plt.close()

        # Analysis
        from scipy.spatial.distance import pdist, squareform

        # compare the clustering of the inferred initial states:
        # in the original space of neuron activations

        # collect all initial states for one pattern
        all_is_per_cluster = np.empty((num_patterns,1), dtype=object)
        inner_distances_is_clusters = []
        dist_to_corresponding_training_is = []
        for pat in range(num_patterns):
            all_is_per_cluster[pat,0] = np.zeros((len(plotting_test_hyps) * num_inferences, num_neurons))
            for i in range(len(plotting_test_hyps)): #range(len(test_hyp_all)):
                for j in range(num_inferences):
                    all_is_per_cluster[pat,0][i*num_inferences+j,:] = pca_inferred_is[i,0][j,0][pat,:]
                    dist_to_corresponding_training_is.append(scipy.spatial.distance.euclidean(pca_inferred_is[i,0][j,0][pat,:], pca_trained_is[pat,:]))

            inner_distances_is_clusters.append(np.mean(pdist(all_is_per_cluster[pat,0])))


        outer_distances_is_clusters = []
        for pat in range(num_patterns):
            for pat2 in range(num_patterns):
                if pat2 != pat:
                    for i in range(all_is_per_cluster[pat,0].shape[0]):
                        for j in range(all_is_per_cluster[pat2,0].shape[0]):
                            outer_distances_is_clusters.append(scipy.spatial.distance.euclidean(all_is_per_cluster[pat,0][i,:], all_is_per_cluster[pat2,0][j,:]))


        dist_to_corresponding_training_is_all[curr_r] = dist_to_corresponding_training_is

        dist_between_training_is_all[curr_r] = np.mean(pdist(pca_trained_is))

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


        # per time step, how far away from the other-pattern trajectories?
        for t in range(num_timesteps):

            outer_distances = []
            for pat in range(num_patterns):
                current_matrix = np.zeros((len(plotting_test_hyps) * num_inferences, num_neurons))
                current_other_matrix = np.zeros((len(plotting_test_hyps) * num_inferences * (num_patterns-1), num_neurons))

                for i in range(len(plotting_test_hyps)):
                    for j in range(num_inferences):
                        current_matrix[i*num_inferences+j,:] = pca_uh_history[i,0][j,pat][t,:]

                        offset=0
                        for pat2 in range(num_patterns):
                            if pat != pat2:
                                current_other_matrix[i*num_inferences+j+offset,:] = pca_uh_history[i,0][j,pat2][t,:]
                                offset += num_inferences*len(plotting_test_hyps)

                # now compare all entries in current_matrix to current_other_matrix
                for c1 in range(current_matrix.shape[0]):
                    for c2 in range(current_other_matrix.shape[0]):
                        outer_distances.append(scipy.spatial.distance.euclidean(current_matrix[c1,:], current_other_matrix[c2,:]))

            # take the mean of measured distances for this timestep for each timestep
            mean_outer_dist_per_timestep[curr_r,t] = np.mean(outer_distances)
            # take the mean over variance of patterns to get the variance
            var_outer_dist_per_timestep[curr_r,t] = np.var(outer_distances)

        # per epoch during inference, how close are the inferred IS?
        if not no_cupy:
            for t in range(len(inf_epochs)):

                inner_distances_is_epochs = []
                inner_dist_is_epochs_var = []
                dist_training_is_epochs = []
                for pat in range(num_patterns):
                    current_matrix = np.zeros((len(plotting_test_hyps) * num_inferences, num_neurons))

                    for i in range(len(plotting_test_hyps)): #range(len(test_hyp_all)):
                        for j in range(num_inferences):
                            current_matrix[i*num_inferences+j,:] = pca_is_history[i,0][j,pat][t,:]
                            dist_training_is_epochs.append(scipy.spatial.distance.euclidean(pca_is_history[i,0][j,pat][t,:], pca_trained_is[pat,:]))

                    inner_distances_is_epochs.append(np.mean(pdist(current_matrix)))
                    inner_dist_is_epochs_var.append(np.var(pdist(current_matrix)))

                # take the mean of measured distances for this timestep for each timestep
                mean_inner_dist_is_epochs_per_timestep[curr_r,t] = np.mean(inner_distances_is_epochs)
                # take the mean over variance of patterns to get the variance
                var_inner_dist_is_epochs_per_timestep[curr_r,t] = np.mean(inner_dist_is_epochs_var)

                mean_dist_training_is_epochs[curr_r,t] = np.mean(dist_training_is_epochs)

        if not no_cupy:
            for t in range(len(inf_epochs)):

                outer_distances_is_epochs = []
                for pat in range(num_patterns):
                    current_matrix = np.zeros((len(plotting_test_hyps) * num_inferences, num_neurons))
                    current_other_matrix = np.zeros((len(plotting_test_hyps) * num_inferences * (num_patterns-1), num_neurons))

                    for i in range(len(plotting_test_hyps)):
                        for j in range(num_inferences):
                            current_matrix[i*num_inferences+j,:] = pca_is_history[i,0][j,pat][t,:]

                            offset=0
                            for pat2 in range(num_patterns):
                                if pat != pat2:
                                    current_other_matrix[i*num_inferences+j+offset,:] = pca_is_history[i,0][j,pat2][t,:]
                                    offset += num_inferences*len(plotting_test_hyps)

                    # now compare all entries in current_matrix to current_other_matrix
                    for c1 in range(current_matrix.shape[0]):
                        for c2 in range(current_other_matrix.shape[0]):
                            outer_distances_is_epochs.append(scipy.spatial.distance.euclidean(current_matrix[c1,:], current_other_matrix[c2,:]))

                # take the mean of measured distances for this timestep for each timestep
                mean_outer_dist_is_epochs_per_timestep[curr_r,t] = np.mean(outer_distances_is_epochs)
                # take the mean over variance of patterns to get the variance
                var_outer_dist_is_epochs_per_timestep[curr_r,t] = np.var(outer_distances_is_epochs)

        # per time step, how close are the trajectories to the "reactive" trajectory?
        for t in range(num_timesteps):

            dist_to_reactive = []
            dist_to_reactive_var = []
            for pat in range(num_patterns): # separately for every pattern
                current_matrix = np.zeros((len(plotting_test_hyps) * num_inferences, 1))

                for i in range(len(plotting_test_hyps)): #range(len(test_hyp_all)):
                    for j in range(num_inferences):

                        current_matrix[i*num_inferences+j] = scipy.spatial.distance.euclidean(pca_uh_history[i,0][j,pat][t,:], pca_uh_history_reactive[pat][t,:])

                dist_to_reactive.append(current_matrix)
                dist_to_reactive_var.append(current_matrix)

            # take the mean of measured distances for this timestep for each timestep
            mean_dist_reactive_per_timestep[curr_r*num_inferences*num_patterns:curr_r*num_inferences*num_patterns+num_inferences*num_patterns,t] = np.concatenate(dist_to_reactive).reshape((num_inferences*num_patterns,))
            # take the mean over variance of patterns to get the variance
            var_dist_reactive_per_timestep[curr_r*num_inferences*num_patterns:curr_r*num_inferences*num_patterns+num_inferences*num_patterns,t] = np.concatenate(dist_to_reactive_var).reshape((num_inferences*num_patterns,))

    np.save(os.path.join(result_dir, eval_setting + '_inner_dist_mean.npy'), mean_inner_dist_per_timestep)
    np.save(os.path.join(result_dir, eval_setting + '_inner_dist_var.npy'), var_inner_dist_per_timestep)
    np.save(os.path.join(result_dir, eval_setting + '_inner_dist_is_clusters.npy'), inner_distances_is_clusters)
    np.save(os.path.join(result_dir, eval_setting + '_inner_dist_is_epochs_mean.npy'), mean_inner_dist_is_epochs_per_timestep)
    np.save(os.path.join(result_dir, eval_setting + '_inner_dist_is_epochs_var.npy'), var_inner_dist_is_epochs_per_timestep)
    np.save(os.path.join(result_dir, eval_setting + '_dist_training_is_epochs.npy'), mean_dist_training_is_epochs)

    np.save(os.path.join(result_dir, eval_setting + '_outer_dist_mean.npy'), mean_outer_dist_per_timestep)
    np.save(os.path.join(result_dir, eval_setting + '_outer_dist_var.npy'), var_outer_dist_per_timestep)
    np.save(os.path.join(result_dir, eval_setting + '_outer_dist_is_epochs_mean.npy'), mean_outer_dist_is_epochs_per_timestep)
    np.save(os.path.join(result_dir, eval_setting + '_outer_dist_is_epochs_var.npy'), var_outer_dist_is_epochs_per_timestep)
    np.save(os.path.join(result_dir, eval_setting + '_outer_dist_is_clusters.npy'), outer_distances_is_clusters)

    np.save(os.path.join(result_dir, eval_setting + "_dist_to_corresponding_training_is.npy"), dist_to_corresponding_training_is_all)
    np.save(os.path.join(result_dir, eval_setting + "_dist_between_training_is.npy"), dist_between_training_is_all)

    np.save(os.path.join(result_dir, eval_setting + '_dist_reactive_mean.npy'), mean_dist_reactive_per_timestep)
    np.save(os.path.join(result_dir, eval_setting + '_dist_reactive_var.npy'), var_dist_reactive_per_timestep)


    with open(os.path.join(result_dir, eval_setting + "_within_IS_dists_H-" + str(training_hyp) + ".txt"), 'w') as f:
        f.write("h\tdist\n")
        for i in range(len(dir_list)):
            #for j in range(len(within_dist[test_hyp_idx,i])):
                f.write(str(training_hyp) + "\t" + str(within_dist[test_hyp_idx,i]) + "\n")
    with open(os.path.join(result_dir, eval_setting + "_between_IS_dists_H-" + str(training_hyp) + ".txt"), 'w') as f:
        f.write("h\tdist\n")
        for i in range(len(dir_list)):
            #for j in range(len(between_dist[test_hyp_idx,i])):
                f.write(str(training_hyp) + "\t" + str(between_dist[test_hyp_idx,i]) + "\n")
                
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


    # mean distance score of between-class-differences of neuron activations per time step, averaged across all classes, for each network separately, and mean and variance among networks
    with open(os.path.join(result_dir, eval_setting + "_neuron-act-outer-distances-per-time_H-" + str(training_hyp) + ".txt"), 'w') as f:
        f.write("t\t")
        for net in range(num_runs):
            f.write("net" + str(net) + "\t")
        f.write("mean\tstd\n")
        for t in range(num_timesteps):
            f.write(str(t) + "\t")
            for net in range(num_runs):
                f.write(str(mean_outer_dist_per_timestep[net,t]) + "\t")
            f.write(str(np.mean(mean_outer_dist_per_timestep[:,t])) + "\t" + str(np.sqrt(np.var(mean_outer_dist_per_timestep[:,t]))) + "\n")

    with open(os.path.join(result_dir, eval_setting + "_dist_to_corresponding_training_is_H-" + str(training_hyp) + ".txt"), 'w') as f:
        f.write("h\tdist\n")
        for i in range(len(dist_to_corresponding_training_is_all)):
            for j in range(len(dist_to_corresponding_training_is_all[i])):
                f.write(str(training_hyp) + "\t" + str(dist_to_corresponding_training_is_all[i][j]) + "\n")

    with open(os.path.join(result_dir, eval_setting + "_dist_between_training_is_H-" + str(training_hyp) + ".txt"), 'w') as f:
        f.write("h\tdist\n")
        for i in range(len(dist_between_training_is_all)):
            f.write(str(training_hyp) + "\t" + str(dist_between_training_is_all[i]) + "\n")



    # and same thing for the IS epochs
    # mean distance score of within-class-differences
    with open(os.path.join(result_dir, eval_setting + "_neuron-act-inner-distances-is-epochs-per-time_H-" + str(training_hyp) + ".txt"), 'w') as f:
        f.write("t\t")
        for net in range(num_runs):
            f.write("net" + str(net) + "\t")
        f.write("mean\tstd\n")
        for t in range(len(inf_epochs)):
            f.write(str(t) + "\t")
            for net in range(num_runs):
                f.write(str(mean_inner_dist_is_epochs_per_timestep[net,t]) + "\t")
            f.write(str(np.mean(mean_inner_dist_is_epochs_per_timestep[:,t])) + "\t" + str(np.sqrt(np.var(mean_inner_dist_is_epochs_per_timestep[:,t]))) + "\n")


    # mean distance score of between-class-differences
    with open(os.path.join(result_dir, eval_setting + "_neuron-act-outer-distances-is-epochs-per-time_H-" + str(training_hyp) + ".txt"), 'w') as f:
        f.write("t\t")
        for net in range(num_runs):
            f.write("net" + str(net) + "\t")
        f.write("mean\tstd\n")
        for t in range(len(inf_epochs)):
            f.write(str(t) + "\t")
            for net in range(num_runs):
                f.write(str(mean_outer_dist_is_epochs_per_timestep[net,t]) + "\t")
            f.write(str(np.mean(mean_outer_dist_is_epochs_per_timestep[:,t])) + "\t" + str(np.sqrt(np.var(mean_outer_dist_is_epochs_per_timestep[:,t]))) + "\n")

    with open(os.path.join(result_dir, eval_setting + "_dist-to-training-is-epochs_H-" + str(training_hyp) + ".txt"), 'w') as f:
        f.write("t\t")
        for net in range(num_runs):
            f.write("net" + str(net) + "\t")
        f.write("mean\tstd\n")
        for t in range(len(inf_epochs)):
            f.write(str(t) + "\t")
            for net in range(num_runs):
                f.write(str(mean_dist_training_is_epochs[net,t]) + "\t")
            f.write(str(np.mean(mean_dist_training_is_epochs[:,t])) + "\t" + str(np.sqrt(np.var(mean_dist_training_is_epochs[:,t]))) + "\n")

# numbers of confusions / correct classifications
num_sample_per_pattern = num_runs * num_inferences
all_correct_class_percentage = np.asarray([np.sum(x) for x in correct_class])/(num_patterns*num_sample_per_pattern)*100
all_confusion_percentage = 100 - all_correct_class_percentage






