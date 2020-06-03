import numpy as np
from chainer import cuda
import sys
sys.path.append('../')
import os
import pathlib
from distutils.dir_util import copy_tree
import shutil

# local imports
from nets import SCTRNN, make_initial_state_zero, make_initial_state_random, NetworkParameterSetting, save_network, load_network
from drawing_completion_functions import complete_drawing
from inference import infer_initial_states_sctrnn
from utils.visualization import plot_multistroke
from utils.distance_measures import distance_measure

gpu_id = 0 # -1 for CPU
xp = np
if gpu_id >= 0 and cuda.available:
    print("Use GPU!")
    cuda.get_device_from_id(gpu_id).use()
    xp = cuda.cupy
else:
    print("Use CPU!")
    gpu_id = -1

# if defined, this is the name of a subfolder of results/training where the trained networks to be used here are located
#data_set_name = "example"
data_set_name = "final_0.01-100_6x7"#"tmpComp1-training-set"#"training-2020-02-new-completion"#"training-2020-03_noise0.01"#test-set"
#data_set_name = "2019-11-all-test-set"

# which training parameter conditions to check
condition_directories = ['1']
# which hyp_prior condition to use for testing:
test_hyp_priors = [1]

# which value for σ2_sensor should be assumed if no external input is available (affects amount of randomness of drawing in hypo-prior condition)
high_sensory_variance = np.inf

used_measure = 'dtw'

# trajectory data
training_data_file = "data_generation/drawing-data-sets/drawings-191105-6x3-test.npy"#-drawings.npy"
training_data_file_classes = "data_generation/drawing-data-sets/drawings-191105-6x3-test-classes.npy"#drawings-classes.npy"
#training_data_file = "data_generation/drawing-data-sets/drawings-191105-6-drawings-test-set.npy"
#training_data_file_classes = "data_generation/drawing-data-sets/drawings-191105-6-drawings-test-set-classes.npy"
num_timesteps = 90
num_classes = 6
num_io = 3

# Which initial state to use for generation
# is_selection_mode = 'zero' # take zero vector
# is_selection_mode = 'mean' # take the mean of all training initial states
# is_selection_mode = 'best' # try all available initial states and use the one that best replicates the existing part
is_selection_mode = 'inference' # use backpropagation inference to infer the best fitting initial states

# For how many epochs to perform inference
inference_epochs=2000
# How many of the num_timesteps do we want to provide to the network
reduced_time_steps_list = [30]

add_BI_variance = True

# number of drawings available per class (completion is tried once for every available drawing)
drawings_per_class = 3

x_train = np.float32(np.load(training_data_file))
# the mean of all trajectory starting points
x_start = np.reshape(np.mean(x_train[:,0:num_io],axis=0), (1,-1))
if gpu_id >= 0:
    x_train = cuda.to_gpu(x_train)
    x_start = cuda.to_gpu(x_start)


# where to find the training networks
head_directory = "./results"
info = ""
info += is_selection_mode # for creating a subfolder

training_dir = os.path.join(head_directory, "training/"+data_set_name)
completion_dir = os.path.join(head_directory, "completion/"+data_set_name)

run_directories = next(os.walk(training_dir))[1]
for current_r in range(len(run_directories)):
    run_dir = os.path.join(training_dir, run_directories[current_r])

    for current_c in range(len(condition_directories)):
        network_dir = os.path.join(run_dir, condition_directories[current_c])
        inference_results_dir = os.path.join(os.path.join(completion_dir, run_directories[current_r]), condition_directories[current_c])
        pathlib.Path(inference_results_dir).mkdir(parents=True, exist_ok=True)

        params, model = load_network(network_dir, model_filename='network-epoch-best.npz')

        # do the inference and generation for all testing hyp_prior values, and for all training trajectories
        for hyp_prior in test_hyp_priors:
            params, model = load_network(network_dir, model_filename='network-epoch-best.npz')

            if gpu_id >= 0:
                model.to_gpu(gpu_id)

            # arrays for keeping the final results
            final_results_path = np.empty((num_classes, len(reduced_time_steps_list)), dtype=object)
            final_res = np.empty((num_classes, len(reduced_time_steps_list)), dtype=object)
            final_err_vis_corr = np.empty((num_classes, len(reduced_time_steps_list)), dtype=object)
            final_err_vis_best = np.empty((num_classes, len(reduced_time_steps_list)), dtype=object)
            final_err_new_corr = np.empty((num_classes, len(reduced_time_steps_list)), dtype=object)
            final_err_new_best = np.empty((num_classes, len(reduced_time_steps_list)), dtype=object)
            final_err_vis_largest = np.empty((num_classes, len(reduced_time_steps_list)), dtype=object)
            final_err_new_largest = np.empty((num_classes, len(reduced_time_steps_list)), dtype=object)
            final_vis_best_class = np.empty((num_classes, len(reduced_time_steps_list)), dtype=object)
            final_new_best_class = np.empty((num_classes, len(reduced_time_steps_list)), dtype=object)
            final_uh_history = np.empty((num_classes, len(reduced_time_steps_list)), dtype=object)
            final_inferred_is = np.empty((len(reduced_time_steps_list),), dtype=object)

            for i in range(final_res.shape[0]):
                for j in range(final_res.shape[1]):
                    final_results_path[i,j] = []
                    final_res[i,j] = []
                    final_err_vis_corr[i,j] = []
                    final_err_vis_best[i,j] = []
                    final_err_new_corr[i,j] = []
                    final_err_new_best[i,j] = []
                    final_err_vis_largest[i,j] = []
                    final_err_new_largest[i,j] = []
                    final_vis_best_class[i,j] = []
                    final_new_best_class[i,j] = []
                    final_uh_history[i,j] = []
                    final_inferred_is[j] = []

            # for all input trajectory patterns

            # where to store the results for this inference
            results_dir = os.path.join(inference_results_dir, info + "/test-" + str(hyp_prior))
            pathlib.Path(results_dir).mkdir(parents=True, exist_ok=True)
            print("Working on results directory: " + results_dir)

            # if inference for this network is already done, skip
            if os.path.isfile(os.path.join(results_dir, 'final-res_hyp-' + str(hyp_prior) + '_mode-' + str(is_selection_mode) + '.npy')):
                print("Skip already existing results for " + results_dir)
                continue

            # infer all three classes at once
            r=0
            for p in range(0, num_classes*drawings_per_class, num_classes):
                input_traj = x_train[p:p+num_classes,:]

                # for different numbers of available steps
                for reduced_time_steps in reduced_time_steps_list:
                    plottingFile = os.path.join(results_dir, 'hyp-' + str(hyp_prior) + '_mode-' + str(is_selection_mode) + '_reduced-' + str(reduced_time_steps) +'_run-' + str(r))

                    init_state, res, results_path, u_h_history = complete_drawing(model, params, input_traj, reduced_time_steps, is_selection_mode = is_selection_mode, hyp_prior = hyp_prior, high_sensory_variance = high_sensory_variance, x_start = x_start, plottingFile = plottingFile, add_BI_variance = add_BI_variance, inference_epochs=inference_epochs, gpu_id = gpu_id)

                    final_inferred_is[reduced_time_steps_list.index(reduced_time_steps)].append(cuda.to_cpu(init_state))
                    for curr_class in range(num_classes):
                        final_res[curr_class, reduced_time_steps_list.index(reduced_time_steps)].append(cuda.to_cpu(res[curr_class,:]))
                        final_uh_history[curr_class, reduced_time_steps_list.index(reduced_time_steps)].append(cuda.to_cpu(u_h_history[curr_class,:]))

                        generated_trajectory = res[curr_class,:].reshape((-1,model.num_io))
                        correct_trajectory = input_traj[curr_class,:].reshape((-1,model.num_io))
                        traj_vis_to_corr = distance_measure(correct_trajectory[1:reduced_time_steps,:], generated_trajectory[0:reduced_time_steps-1,:], method=used_measure)

                        all_trajectories = x_train[p:p+num_classes,:]
                        traj_vis_to_best = 100 # some high PE for init
                        curr_largest_PE_vis = 0
                        for i in range(all_trajectories.shape[0]):
                            # compare to every trajectory
                            current_trajectory = all_trajectories[i,:].reshape((-1,model.num_io))
                            current_best_PE = distance_measure(current_trajectory[1:reduced_time_steps,:], generated_trajectory[0:reduced_time_steps-1,:], method=used_measure)
                            # store if it has the smallest PE so far:
                            if current_best_PE < traj_vis_to_best:
                                traj_vis_to_best = current_best_PE
                                traj_vis_best_class = i
                            if current_best_PE > curr_largest_PE_vis:
                                # remember the largest one (might be used later for scaling to relative instead of absolute errors!)
                                curr_largest_PE_vis = current_best_PE

                        # PE of traj_new part
                        # to the correct trajectory
                        traj_new_to_corr = distance_measure(correct_trajectory[reduced_time_steps+1:,:], generated_trajectory[reduced_time_steps:-1,:], method=used_measure)

                        # to any (the best) trajectory
                        traj_new_to_best = 100 # some high PE for init
                        curr_largest_PE_new = 0
                        for i in range(all_trajectories.shape[0]):
                            # compare to every trajectory
                            current_trajectory = all_trajectories[i,:].reshape((-1,model.num_io))
                            current_best_PE = distance_measure(current_trajectory[reduced_time_steps+1:,:], generated_trajectory[reduced_time_steps:-1,:], method=used_measure)
                            # store if it has the smallest PE so far:
                            if current_best_PE < traj_new_to_best:
                                traj_new_to_best = current_best_PE
                                traj_new_best_class = i
                            if current_best_PE > curr_largest_PE_new:
                                # remember the largest one (might be used later for scaling to relative instead of absolute errors!)
                                curr_largest_PE_new = current_best_PE

                        final_err_vis_corr[curr_class, reduced_time_steps_list.index(reduced_time_steps)].append(traj_vis_to_corr)
                        final_err_vis_best[curr_class, reduced_time_steps_list.index(reduced_time_steps)].append(traj_vis_to_best)
                        final_err_new_corr[curr_class, reduced_time_steps_list.index(reduced_time_steps)].append(traj_new_to_corr)
                        final_err_new_best[curr_class, reduced_time_steps_list.index(reduced_time_steps)].append(traj_new_to_best)
                        final_err_vis_largest[curr_class, reduced_time_steps_list.index(reduced_time_steps)].append(curr_largest_PE_vis)
                        final_err_new_largest[curr_class, reduced_time_steps_list.index(reduced_time_steps)].append(curr_largest_PE_new)
                        final_vis_best_class[curr_class, reduced_time_steps_list.index(reduced_time_steps)].append(traj_vis_best_class)
                        final_new_best_class[curr_class, reduced_time_steps_list.index(reduced_time_steps)].append(traj_new_best_class)

                        if curr_class == num_classes-1:
                            # we are done with all classes, next trial starts
                            r += 1

                            np.save(os.path.join(results_dir, 'final-res_hyp-' + str(hyp_prior) + '_mode-' + str(is_selection_mode) + '.npy'), final_res)
                            np.save(os.path.join(results_dir, 'final-resultspath_hyp-' + str(hyp_prior) + '_mode-' + str(is_selection_mode) + '.npy'), final_results_path)
                            np.save(os.path.join(results_dir, 'final-err_vis_corr-' + str(hyp_prior) + '_mode-' + str(is_selection_mode) + '.npy'), final_err_vis_corr)
                            np.save(os.path.join(results_dir, 'final-err_vis_best-' + str(hyp_prior) + '_mode-' + str(is_selection_mode) + '.npy'), final_err_vis_best)
                            np.save(os.path.join(results_dir, 'final-err_new_corr-' + str(hyp_prior) + '_mode-' + str(is_selection_mode) + '.npy'), final_err_new_corr)
                            np.save(os.path.join(results_dir, 'final-err_new_best-' + str(hyp_prior) + '_mode-' + str(is_selection_mode) + '.npy'), final_err_new_best)
                            np.save(os.path.join(results_dir, 'final-err_vis_largest-' + str(hyp_prior) + '_mode-' + str(is_selection_mode) + '.npy'), final_err_vis_largest)
                            np.save(os.path.join(results_dir, 'final-err_new_largest-' + str(hyp_prior) + '_mode-' + str(is_selection_mode) + '.npy'), final_err_new_largest)
                            np.save(os.path.join(results_dir, 'final-vis_best_class-' + str(hyp_prior) + '_mode-' + str(is_selection_mode) + '.npy'), final_vis_best_class)
                            np.save(os.path.join(results_dir, 'final-new_best_class-' + str(hyp_prior) + '_mode-' + str(is_selection_mode) + '.npy'), final_new_best_class)
                            np.save(os.path.join(results_dir, 'final-uh-history-' + str(hyp_prior) + '_mode-' + str(is_selection_mode) + '.npy'), final_uh_history)
                            np.save(os.path.join(results_dir, 'final-inferred-is-' + str(hyp_prior) + '_mode-' + str(is_selection_mode) + '.npy'), final_inferred_is)


                np.save(os.path.join(results_dir, 'final-res_hyp-' + str(hyp_prior) + '_mode-' + str(is_selection_mode) + '.npy'), final_res)
                np.save(os.path.join(results_dir, 'final-resultspath_hyp-' + str(hyp_prior) + '_mode-' + str(is_selection_mode) + '.npy'), final_results_path)
                np.save(os.path.join(results_dir, 'final-err_vis_corr-' + str(hyp_prior) + '_mode-' + str(is_selection_mode) + '.npy'), final_err_vis_corr)
                np.save(os.path.join(results_dir, 'final-err_vis_best-' + str(hyp_prior) + '_mode-' + str(is_selection_mode) + '.npy'), final_err_vis_best)
                np.save(os.path.join(results_dir, 'final-err_new_corr-' + str(hyp_prior) + '_mode-' + str(is_selection_mode) + '.npy'), final_err_new_corr)
                np.save(os.path.join(results_dir, 'final-err_new_best-' + str(hyp_prior) + '_mode-' + str(is_selection_mode) + '.npy'), final_err_new_best)
                np.save(os.path.join(results_dir, 'final-err_vis_largest-' + str(hyp_prior) + '_mode-' + str(is_selection_mode) + '.npy'), final_err_vis_largest)
                np.save(os.path.join(results_dir, 'final-err_new_largest-' + str(hyp_prior) + '_mode-' + str(is_selection_mode) + '.npy'), final_err_new_largest)
                np.save(os.path.join(results_dir, 'final-vis_best_class-' + str(hyp_prior) + '_mode-' + str(is_selection_mode) + '.npy'), final_vis_best_class)
                np.save(os.path.join(results_dir, 'final-new_best_class-' + str(hyp_prior) + '_mode-' + str(is_selection_mode) + '.npy'), final_new_best_class)
                np.save(os.path.join(results_dir, 'final-uh-history-' + str(hyp_prior) + '_mode-' + str(is_selection_mode) + '.npy'), final_uh_history)
                np.save(os.path.join(results_dir, 'final-inferred-is-' + str(hyp_prior) + '_mode-' + str(is_selection_mode) + '.npy'), final_inferred_is)
                
                # for later evaluation, store parameters
                if is_selection_mode == 'inference':
                    # results_path is the folder with the inference results
                    final_results_path[curr_class, reduced_time_steps_list.index(reduced_time_steps)].append(results_path)

                    # store the corresponding networks
                    dest = results_dir + "/inference_networks/"
                    pathlib.Path(dest).mkdir(parents=True, exist_ok=True)
                    copy_tree(results_path, dest + results_path.split('/')[-1])
                    # and delete the old network at the old location
                    shutil.rmtree(results_path)

