"""
Investigate how various H/sigmal configurations affect the drawing.
"""

import os
import numpy as np
import chainer
import matplotlib.pyplot as plt
import pathlib

from nets import SCTRNN, make_initial_state_zero, make_initial_state_random, NetworkParameterSetting, save_network, load_network, compute_prediction_error, compute_weighted_prediction_error
from inference import infer_initial_states_sctrnn
from utils.distance_measures import distance_measure

gpu_id=-1
xp=np

data_set_name = "final_0.01-100_6x7"
# go through the networks with H=1
#condition_directories = ['1']
hyp_prior_train = '1'

# multiplicators – change is relative to the original training parameters
test_reliance_priors = [0.001, 1, 1000]
test_reliance_input = [0.001, 1, 1000]

# get the data that should be completed – always use only a circle
num_timesteps = 90
reduced_time_steps = 30
num_classes = 6
num_io = 3
#training_data_file = "data_generation/drawing-data-sets/drawings-191105-6x7-train.npy"
#training_data_file_classes = "data_generation/drawing-data-sets/drawings-191105-6x7-train-classes.npy"
training_data_file = "data_generation/drawing-data-sets/drawings-191105-6x3-test.npy"
training_data_file_classes = "data_generation/drawing-data-sets/drawings-191105-6x3-test-classes.npy"
x_train = np.float32(np.load(training_data_file))
x_start = np.reshape(np.mean(x_train[:,0:num_io],axis=0), (1,-1))
classes = np.float32(np.load(training_data_file_classes))

input_class_to_use = [0,1,2,3,4,5]

# other parameters necessary for completion
high_sensory_variance = np.inf
is_selection_mode = 'inference'
inference_epochs = 100
start_inference = 'mean'
#start_inference = 'wrong'
#wrong_init_state = 0

used_measure='mse'

head_directory = "./results"
info = 'use_init_state_loss_input-class-' + str(input_class_to_use) + "_" + is_selection_mode  # for creating a subfolder
if is_selection_mode == 'inference':
    info += "-" + str(inference_epochs)
    if start_inference == 'mean':
        info += "_start_from_mean"
    elif start_inference == 'wrong':
        info += "_start_from_wrong-" + str(wrong_init_state)
elif is_selection_mode == 'inbetween':
    info += "-correct-" + str(input_class_to_use) + "_and-wrong-" + str(wrong_init_state)

final_res = np.empty((len(test_reliance_priors), len(test_reliance_input), num_classes), dtype=object)
final_err_vis_corr = np.empty((len(test_reliance_priors), len(test_reliance_input), num_classes), dtype=object)
final_err_vis_best = np.empty((len(test_reliance_priors), len(test_reliance_input), num_classes), dtype=object)
final_err_new_corr = np.empty((len(test_reliance_priors), len(test_reliance_input), num_classes), dtype=object)
final_err_new_best = np.empty((len(test_reliance_priors), len(test_reliance_input), num_classes), dtype=object)
final_err_vis_largest = np.empty((len(test_reliance_priors), len(test_reliance_input), num_classes), dtype=object)
final_err_new_largest = np.empty((len(test_reliance_priors), len(test_reliance_input), num_classes), dtype=object)
final_vis_best_class = np.empty((len(test_reliance_priors), len(test_reliance_input), num_classes), dtype=object)
final_new_best_class = np.empty((len(test_reliance_priors), len(test_reliance_input), num_classes), dtype=object)
final_uh_history = np.empty((len(test_reliance_priors), len(test_reliance_input), num_classes), dtype=object)

for i in range(final_res.shape[0]):
    for j in range(final_res.shape[1]):
        for k in range(final_res.shape[2]):
            final_res[i,j,k] = []
            final_err_vis_corr[i,j,k] = []
            final_err_vis_best[i,j,k] = []
            final_err_new_corr[i,j,k] = []
            final_err_new_best[i,j,k] = []
            final_err_vis_largest[i,j,k] = []
            final_err_new_largest[i,j,k] = []
            final_vis_best_class[i,j,k] = []
            final_new_best_class[i,j,k] = []
            final_uh_history[i,j,k] = []

for data_offset in range(0,x_train.shape[0], len(input_class_to_use)):

    input_data = x_train[data_offset:data_offset+len(input_class_to_use),:] #x_train[classes==input_class_to_use[0],:]
    #for i in input_class_to_use[1:]:
    #    input_data = np.concatenate((input_data, x_train[classes==input_class_to_use[0],:]),axis=0)

    delete_from = 30
    if gpu_id > -1:
        input_traj_zero = chainer.cuda.to_gpu(xp.copy(chainer.cuda.to_cpu(input_data.reshape((-1,num_io)))))
    else:
        input_traj_zero = xp.copy(input_data.reshape((-1,num_io)))
    for t in range(0, input_traj_zero.shape[0], num_timesteps):
        input_traj_zero[t+delete_from:t+num_timesteps,:] = 0
    input_traj_zero = input_traj_zero.reshape((input_data.shape[0],-1))

    training_dir = os.path.join(head_directory, "training/"+data_set_name)
    completion_dir = os.path.join(head_directory, "hyper_hypo_prior_input/"+data_set_name)

    #run_directories = next(os.walk(training_dir))[1]
    #for directory in run_directories:
    network_directories = ['2020-03-27_14-13_0354749', '2020-03-31_14-36_0053328', '2020-03-27_14-44_0454574', '2020-03-27_14-45_0803645', '2020-03-27_14-47_0212373', '2020-03-27_14-50_0500739', '2020-03-31_14-36_0152314', '2020-03-31_14-36_0777152', '2020-03-31_14-37_0004609', '2020-03-31_14-37_0843028']

    eval_vis_error_to_best = np.zeros((len(test_reliance_priors), len(test_reliance_input), len(network_directories)))
    eval_vis_confused = np.zeros((len(test_reliance_priors), len(test_reliance_input), len(network_directories)))
    eval_new_error_to_best = np.zeros((len(test_reliance_priors), len(test_reliance_input), len(network_directories)))
    eval_new_confused = np.zeros((len(test_reliance_priors), len(test_reliance_input), len(network_directories)))

    for directory in network_directories:
        run_dir = os.path.join(training_dir, directory)

        network_dir = os.path.join(run_dir, hyp_prior_train)

        result_dir = os.path.join(completion_dir, info)
        pathlib.Path(result_dir).mkdir(parents=True, exist_ok=True)

        params, model = load_network(network_dir, model_filename='network-epoch-best.npz')
        default_hyp_prior = params.hyp_prior
        default_sigma_input = params.external_signal_variance
        print("Training H: " + str(default_hyp_prior))


        print("Training parameters:\nH: " + str(default_hyp_prior) + "\nExternal signal variance: " + str(default_sigma_input))

        # for simplicity: use only the mean initial state
        # or: infer the initial state, one time for each circle... but that would always give face and could be rather easy... which is maybe what I want... with different priors it could give confusions... maybe I have to create my own data with circles that are not all closed, to make confusions more likely...
        # TODO or maybe use each potential initial states one time
        #init_state, res, results_path, u_h_history = complete_drawing(model, params, input_traj, reduced_time_steps, is_selection_mode = is_selection_mode, hyp_prior = hyp_prior, high_sensory_variance = high_sensory_variance, x_start = x_start, plottingFile = plottingFile, add_BI_variance = add_BI_variance, inference_epochs=inference_epochs, gpu_id = gpu_id)

        # in these cases the same initial state is used, regardless of H and sigma
        if is_selection_mode=='mean':
            init_state = np.mean(model.initial_states.W.array,axis=0).reshape((1, model.num_c))
        elif is_selection_mode=='best':
            init_state = np.tile(model.initial_states.W.array[input_class_to_use,:], (len(input_class_to_use),1))
        elif is_selection_mode=='wrong':
            init_state = np.tile(model.initial_states.W.array[wrong_init_state,:], (len(input_class_to_use),1))
        elif is_selection_mode=='inbetween':
            init_state = np.tile(0.5*model.initial_states.W.array[wrong_init_state,:]+0.5*model.initial_states.W.array[input_class_to_use,:], (len(input_class_to_use),1))


        for i_H, H_test in enumerate(test_reliance_priors):
            for i_s, sigma_test in enumerate(test_reliance_input):
                sigma_per_timestep = np.concatenate((np.tile(default_sigma_input*sigma_test, (reduced_time_steps,)), np.tile(np.inf,num_timesteps-reduced_time_steps)))

                if is_selection_mode=='inference':

                    init_state_file = os.path.join(result_dir, "offset-" + str(data_offset) + "_inference-result_" + directory + "_H-train_" + str(hyp_prior_train) + "_H-scale_" + str(H_test) + "_sigma-scale_" + str(sigma_test))

                    try:
                        init_state = np.load(init_state_file + ".npy")
                    except FileNotFoundError:
                        print("Could not load file: " + init_state_file + "\nGenerate...")

                        if start_inference == 'mean':
                            init_state, is_history, res, resm, results_path = infer_initial_states_sctrnn(params, model, input_data, epochs=inference_epochs, start_is='mean', num_timesteps = 30, hyp_prior = default_hyp_prior*H_test, external_signal_variance = default_sigma_input*sigma_test, x_start = x_start, use_init_state_loss=True)

                        elif start_inference == 'wrong':
                            init_state, is_history, res, resm, results_path = infer_initial_states_sctrnn(params, model, input_data, epochs=inference_epochs, start_is=np.tile(model.initial_states.W.array[wrong_init_state,:], (len(input_class_to_use),1)), num_timesteps = 30, hyp_prior = default_hyp_prior*H_test, external_signal_variance = default_sigma_input*sigma_test, x_start = x_start, use_init_state_loss=True)

                        init_state = init_state.W.array
                        np.save(init_state_file, init_state)

                res, resv, resm, pe, wpe, u_h_history, respost = model.generate(init_state, num_timesteps, external_input = input_data, add_variance_to_output = 0, hyp_prior = default_hyp_prior*H_test, external_signal_variance = sigma_per_timestep, additional_output='activations', x_start = x_start)

                pred_error = xp.zeros((input_data.shape[0],(num_timesteps-1)),dtype=xp.float32)
                weighted_pred_error = xp.zeros((input_data.shape[0],(num_timesteps-1)),dtype=xp.float32)
                for i in range(input_data.shape[0]):
                    pred_error[i,:] = compute_prediction_error(xp.reshape(input_traj_zero[i,:], (num_timesteps,-1)), xp.reshape(res[i,:],(num_timesteps, -1)), axis=1)
                    weighted_pred_error[i,:] = compute_weighted_prediction_error(xp.reshape(input_traj_zero[i,:],(num_timesteps, -1)), xp.reshape(res[i,:],(num_timesteps, -1)), xp.reshape(resv[i,:],(num_timesteps, -1)), axis=1)

                # get the posterior mean and standard deviation via BI
                sigma_per_timestep_approx = np.copy(sigma_per_timestep)
                sigma_per_timestep_approx[sigma_per_timestep_approx==np.inf] = 1000
                plt.figure()
                for i in range(len(input_class_to_use)):
                    input_mean = input_data[i,:]
                    pred_mean = res[i,:]
                    input_var = np.reshape(np.tile(sigma_per_timestep_approx, (num_io,1)), (-1))
                    pred_var = resv[i,:]

                    sigma_BI = np.sqrt(np.divide(np.multiply(input_var, pred_var), (input_var + pred_var)))
                    mu_BI = np.power(sigma_BI, 2) * (np.divide(pred_mean, pred_var) + np.divide(input_mean, input_var))

                    # if nan values occur it is because input variance is infinity, set to prediction values
                    mu_BI[np.isnan(mu_BI)] = pred_mean[np.isnan(mu_BI)]
                    sigma_BI[np.isnan(sigma_BI)] = pred_var[np.isnan(sigma_BI)]

                    mu_BI = mu_BI.reshape((-1,num_io))
                    sigma_BI = sigma_BI.reshape((-1,num_io))

                    plt.plot(np.arange(90), np.sum(sigma_BI,axis=1))
                plt.title('H=' + str(default_hyp_prior*H_test) + ', sigma=' + str(default_sigma_input*sigma_test))
                plt.savefig(os.path.join(result_dir, directory + "_offset-" + str(data_offset) + '_sigma_BI-' + str(H_test) + '_sigma-' + str(sigma_test) + '.png'))
                plt.close()

                # evaluate "drawing style"

                for i_c, curr_class in enumerate(input_class_to_use):
                    final_res[i_H, i_s, i_c].append(res[curr_class,:])
                    final_uh_history[i_H, i_s, i_c].append(u_h_history[curr_class,:])

                    generated_trajectory = res[curr_class,:].reshape((-1,model.num_io))
                    correct_trajectory = input_data[i_c,:].reshape((-1,model.num_io))
                    traj_vis_to_corr = distance_measure(correct_trajectory[0:reduced_time_steps,:], generated_trajectory[0:reduced_time_steps,:], method=used_measure)
                    traj_new_to_corr = distance_measure(correct_trajectory[reduced_time_steps:,:], generated_trajectory[reduced_time_steps:,:], method=used_measure)

                    all_trajectories = input_data
                    traj_vis_to_best = np.inf # some high PE for init
                    curr_largest_PE_vis = 0
                    for i in range(all_trajectories.shape[0]):
                        # compare to every trajectory
                        current_trajectory = all_trajectories[i,:].reshape((-1,model.num_io))
                        current_best_PE = distance_measure(current_trajectory[0:reduced_time_steps,:], generated_trajectory[0:reduced_time_steps,:], method=used_measure)
                        # store if it has the smallest PE so far:
                        if current_best_PE < traj_vis_to_best:
                            traj_vis_to_best = current_best_PE
                            traj_vis_best_class = i
                        if current_best_PE > curr_largest_PE_vis:
                            # remember the largest one (might be used later for scaling to relative instead of absolute errors!)
                            curr_largest_PE_vis = current_best_PE

                    # to any (the best) trajectory
                    traj_new_to_best = np.inf # some high PE for init
                    curr_largest_PE_new = 0
                    for i in range(all_trajectories.shape[0]):
                        # compare to every trajectory
                        current_trajectory = all_trajectories[i,:].reshape((-1,model.num_io))
                        current_best_PE = distance_measure(current_trajectory[reduced_time_steps:,:], generated_trajectory[reduced_time_steps:,:], method=used_measure)
                        # store if it has the smallest PE so far:
                        if current_best_PE < traj_new_to_best:
                            traj_new_to_best = current_best_PE
                            traj_new_best_class = i
                        if current_best_PE > curr_largest_PE_new:
                            # remember the largest one (might be used later for scaling to relative instead of absolute errors!)
                            curr_largest_PE_new = current_best_PE

                    final_err_vis_corr[i_H, i_s, curr_class].append(traj_vis_to_corr)
                    final_err_vis_best[i_H, i_s, curr_class].append(traj_vis_to_best)
                    final_err_new_corr[i_H, i_s, curr_class].append(traj_new_to_corr)
                    final_err_new_best[i_H, i_s, curr_class].append(traj_new_to_best)
                    final_err_vis_largest[i_H, i_s, curr_class].append(curr_largest_PE_vis)
                    final_err_new_largest[i_H, i_s, curr_class].append(curr_largest_PE_new)
                    final_vis_best_class[i_H, i_s, curr_class].append(traj_vis_best_class)
                    final_new_best_class[i_H, i_s, curr_class].append(traj_new_best_class)

                # print('Drawing style eval (H factor ' + str(H_test) + ', sigma factor ' + str(sigma_test) + '):\n')
                # print('traj_vis_to_corr: ' + str(final_err_vis_corr[i_H, i_s]))
                # print('traj_vis_to_best: ' + str(final_err_vis_best[i_H, i_s]))
                # print('traj_new_to_corr: ' + str(final_err_new_corr[i_H, i_s]))
                # print('traj_new_to_best: ' + str(final_err_new_best[i_H, i_s]))
                # print('traj_vis_best_class: ' + str(final_vis_best_class[i_H, i_s]))
                # print('traj_new_best_class: ' + str(final_new_best_class[i_H, i_s]))

                plt.figure()
                plt.plot(np.arange(90-1), pred_error[0,:])
                plt.plot(np.arange(90-1), pred_error[1,:])
                plt.plot(np.arange(90-1), pred_error[2,:])
                plt.ylim([0,1])
                plt.title('H=' + str(params.hyp_prior*H_test) + ', sigma=' + str(default_sigma_input*sigma_test))
                plt.savefig(os.path.join(result_dir, directory + "_offset-" + str(data_offset) + '_PE_H-' + str(H_test) + '_sigma-' + str(sigma_test) + '.png'))
                plt.close()

                plt.figure()
                plt.plot(np.arange(90-1), weighted_pred_error[0,:])
                plt.plot(np.arange(90-1), weighted_pred_error[1,:])
                plt.plot(np.arange(90-1), weighted_pred_error[2,:])
                plt.ylim([0,100])
                plt.title('H=' + str(params.hyp_prior*H_test) + ', sigma=' + str(default_sigma_input*sigma_test))
                plt.savefig(os.path.join(result_dir, directory + "_offset-" + str(data_offset) + '_weighted_PE_H-' + str(H_test) + '_sigma-' + str(sigma_test) + '.png'))
                plt.close()

                plt.figure()
                plt.plot(np.arange(90), np.mean(resv[0,:].reshape((-1,num_io)),axis=1))
                plt.plot(np.arange(90), np.mean(resv[1,:].reshape((-1,num_io)),axis=1))
                plt.plot(np.arange(90), np.mean(resv[2,:].reshape((-1,num_io)),axis=1))
                plt.ylim([0,1])
                plt.title('H=' + str(params.hyp_prior*H_test) + ', sigma=' + str(default_sigma_input*sigma_test))
                plt.savefig(os.path.join(result_dir, directory + "_offset-" + str(data_offset) + '_resv_H-' + str(H_test) + '_sigma-' + str(sigma_test) + '.png'))
                plt.close()

                #colors = ['b','b','b','y','y','y','g','g','g']
                colors = ['b','y','g','r','m','k']

                fig = plt.figure(figsize = (15,10))
                parameters = {'xtick.labelsize': 20,
                              'ytick.labelsize': 20,
                              'axes.labelsize': 25,
                              'axes.titlesize': 25}
                plt.rcParams.update(parameters)
                for s in range(input_data.shape[0]):
                    fig.add_subplot(2,3,s+1)
                    traj = res[s,:].reshape((num_timesteps,-1))
                    for t in range(1, num_timesteps):
                        if int(np.round(traj[t,2])) == 1:
                            if t < reduced_time_steps:
                                plt.plot(traj[t-1:t+1,0], traj[t-1:t+1,1], 'k', linewidth=5)
                            else:
                                plt.plot(traj[t-1:t+1,0], traj[t-1:t+1,1], 'g', linewidth=5)
                        else:
                            if t < reduced_time_steps:
                                plt.plot(traj[t-1:t+1,0], traj[t-1:t+1,1], 'lightgray', linewidth=5)
                            else:
                                plt.plot(traj[t-1:t+1,0], traj[t-1:t+1,1], '#11ff11', linewidth=5)

                    #plt.plot(traj[:,0],traj[:,1], color=colors[s])
                    if s==1:
                        plt.title('H=' + str(params.hyp_prior*H_test) + ', sigma=' + str(default_sigma_input*sigma_test))
                plt.xlim([-1,1])
                plt.ylim([-1,1])
                #plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(result_dir, directory + "_offset-" + str(data_offset) + '_drawing_H-' + str(H_test) + '_sigma-' + str(sigma_test) + '.pdf'))
                plt.close()


    np.save(os.path.join(result_dir, 'all-drawing-style-evals.npy'), [final_res, final_uh_history, final_err_vis_corr, final_err_vis_best, final_err_new_corr, final_err_new_best, final_err_vis_largest, final_err_new_largest, final_vis_best_class, final_new_best_class])


