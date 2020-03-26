import os
import numpy as np
import matplotlib.pyplot as plt
from chainer import cuda

from nets import SCTRNN, make_initial_state_zero, make_initial_state_random, NetworkParameterSetting, save_network, load_network
from drawing_completion_functions import complete_drawing
from utils.distance_measures import distance_measure

reduced_time_steps = 30
add_BI_variance = True
num_classes = 6
num_runs = 5

# which training parameter conditions to check
condition_directories = ['0.001', '1', '1000']
# which hyp_prior condition to use for testing:
test_hyp_priors =  [0.001, 1, 1000]

data_set_name = 'training-2020-02-test-set'

# where to find the training networks
head_directory = "./results"

training_dir = os.path.join(head_directory, "training/"+data_set_name)
eval_dir = os.path.join(head_directory, "evaluation/"+data_set_name)

interps = np.arange(10,-1, -1)/10

gpu_id = -1 # -1 for CPU
xp = np
if gpu_id >= 0 and cuda.available:
    print("Use GPU!")
    cuda.get_device_from_id(gpu_id).use()
    xp = cuda.cupy
else:
    print("Use CPU!")
    gpu_id = -1

training_data_file = "data_generation/drawing-data-sets/drawings-191105-6-drawings.npy"
training_data_file_classes = "data_generation/drawing-data-sets/drawings-191105-6-drawings-classes.npy"
x_train = np.float32(np.load(training_data_file))
if gpu_id >= 0:
    x_train = cuda.to_gpu(x_train)
# else:
#     x_train = cuda.to_cpu(x_train)


# for the different runs
run_directories = next(os.walk(training_dir))[1]
for current_c in range(len(condition_directories)):

    try:
        x = 5/0
        attractor_distance_vis_error_results_1 = np.load(os.path.join(eval_dir, 'hyp-' + str(test_hyp_priors[current_c]) + '_attractor-dist-visible-1.npy'))
        attractor_distance_new_error_results_1 = np.load(os.path.join(eval_dir, 'hyp-' + str(test_hyp_priors[current_c]) + '_attractor-dist-new-1.npy'))
        attractor_distance_vis_error_results_2 = np.load(os.path.join(eval_dir, 'hyp-' + str(test_hyp_priors[current_c]) + '_attractor-dist-visible-2.npy'))
        attractor_distance_new_error_results_2 = np.load(os.path.join(eval_dir, 'hyp-' + str(test_hyp_priors[current_c]) + '_attractor-dist-new-2.npy'))
        print("loaded")

    except:
        print("generate")

        # the distance to the first pattern (interp == 1)
        attractor_distance_vis_error_results_1 = np.empty((len(run_directories),),dtype=object)
        attractor_distance_new_error_results_1 = np.empty((len(run_directories),),dtype=object)
        # the distance to the second pattern (interp == 0)
        attractor_distance_vis_error_results_2 = np.empty((len(run_directories),),dtype=object)
        attractor_distance_new_error_results_2 = np.empty((len(run_directories),),dtype=object)

        for current_r in range(len(run_directories)):
            run_dir = os.path.join(training_dir, run_directories[current_r])
            network_dir = os.path.join(run_dir, condition_directories[current_c])
            params, model = load_network(network_dir, model_filename='network-epoch-best.npz')

            # get the initial states inferred during training
            train_is = model.initial_states.W.array

            attractor_distance_vis_error_results_1[current_r] = np.empty((len(interps),), dtype=object)
            attractor_distance_new_error_results_1[current_r] = np.empty((len(interps),), dtype=object)
            attractor_distance_vis_error_results_2[current_r] = np.empty((len(interps),), dtype=object)
            attractor_distance_new_error_results_2[current_r] = np.empty((len(interps),), dtype=object)

            # interpolation coefficient
            for j in interps:
                attractor_distance_vis_error_results_1[current_r][interps.tolist().index(j)] = []
                attractor_distance_new_error_results_1[current_r][interps.tolist().index(j)] = []
                attractor_distance_vis_error_results_2[current_r][interps.tolist().index(j)] = []
                attractor_distance_new_error_results_2[current_r][interps.tolist().index(j)] = []

                # for two fixed patters
                for i_1 in [0]:#np.arange(6):
                    for i_2 in [1]: #np.arange(6):
                        print(str(i_1) + ", " + str(i_2))
                # for every pair of two IS
#                for i_1 in range(train_is.shape[0]):
#                    for i_2 in range(train_is.shape[0]):
                        if i_1 == i_2:
                            continue
                        print(str(i_1) + ", " + str(i_2))

                        is_1 = train_is[i_1,:]
                        is_2 = train_is[i_2,:]

                        # interpolated initial state
                        interpol_is = j * is_1 + (1-j) * is_2


                        # use this interpolated IS to generate pattern 1

                        # corresponding training trajectory
                        input_traj = xp.tile(x_train[i_1,:], (num_classes,1))

                        # generate the trajectory using the corresponding training_hyprun
                        plottingFile = os.path.join(eval_dir, 'pattern-1_hyp-' + str(test_hyp_priors[current_c]) + '_run-' + str(current_r) + '_interp-' + str(np.around(j,2)) + '_attractors_' + str(i_1) + '-' + str(i_2) +  '_')

                        init_state, res, results_path, u_h_history = complete_drawing(model, params, input_traj, reduced_time_steps, is_selection_mode = np.tile(interpol_is,(num_classes,1)), hyp_prior = test_hyp_priors[current_c], x_start = None, plottingFile = plottingFile, add_BI_variance = add_BI_variance, gpu_id=-1)

                        # calculate error of the generated shape to the intended shape
                        # visible part (to pattern 1)
                        cl = i_1
                        generated_trajectory = res[cl,:].reshape((-1,model.num_io))
                        correct_trajectory = input_traj[cl,:].reshape((-1,model.num_io))
                        attractor_distance_vis_error_results_1[current_r][interps.tolist().index(j)].append(distance_measure(correct_trajectory[1:reduced_time_steps,:], generated_trajectory[0:reduced_time_steps-1,:], method = 'dtw'))
                        # invisible part (to pattern 1)
                        generated_trajectory = res[cl,:].reshape((-1,model.num_io))
                        correct_trajectory = input_traj[cl,:].reshape((-1,model.num_io))
                        attractor_distance_new_error_results_1[current_r][interps.tolist().index(j)].append(distance_measure(correct_trajectory[reduced_time_steps:,:], generated_trajectory[reduced_time_steps:,:], method = 'dtw'))

                        # use this interpolated IS to generate pattern 2

                        # corresponding training trajectory
                        input_traj = xp.tile(x_train[i_2,:], (num_classes,1))

                        # generate the trajectory using the corresponding training_hyprun
                        plottingFile = os.path.join(eval_dir, 'pattern-2_hyp-' + str(test_hyp_priors[current_c]) + '_run-' + str(current_r) + '_interp-' + str(np.around(j,2)) + '_attractors_' + str(i_1) + '-' + str(i_2) +  '_')

                        init_state, res, results_path, u_h_history = complete_drawing(model, params, input_traj, reduced_time_steps, is_selection_mode = np.tile(interpol_is,(num_classes,1)), hyp_prior = test_hyp_priors[current_c], x_start = None, plottingFile = plottingFile, add_BI_variance = add_BI_variance, gpu_id=-1)

                        # visible part (to pattern 2)
                        cl = i_2 #range(num_classes):
                        generated_trajectory = res[cl,:].reshape((-1,model.num_io))
                        correct_trajectory = input_traj[cl,:].reshape((-1,model.num_io))
                        attractor_distance_vis_error_results_2[current_r][interps.tolist().index(j)].append(distance_measure(correct_trajectory[1:reduced_time_steps,:], generated_trajectory[0:reduced_time_steps-1,:], method = 'dtw'))
                        # invisible part (to pattern 2)
                        generated_trajectory = res[cl,:].reshape((-1,model.num_io))
                        correct_trajectory = input_traj[cl,:].reshape((-1,model.num_io))
                        attractor_distance_new_error_results_2[current_r][interps.tolist().index(j)].append(distance_measure(correct_trajectory[reduced_time_steps:,:], generated_trajectory[reduced_time_steps:,:], method = 'dtw'))


        np.save(os.path.join(eval_dir, 'hyp-' + str(test_hyp_priors[current_c]) + '_attractor-dist-visible-1.npy'), attractor_distance_vis_error_results_1)
        np.save(os.path.join(eval_dir, 'hyp-' + str(test_hyp_priors[current_c]) + '_attractor-dist-new-1.npy'), attractor_distance_new_error_results_1)
        np.save(os.path.join(eval_dir, 'hyp-' + str(test_hyp_priors[current_c]) + '_attractor-dist-visible-2.npy'), attractor_distance_vis_error_results_2)
        np.save(os.path.join(eval_dir, 'hyp-' + str(test_hyp_priors[current_c]) + '_attractor-dist-new-2.npy'), attractor_distance_new_error_results_2)

# for now, all classes are condensed
colors = ['blue', 'red', 'green']

fig = plt.figure('New_best', figsize=(35.0, 12.0))
plt.rcParams.update({'font.size': 20, 'legend.fontsize': 20})
ax1 = fig.add_subplot(131) # visible part
ax2 = fig.add_subplot(132) # new part
ax3 = fig.add_subplot(133) # new part

# load parameter condition
c=0
for current_c in condition_directories:
    attractor_distance_vis_error_results_1 = np.load(os.path.join(eval_dir, 'hyp-' + current_c + '_attractor-dist-visible-1.npy'))
    attractor_distance_new_error_results_1 = np.load(os.path.join(eval_dir, 'hyp-' + current_c + '_attractor-dist-new-1.npy'))
    attractor_distance_vis_error_results_2 = np.load(os.path.join(eval_dir, 'hyp-' + current_c + '_attractor-dist-visible-2.npy'))
    attractor_distance_new_error_results_2 = np.load(os.path.join(eval_dir, 'hyp-' + current_c + '_attractor-dist-new-2.npy'))

    
    # all "other" patterns compared to pattern 0, plotting patterns individually

    collect_over_patterns_1 = np.zeros((num_classes-1, len(interps), num_runs))
    for r in range(5):
        all_error_trajs = np.concatenate(attractor_distance_new_error_results_1[r]).reshape((len(interps),-1))[:,0:num_classes-1]
        for pat in range(num_classes-1):
            collect_over_patterns_1[pat][:,r] = all_error_trajs[:,pat]
    for pat in range(num_classes-1):
        meanpat = np.mean(collect_over_patterns_1[pat], axis=1)
        stdpat = np.std(collect_over_patterns_1[pat], axis=1)
        if c==0:
            ax1.errorbar(interps, meanpat, stdpat, color=colors[0])#, label=str(current_c) + " (1)")
        if c==1:
            ax2.errorbar(interps, meanpat, stdpat, color=colors[0])#, label=str(current_c) + " (1)")
        if c==2:
            ax3.errorbar(interps, meanpat, stdpat, color=colors[0])#, label=str(current_c) + " (1)")

    collect_over_patterns_2 = np.zeros((num_classes-1, len(interps), num_runs))
    for r in range(5):
        all_error_trajs = np.concatenate(attractor_distance_new_error_results_2[r]).reshape((len(interps),-1))[:,0:num_classes-1]
        for pat in range(num_classes-1):
            collect_over_patterns_2[pat][:,r] = all_error_trajs[:,pat]
    for pat in range(num_classes-1):
        meanpat = np.mean(collect_over_patterns_2[pat], axis=1)
        stdpat = np.std(collect_over_patterns_2[pat], axis=1)
        if c==0:
            ebplot = ax1.errorbar(interps, meanpat, stdpat, color=colors[1])#, label=str(current_c) + " (1)")
            ebplot[-1][0].set_linestyle('--')
        if c==1:
            ebplot = ax2.errorbar(interps, meanpat, stdpat, color=colors[1])#, label=str(current_c) + " (1)")
            ebplot[-1][0].set_linestyle('--')
        if c==2:
            ebplot = ax3.errorbar(interps, meanpat, stdpat, color=colors[1])#, label=str(current_c) + " (1)")
            ebplot[-1][0].set_linestyle('--')


    with open("indiv-pattern-error-1_" + current_c + ".txt", 'w') as f:
        f.write("ip\tmeanp1\tstdp1\tmeanp2\tstdp2\tmeanp3\tstdp3\tmeanp4\tstdp4\tmeanp5\tstdp5\n")
        for ip in range(len(interps)):
            f.write(str(ip) + "\t")
            for pat in range(num_classes-1):
                f.write(str(np.mean(collect_over_patterns_1[pat], axis=1)[ip]) + "\t" + str(np.std(collect_over_patterns_1[pat], axis=1)[ip]) + "\t")
            f.write("\n")

    with open("indiv-pattern-2_" + current_c + ".txt", 'w') as f:
        f.write("ip\tmeanp1\tstdp1\tmeanp2\tstdp2\tmeanp3\tstdp3\tmeanp4\tstdp4\tmeanp5\tstdp5\n")
        for ip in range(len(interps)):
            f.write(str(ip) + "\t")            
            for pat in range(num_classes-1):
                f.write(str(np.mean(collect_over_patterns_2[pat], axis=1)[ip]) + "\t" + str(np.std(collect_over_patterns_2[pat], axis=1)[ip]) + "\t")
            f.write("\n")

    """
    
    # "other" patterns compared to pattern 0, averaging over all patterns

    collect_over_patterns = np.zeros((num_classes-1, len(interps), num_runs))
    for r in range(5):
        all_error_trajs = np.concatenate(attractor_distance_new_error_results_1[r]).reshape((len(interps),-1))[:,0:num_classes-1]
        for pat in range(num_classes-1):
            collect_over_patterns[pat][:,r] = all_error_trajs[:,pat]

    mean_error_per_ip_1 = np.zeros((len(interps),))
    std_error_per_ip_1 = np.zeros((len(interps),))
    for ip in range(len(interps)):
        all_errors = [x[ip,:] for x in collect_over_patterns]
        mean_error_per_ip_1[ip] = np.mean(all_errors)
        std_error_per_ip_1[ip] = np.std(all_errors)

    if c==0:
        ax1.errorbar(interps, mean_error_per_ip_1, std_error_per_ip_1, color=colors[0])
    if c==1:
        ax2.errorbar(interps, mean_error_per_ip_1, std_error_per_ip_1, color=colors[0])
    if c==2:
        ax3.errorbar(interps, mean_error_per_ip_1, std_error_per_ip_1, color=colors[0])

    collect_over_patterns = np.zeros((num_classes-1, len(interps), num_runs))
    for r in range(5):
        all_error_trajs = np.concatenate(attractor_distance_new_error_results_2[r]).reshape((len(interps),-1))[:,0:num_classes-1]
        for pat in range(num_classes-1):
            collect_over_patterns[pat][:,r] = all_error_trajs[:,pat]

    mean_error_per_ip_2 = np.zeros((len(interps),))
    std_error_per_ip_2 = np.zeros((len(interps),))
    for ip in range(len(interps)):
        all_errors = [x[ip,:] for x in collect_over_patterns]
        mean_error_per_ip_2[ip] = np.mean(all_errors)
        std_error_per_ip_2[ip] = np.std(all_errors)

    if c==0:
        ax1.errorbar(interps, mean_error_per_ip_2, std_error_per_ip_2, color=colors[1])
    if c==1:
        ax2.errorbar(interps, mean_error_per_ip_2, std_error_per_ip_2, color=colors[1])
    if c==2:
        ax3.errorbar(interps, mean_error_per_ip_2, std_error_per_ip_2, color=colors[1])

    with open("error-1_" + current_c + ".txt", 'w') as f:
        f.write("ip\tmean\tstd\n")
        for ip in range(len(interps)):
            f.write(str(ip) + "\t" + str(mean_error_per_ip_1[ip]) + "\t" + str(std_error_per_ip_1[ip]) + "\n")
    with open("error-2_" + current_c + ".txt", 'w') as f:
        f.write("ip\tmean\tstd\n")
        for ip in range(len(interps)):
            f.write(str(ip) + "\t" + str(mean_error_per_ip_2[ip]) + "\t" + str(std_error_per_ip_2[ip]) + "\n")

    """

    c += 1


ax1.set_xlabel("interpolation factor (1: face, 0: other class)")
ax2.set_xlabel("interpolation factor (1: face, 0: other class)")
ax3.set_xlabel("interpolation factor (1: face, 0: other class)")
ax1.set_ylabel("Hyper-prior (0.001)")
ax2.set_ylabel("Normal prior (1)")
ax3.set_ylabel("Hypo-prior (1000)")
ax1.set_ylim([0, 0.2])
ax2.set_ylim([0, 0.2])
ax3.set_ylim([0, 0.2])
plt.savefig('bla.pdf')#(os.path.join(eval_dir, '../bla.png'))
plt.close()






"""
# mean over all pattern1-pattern2 combinations
over_all_vis_mean_1 = np.zeros((len(interps),))
over_all_vis_std_1 = np.zeros((len(interps),))
over_all_vis_mean_2 = np.zeros((len(interps),))
over_all_vis_std_2 = np.zeros((len(interps),))
for ip in range(len(interps)):
    over_all_vis_mean_1[ip] = np.mean(np.concatenate([x[ip] for x in attractor_distance_vis_error_results_1]))
    over_all_vis_std_1[ip] = np.std(np.concatenate([x[ip] for x in attractor_distance_vis_error_results_1]))
    over_all_vis_mean_2[ip] = np.mean(np.concatenate([x[ip] for x in attractor_distance_vis_error_results_2]))
    over_all_vis_std_2[ip] = np.std(np.concatenate([x[ip] for x in attractor_distance_vis_error_results_2]))

ax1.errorbar(interps, over_all_vis_mean_2, yerr = over_all_vis_std_2, color=colors[c], label=str(current_c) + " (1)")
ebplot = ax1.errorbar(interps, over_all_vis_mean_1, yerr = over_all_vis_std_1, color=colors[c], label=str(current_c) + " (2)")
ebplot[-1][0].set_linestyle('-')

over_all_new_mean_1 = np.zeros((len(interps),))
over_all_new_std_1 = np.zeros((len(interps),))
over_all_new_mean_2 = np.zeros((len(interps),))
over_all_new_std_2 = np.zeros((len(interps),))
for ip in range(len(interps)):
    over_all_new_mean_1[ip] = np.mean(np.concatenate([x[ip] for x in attractor_distance_new_error_results_1]))
    over_all_new_std_1[ip] = np.std(np.concatenate([x[ip] for x in attractor_distance_new_error_results_1]))
    over_all_new_mean_2[ip] = np.mean(np.concatenate([x[ip] for x in attractor_distance_new_error_results_2]))
    over_all_new_std_2[ip] = np.std(np.concatenate([x[ip] for x in attractor_distance_new_error_results_2]))

ax2.errorbar(interps, over_all_new_mean_2, yerr = over_all_new_std_2, color=colors[c], label=str(current_c) + " (1)")
ebplot = ax2.errorbar(interps, over_all_new_mean_1, yerr = over_all_new_std_1, color=colors[c], linestyle='-', label=str(current_c) + " (2)")
ebplot[-1][0].set_linestyle('-')

c += 1
"""


# for r in range(attractor_distance_vis_error_results.shape[0]):
#     ax.errorbar(interps, [np.mean(x) for x in attractor_distance_vis_error_results[r]], yerr=[np.var(x) for x in attractor_distance_vis_error_results[r]])
# ax.set_ylim([0, 0.05])
# plt.savefig(os.path.join(eval_dir, 'hyp-' + str(test_hyp_priors[current_c]) + '_visible.pdf'))
# plt.close()
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# # plot all runs in the same plot
# for r in range(attractor_distance_new_error_results.shape[0]):
#     ax.errorbar(interps, [np.mean(x) for x in attractor_distance_new_error_results[r]], yerr=[np.var(x) for x in attractor_distance_new_error_results[r]])
# ax.set_ylim([0, 0.05])
# plt.savefig(os.path.join(eval_dir, 'hyp-' + str(test_hyp_priors[current_c]) + '_new-part.pdf'))
# plt.close()

# collapsed over all runs:



