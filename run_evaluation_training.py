import os
import numpy as np
from nets import load_network

# which training parameter conditions to test
train_hyp_all = ['0.001', '0.01', '0.1', '1', '10', '100', '1000']

# maximum epoch used in training
max_epoch = 30000
#data_set_name = 'example'
data_set_name = "final_0.01-100_6x7"#"training-2020-03_noise0.01"
eval_head_dir = './results/training/' + data_set_name

error_training_proactive = np.empty((len(train_hyp_all),), dtype=object)
error_training_reactive = np.empty((len(train_hyp_all),), dtype=object)
variance_estimation = np.empty((len(train_hyp_all),), dtype=object)

dir_list = next(os.walk(eval_head_dir))[1]
for train_hyp in train_hyp_all:
    error_training_proactive[train_hyp_all.index(train_hyp)] = np.zeros((len(dir_list),np.int(max_epoch/100)+2))+1000 # initially set to some high value
    error_training_reactive[train_hyp_all.index(train_hyp)] = np.zeros((len(dir_list),np.int(max_epoch/100)+2))+1000 # initially set to some high value
    variance_estimation[train_hyp_all.index(train_hyp)] = np.zeros((len(dir_list),np.int(max_epoch)+2))+1000 # initially set to some high value

    for curr_r in range(len(dir_list)):
        current_run = os.path.join(eval_head_dir, dir_list[curr_r])
        current_net = os.path.join(current_run, train_hyp)
        print("Evaluate for " + str(current_net))

        current_error_pro = np.load(os.path.join(current_net, 'history_generation_error_proactive.npy'))
        current_error_re = np.load(os.path.join(current_net, 'history_generation_error_reactive.npy'))
        current_var_est = np.load(os.path.join(current_net, 'history_training_variance_estimation.npy'))

        mean_error_over_time_pro = np.mean(np.concatenate(current_error_pro, axis=0).reshape((6,-1)),axis=0)
        mean_error_over_time_re = np.mean(np.concatenate(current_error_re, axis=0).reshape((6,-1)),axis=0)

        error_training_proactive[train_hyp_all.index(train_hyp)][curr_r,:len(mean_error_over_time_pro)] = mean_error_over_time_pro
        error_training_reactive[train_hyp_all.index(train_hyp)][curr_r,:len(mean_error_over_time_re)] = mean_error_over_time_re
        variance_estimation[train_hyp_all.index(train_hyp)][curr_r,:len(current_var_est)-1] = current_var_est[1:]

        # find the best epoch and set everything afterwards to that error
        min_idx = np.argmin(error_training_proactive[train_hyp_all.index(train_hyp)][curr_r,:])
        error_training_proactive[train_hyp_all.index(train_hyp)][curr_r,min_idx+1:] = error_training_proactive[train_hyp_all.index(train_hyp)][curr_r,min_idx]

        min_idx = np.argmin(error_training_reactive[train_hyp_all.index(train_hyp)][curr_r,:])
        error_training_reactive[train_hyp_all.index(train_hyp)][curr_r,min_idx+1:] = error_training_reactive[train_hyp_all.index(train_hyp)][curr_r,min_idx]

        min_idx = np.argmin(variance_estimation[train_hyp_all.index(train_hyp)][curr_r,:])
        variance_estimation[train_hyp_all.index(train_hyp)][curr_r,min_idx+1:] = variance_estimation[train_hyp_all.index(train_hyp)][curr_r,min_idx]


all_error_pro_mean = np.zeros((len(train_hyp_all), np.int(max_epoch/100)+2))
all_error_pro_std = np.zeros((len(train_hyp_all), np.int(max_epoch/100)+2))
all_error_re_mean = np.zeros((len(train_hyp_all), np.int(max_epoch/100)+2))
all_error_re_std = np.zeros((len(train_hyp_all), np.int(max_epoch/100)+2))
all_varest_mean = np.zeros((len(train_hyp_all), np.int(max_epoch)+2))
all_varest_std = np.zeros((len(train_hyp_all), np.int(max_epoch)+2))

for train_hyp in train_hyp_all:
    all_error_pro_mean[train_hyp_all.index(train_hyp)] = np.mean(error_training_proactive[train_hyp_all.index(train_hyp)],axis=0)
    all_error_pro_std[train_hyp_all.index(train_hyp)] = np.std(error_training_proactive[train_hyp_all.index(train_hyp)],axis=0)
    all_error_re_mean[train_hyp_all.index(train_hyp)] = np.mean(error_training_reactive[train_hyp_all.index(train_hyp)],axis=0)
    all_error_re_std[train_hyp_all.index(train_hyp)] = np.std(error_training_reactive[train_hyp_all.index(train_hyp)],axis=0)
    all_varest_mean[train_hyp_all.index(train_hyp)] = np.mean(variance_estimation[train_hyp_all.index(train_hyp)],axis=0)
    all_varest_std[train_hyp_all.index(train_hyp)] = np.std(variance_estimation[train_hyp_all.index(train_hyp)],axis=0)

    # write results to file
    with open(os.path.join(eval_head_dir, "training-error-proactive-" + str(train_hyp) + ".txt"), 'w') as f:
        f.write("ep\tmean\tstd\n")
        for ep in range(np.int(max_epoch/100)+2):
            f.write(str(ep) + "\t" + str(all_error_pro_mean[train_hyp_all.index(train_hyp),ep]) + "\t" + str(all_error_pro_std[train_hyp_all.index(train_hyp),ep]) + "\n")

    # write results to file
    with open(os.path.join(eval_head_dir, "training-error-reactive-" + str(train_hyp) + ".txt"), 'w') as f:
        f.write("ep\tmean\tstd\n")
        for ep in range(np.int(max_epoch/100)+2):
            f.write(str(ep) + "\t" + str(all_error_re_mean[train_hyp_all.index(train_hyp),ep]) + "\t" + str(all_error_re_std[train_hyp_all.index(train_hyp),ep]) + "\n")

    var_eval_points = [0,1,10,100,1000,10000,30000]
    # write results to file
    with open(os.path.join(eval_head_dir, "variance-estimation-" + str(train_hyp) + ".txt"), 'w') as f:
        f.write("ep\tmean\tstd\n")
        for ep in range(len(var_eval_points)):#range(np.int(max_epoch)+2):
            f.write(str(var_eval_points[ep]) + "\t" + str(all_varest_mean[train_hyp_all.index(train_hyp),var_eval_points[ep]]) + "\t" + str(all_varest_std[train_hyp_all.index(train_hyp),var_eval_points[ep]]) + "\n")




# generate and plot the learned trajectories, using the best epoch
import matplotlib.pyplot as plt
dir_list = next(os.walk(eval_head_dir))[1]
for train_hyp in train_hyp_all:

    # plot all results for one H
    plt.figure()
    fig = plt.figure('Proactive results', figsize=(30, 60.0))
    plt.rcParams.update({'font.size': 40, 'legend.fontsize': 40})

    for curr_r in range(len(dir_list)):

        current_run = os.path.join(eval_head_dir, dir_list[curr_r])
        current_net = os.path.join(current_run, train_hyp)

        params, model = load_network(current_net, model_filename="network-final")

        # proactive
        res, resv, resm, u_h_history = model.generate(model.initial_states.W.array, 90, add_variance_to_output = 0, additional_output='activations', external_signal_variance = model.external_signal_variance)

        for i in range(res.shape[0]):
            ax = fig.add_subplot(10,6,curr_r*6+(i+1))
            traj = res[i,:].reshape((90,3))
            ax.plot(traj[:,0], traj[:,1])

    plt.tight_layout()
    plt.savefig(os.path.join(eval_head_dir, 'proactive-results-H-' + str(train_hyp) + ".pdf"))
    plt.close()


