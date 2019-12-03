import os
import numpy as np

train_hyp_all = ['0.001', '0.01', '0.1', '1', '10', '100', '1000']
max_epoch = 30000
data_set_name = '2019-11-all'
eval_head_dir = './results/training/' + data_set_name

error_training_proactive = np.empty((len(train_hyp_all),), dtype=object)

dir_list = next(os.walk(eval_head_dir))[1]
for train_hyp in train_hyp_all:
    error_training_proactive[train_hyp_all.index(train_hyp)] = np.zeros((len(dir_list),np.int(max_epoch/100)+2))+1000 # initially set to some high value

    for curr_r in range(len(dir_list)):
        current_run = os.path.join(eval_head_dir, dir_list[curr_r])
        current_net = os.path.join(current_run, train_hyp)
        print("Evaluate for " + str(current_net))

        current_error = np.load(os.path.join(current_net, 'history_generation_error_proactive.npy'))

        mean_error_over_time = np.mean(np.concatenate(current_error, axis=0).reshape((6,-1)),axis=0)

        error_training_proactive[train_hyp_all.index(train_hyp)][curr_r,:len(mean_error_over_time)] = mean_error_over_time

        # find the best epoch and set everything afterwards to that error
        min_idx = np.argmin(error_training_proactive[train_hyp_all.index(train_hyp)][curr_r,:])
        error_training_proactive[train_hyp_all.index(train_hyp)][curr_r,min_idx+1:] = error_training_proactive[train_hyp_all.index(train_hyp)][curr_r,min_idx]

all_error_mean = np.zeros((len(train_hyp_all), np.int(max_epoch/100)+2))
all_error_std = np.zeros((len(train_hyp_all), np.int(max_epoch/100)+2))

for train_hyp in train_hyp_all:
    all_error_mean[train_hyp_all.index(train_hyp)] = np.mean(error_training_proactive[train_hyp_all.index(train_hyp)],axis=0)
    all_error_std[train_hyp_all.index(train_hyp)] = np.std(error_training_proactive[train_hyp_all.index(train_hyp)],axis=0)


    # write results to file
    with open(os.path.join(eval_head_dir, "training-error-" + str(train_hyp) + ".txt"), 'w') as f:
        f.write("ep\tmean\tstd\n")
        for ep in range(np.int(max_epoch/100)+2):
            f.write(str(ep) + "\t" + str(all_error_mean[train_hyp_all.index(train_hyp),ep]) + "\t" + str(all_error_std[train_hyp_all.index(train_hyp),ep]) + "\n")



