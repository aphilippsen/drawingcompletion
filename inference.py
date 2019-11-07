# general imports
import os
import pathlib
import sys
import time
import datetime
import random

import matplotlib
# to enable plotting on a remote machine, without a running X connection:
if not 'matplotlib.pyplot' in sys.modules:
    matplotlib.use('Agg')
# Plotting
import matplotlib.pyplot as plt

# ML imports:
# for learning
import chainer
from chainer.backends import cuda
from chainer import optimizers, serializers
from chainer.functions.math import exponential
import numpy as np
# for evaluation
#from sklearn.manifold import MDS
#from matplotlib.mlab import PCA

# local imports
from nets import SCTRNN, make_initial_state_zero, make_initial_state_random, NetworkParameterSetting, save_network, load_network
from utils.normalize import normalize, range2norm, norm2range
from utils.visualization import plot_results, plot_pca_activations


def infer_initial_states_sctrnn(params, old_model, testing_data, num_timesteps = 0, epochs = 2000, start_is = 'mean', error_computation = 'standard', single_recognition = False, plot = True, hyp_prior = None, external_signal_variance = -1, x_start = None):

    likelihood_stop = False
    conv_eval_interval = 500

    # each trajectory is handled as a separate "class", infer initial states per class
    num_classes = testing_data.shape[0]
    # full number of timesteps
    num_timesteps_orig = int(testing_data.shape[1]/params.num_io)
    # timesteps to use for inference
    if num_timesteps == 0:
        num_timesteps = num_timesteps_orig

    gpu_id = 0 # -1 for CPU
    # Determine whether CPU or GPU should be used
    xp = np
    if gpu_id >= 0 and cuda.available:
        print("Use GPU!")
        cuda.get_device_from_id(gpu_id).use()
        xp = cuda.cupy
    else:
        print("Use CPU!")
        gpu_id = -1

    c = []
    num_samples_per_class = 1
    for i in range(num_classes):
        for j in range(num_samples_per_class):
            c.append(i)
    c_train = xp.array(c)

    save_location = "."
    if os.path.exists("/media/AnjaDataDrive"):
        save_location = "/media/AnjaDataDrive"
    save_location += "/results"

    now = datetime.datetime.now()
    expStr = str(now.year).zfill(4) + "-" + str(now.month).zfill(2) + "-" + str(now.day).zfill(2) + "_" + str(now.hour).zfill(2) + "-" + str(now.minute).zfill(2) + "_" + str(now.microsecond).zfill(7) + "_inference"
    save_dir = os.path.join(save_location, expStr)
    if plot:
        print(save_dir)

    if plot == True:
        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

    save_interval = 100            # interval for testing the production capability of the network and saving initial state information
    save_model_interval = 100      # interval for storing the learned model

    # Should better already be done outside this method
    # try:
    #     x_train = range2norm(x_train_orig, params.norm_offset, params.norm_range, minmax = params.minmax)
    #     x_train = xp.float32(x_train)
    #     # N = len(x_train)
    # except:
    #     print("No normalization applicable...")
    #     x_train = testing_data


    # CUT PART OF THE TRAINING SIGNAL (COMPLETION TASK)
    testing_data_cut = testing_data[:,0:params.num_io*num_timesteps]

    if plot == True:
        plot_results(xp.copy(testing_data_cut[0::num_samples_per_class]), num_timesteps, os.path.join(save_dir, 'target_trajectories.png'), params.num_io, twoDim = True)

        info = "same trajectories (original #timesteps: " + str(num_timesteps_orig) + "), used #timesteps: " + str(num_timesteps)

    # copy network model and prepare it for backpropagation inference
    params.learn_weights = False
    params.learn_bias = False
    params.epochs = epochs

    model = SCTRNN(params.num_io, params.num_c, params.tau_c, num_classes, init_state_init = params.init_state_init, init_state_learning = params.learn_init_states, weights_learning = params.learn_weights, bias_learning = params.learn_bias, tau_learning = params.learn_tau, external_contrib = params.training_external_contrib, aberrant_sensory_precision = params.aberrant_sensory_precision, pretrained_model = old_model)
    #model.hyp_prior = params.hyp_prior
    #model.external_signal_variance = params.external_signal_variance
    if not hyp_prior is None:
        model.hyp_prior = hyp_prior
        params.hyp_prior = hyp_prior
    if external_signal_variance is None or external_signal_variance >= 0:
        model.external_signal_variance = external_signal_variance
        params.external_signal_variance = external_signal_variance
    params.lr = 0.01

    with open(os.path.join(save_dir,"info.txt"),'w') as f:
        f.write(params.print_parameters())
        f.write("\n")
        f.write(info)
        f.write("\n")
    f.close()

    if plot:
        print("Inference: model with ext.contrib: " + str(model.external_contrib))
    if start_is is 'mean':
        model.set_initial_states_mean()
    elif start_is is 'zero':
        model.set_initial_states_zero()
    else:
        model.initial_states.W.array = start_is
    #model.apply_estimated_variance = True
    model.set_init_state_learning(c_train)

    if gpu_id >= 0:
        model.to_gpu(gpu_id)
        testing_data = cuda.to_gpu(testing_data)
        x_start = cuda.to_gpu(x_start)

    # import pdb; pdb.set_trace()


    if plot == True:
        save_network(save_dir, params=params, model = model, model_filename = "network-initial")

    # Optimizer
    optimizer = optimizers.Adam(params.lr)
    optimizer.setup(model)
    #optimizer.add_hook(chainer.optimizer.WeightDecay(0))

    history_init_state_var = np.zeros((epochs+1,))
    history_init_state_var[0] = np.mean(np.var(model.initial_states.W.array,axis=0))
    history_generation_error_proactive = np.empty((num_classes,), dtype=object)
    history_generation_error_reactive = np.empty((num_classes,), dtype=object)
    history_training_error = np.zeros((epochs+1,))
    history_training_variance_estimation = np.zeros((epochs+1, num_classes))

    history_initial_states = []

    likelihood_per_epoch = []

    if plot:
        print("actual variance of init_states_0: " + str(history_init_state_var[0]))

    # Evaluate the performance of the untrained network
    test_batch_size = np.min([model.initial_states.W.array.shape[0], testing_data.shape[0]])
    res, resv, resm = model.generate(model.initial_states.W.array, num_timesteps_orig, external_contrib = 0, epsilon_disturbance = 0, x_start = x_start)
    results = res#cuda.to_cpu(res)

    for i in range(num_classes):
        generation_error = chainer.functions.mean_squared_error(results[i,:],testing_data[i,:]).array.tolist()
        history_generation_error_proactive[i] = [generation_error]

        if plot == True:
            with open(os.path.join(save_dir,"evaluation.txt"),'a') as f:
                f.write("before learning: pattern generation error (proactive): " + str(history_generation_error_proactive[i]) + "\n")

    if plot == True:
        plot_results(xp.copy(results), num_timesteps_orig, os.path.join(save_dir, "proactive_before-learning"), params.num_io, twoDim = True)

    res, resv, resm, pe, wpe, respost = model.generate(model.initial_states.W.array, num_timesteps_orig, external_contrib = 1, external_input = xp.asarray(testing_data[0::num_samples_per_class,:]), epsilon_disturbance = 0, x_start = x_start)
    results = res#cuda.to_cpu(res)

    for i in range(num_classes):
        generation_error = chainer.functions.mean_squared_error(results[i,:],testing_data[i,:]).array.tolist()
        history_generation_error_reactive[i] = [generation_error]

        if plot == True:
            with open(os.path.join(save_dir, "evaluation.txt"),'a') as f:
                f.write("before learning: pattern generation error (reactive): " + str(history_generation_error_reactive[i]) + "\n")

    if plot == True:
        plot_results(xp.copy(results), num_timesteps_orig, os.path.join(save_dir, "reactive_before-learning"), params.num_io, twoDim = True)

    # arrays for tracking likelihood and determining stop condition
    all_mean_diffs = []
    all_std_diffs = []
    m1s = []
    s1s = []
    # tmp_epoch_marker = 0
    # conv_eval_interval = 1000 # the length of the interval to consider for determining convergence

    for epoch in range(1, params.epochs + 1):
        epochStart = time.time()

        outv = np.zeros((num_timesteps,))

        # permutate samples in each epoch so that they are randomly ordered
        perm = np.random.permutation(testing_data_cut.shape[0])

        # here, one batch equals the full training set
        x_batch = xp.asarray(testing_data_cut[perm])
        x_batch = x_batch + 0.01 * xp.random.randn(x_batch.shape[0], x_batch.shape[1]).astype('float32')
        model.set_init_state_learning(c_train[perm])

        mean_init_states = chainer.Variable(xp.zeros((),dtype=xp.float32))
        mean_init_states = chainer.functions.average(model.initial_states.W,axis=0) #keepdims=True
        #mean_init_states = xp.mean(c0.array,axis=0) # using this instead causes no difference in resulting gradient of c0

        # initialize error
        acc_loss = chainer.Variable(xp.zeros((),dtype=xp.float32)) # for weight backprop
        acc_init_loss = chainer.Variable(xp.zeros((),dtype=xp.float32)) # for init states backprop
        err = xp.zeros(()) # for evaluation only

        # clear gradients from previous batch
        model.cleargrads()
        # clear output and variance estimations from previous batch
        model.reset_current_output()

        t=0 # iterate through time
        x_t = x_batch[:, params.num_io*t:params.num_io*(t+1)]
        # next time step to be predicted (for evaluation)
        x_t1 = x_batch[:, params.num_io*(t+1):params.num_io*(t+2)]
        # x_t = xp.reshape(x_batch[0][t,:], (1, params.num_io))
        # x_t1 = xp.reshape(x_batch[0][t+1,:], (1, params.num_io))
        # for i in range(1, params.batch_size):
        #     x_t = np.concatenate((x_t, xp.reshape(x_batch[i][t,:], (1,params.num_io))),axis=0)
        #     x_t1 = np.concatenate((x_t1, xp.reshape(x_batch[i][t+1,:], (1,params.num_io))),axis=0)

        # execute first forward step
        u_h, y, v = model(x_t, None) # initial states of u_h are set automatically according to model.classes

        # noisy output estimation
        #y_out = y.array + xp.sqrt(v.array) * xp.random.randn()

        # compute prediction error, averaged over minibatch
        #loss_i = chainer.functions.mean_squared_error(y, chainer.Variable(x_batch[0][t+1:t+2]))
        if error_computation == 'standard':
            loss_i = chainer.functions.gaussian_nll(chainer.Variable(x_t1), y, exponential.log(v))
        elif error_computation == 'integrated':
            integrated_x = params.training_external_contrib * chainer.Variable(x_t1) + (1 - params.training_external_contrib) * (y + chainer.functions.sqrt(v) * xp.random.randn())
            loss_i = chainer.functions.gaussian_nll(integrated_x, y, exponential.log(v))
        acc_loss += loss_i


        acc_loss += loss_i

        # compute error for evaluation purposes
        err += chainer.functions.mean_squared_error(chainer.Variable(x_t), y).array.reshape(()) * params.batch_size

        outv[t] = xp.mean(v.array)

        # rollout trajectory
        for t in range(1,num_timesteps-1):
            # current time step
            x_t = x_batch[:, params.num_io*t:params.num_io*(t+1)]
            # next time step to be predicted (for evaluation)
            x_t1 = x_batch[:, params.num_io*(t+1):params.num_io*(t+2)]

            u_h, y, v = model(x_t, u_h)

            # noisy output estimation
            #y_out = y.array + xp.sqrt(v.array) * xp.random.randn()

            # compute error for backprop for weights
            if error_computation == 'standard':
                loss_i = chainer.functions.gaussian_nll(chainer.Variable(x_t1), y, exponential.log(v))
            elif error_computation == 'integrated':
                integrated_x = params.training_external_contrib * chainer.Variable(x_t1) + (1 - params.training_external_contrib) * (y + chainer.functions.sqrt(v) * xp.random.randn())
                loss_i = chainer.functions.gaussian_nll(integrated_x, y, exponential.log(v))
            acc_loss += loss_i

            # compute error for evaluation purposes
            err += chainer.functions.mean_squared_error(chainer.Variable(x_t), y).array.reshape(()) * params.batch_size

            outv[t] = xp.mean(v.array)

        # for each training sequence of this batch: compute loss for maintaining desired initial state variance
        if not single_recognition:
            for s in range(len(c_train)):
                if gpu_id >= 0:
                    # acc_init_loss += chainer.functions.gaussian_nll(model.initial_states()[model.classes][s], mean_init_states, xp.float32(xp.ones(mean_init_states.shape) * exponential.log(cuda.to_gpu(params.init_state_var, device=gpu_id)).array))
                    acc_init_loss += chainer.functions.gaussian_nll(model.initial_states()[model.classes][s], mean_init_states, xp.ones(mean_init_states.shape) * exponential.log(cuda.to_gpu(params.init_state_var, device=gpu_id)))
                else:
                    acc_init_loss += chainer.functions.gaussian_nll(model.initial_states()[model.classes][s], mean_init_states, exponential.log(params.init_state_var))

            # compute gradients
            # (gradients from L_out and L_init are summed up)
            # gradient of initial states equals:
            # 1/params.init_state_var * (c0[cl]-mean_init_states).array

            epochBatchProcessed = time.time()
            if plot:
                print("Elapsed time (batch processing): " + str(epochBatchProcessed - epochStart))

        #        if params.learn_init_states:
            acc_init_loss *= params.init_state_loss_scaling
            acc_init_loss.backward()
        else:
            epochBatchProcessed = time.time()
            if plot:
                print("Elapsed time (batch processing): " + str(epochBatchProcessed - epochStart))

        #print("acc_init_loss.backward initial_states gradient: " + str(model.initial_states.W.grad[0][0]) + ", " + str(model.initial_states.W.grad[1][0]))
        acc_loss.backward()

        if plot:
            print("update")
        optimizer.update()

        # printing and testing
        if plot:
            print("Done epoch " + str(epoch))
            print("Elapsed time (error computation): " + str(time.time() - epochBatchProcessed))
        # print("Tau: " + str(model.tau_c.W.array[0]))
        error = err/params.batch_size/num_timesteps
        #minVar_t = np.mean([np.min(np.mean(np.sqrt(outv[b][50:-1]),axis=1)) for b in range(params.batch_size)])
        #maxVar_t = np.mean([np.max(np.mean(np.sqrt(outv[b][50:-1]),axis=1)) for b in range(params.batch_size)])
        mean_estimated_var = xp.mean(outv)
        history_training_error[epoch] = error
        history_training_variance_estimation[epoch,:] = mean_estimated_var

        if plot:
            print("train MSE = " + str(error) + "\nmean estimated var: " + str(mean_estimated_var))
            print("init_states = [" + str(model.initial_states.W.array[0][0]) + "," + str(model.initial_states.W.array[0][1]) + "...], var: " + str(np.mean(np.var(model.initial_states.W.array,axis=0))) + ", accs: " + str(acc_loss) + " + " + str(acc_init_loss))

        likelihood_per_epoch.append(np.float64(acc_loss.array+acc_init_loss.array))

        history_init_state_var[epoch] = np.mean(np.var(model.initial_states.W.array,axis=0))

        if plot == True:
            with open(os.path.join(save_dir,"evaluation.txt"),'a') as f:
                f.write("epoch: " + str(epoch)+ "\n")
                f.write("train MSE = " + str(error) + "\nmean estimated var: " + str(mean_estimated_var))
                f.write("initial state var: " + str(history_init_state_var[epoch]) + ", precision loss: " + str(acc_loss) + ", variance loss: " + str(acc_init_loss) + "\ninit states:\n")
                for i in range(num_classes):
                # f.write('init_states = [' + str(model.initial_states.W[0][0]) + ',' + str(model.initial_states.W[0][1]) + '...], [' + str(model.initial_states.W[1][0]) + ',' + str(model.initial_states.W[1][1]) + '...], var: ' + str(np.mean(np.var(model.initial_states.W.array,axis=0))) + ', accs: ' + str(acc_loss) + " + " + str(acc_init_loss)+'\n')
                    f.write("\t[" + str(model.initial_states.W[i][0]) + "," + str(model.initial_states.W[i][1]) + "...]\n")
            f.close()

        if epoch%save_interval == 1 or epoch == params.epochs:
            # evaluate proactive generation

            res, resv, resm, u_h_history = model.generate(model.initial_states.W.array, num_timesteps_orig, external_contrib = 0, epsilon_disturbance = 0, additional_output='activations', x_start = x_start)
            results = res #cuda.to_cpu(res)

            if plot == True:
                plot_results(xp.copy(results), num_timesteps_orig, os.path.join(save_dir, "proactive_epoch-" + str(epoch).zfill(len(str(epochs)))), params.num_io, twoDim = True)

                for i in range(num_classes):
                    generation_error = chainer.functions.mean_squared_error(results[i,:], testing_data[i,:]).array.tolist()
                    history_generation_error_proactive[i].append(generation_error)
                    if plot == True:
                        with open(os.path.join(save_dir, "evaluation.txt"), 'a') as f:
                            f.write("pattern generation error (proactive): " + str(generation_error) + "\n")
                        f.close()

            # evaluate reactive generation
            res, resv, resm, pe, wpe, u_h_history, respost = model.generate(model.initial_states.W.array, num_timesteps_orig, external_contrib = 1, external_input = xp.asarray(testing_data[0::num_samples_per_class,:]), additional_output='activations', x_start = x_start)
            results = res#cuda.to_cpu(res)

            if plot == True:
                plot_results(xp.copy(results), num_timesteps_orig, os.path.join(save_dir, "reactive_epoch-" + str(epoch).zfill(len(str(epochs)))), params.num_io, twoDim = True)

            for i in range(test_batch_size):
                generation_error = chainer.functions.mean_squared_error(results[i,:], testing_data[i,:]).array.tolist()
                history_generation_error_reactive[i].append(generation_error)
                if plot == True:
                    with open(os.path.join(save_dir, "evaluation.txt"), 'a') as f:
                        f.write("pattern generation error (reactive): " + str(generation_error) + "\n")
                    f.close()
        if plot == True:
            if epoch%save_model_interval == 1 or epoch == params.epochs:
                save_network(save_dir, params, model, model_filename="network-epoch-"+str(epoch).zfill(len(str(epochs))))
                np.save(os.path.join(save_dir,"history_init_state_var"), np.array(history_init_state_var))
                np.save(os.path.join(save_dir,"history_generation_error_proactive"), np.array(history_generation_error_proactive))
                np.save(os.path.join(save_dir,"history_generation_error_reactive"), np.array(history_generation_error_reactive))
                np.save(os.path.join(save_dir,"history_training_error"), np.array(history_training_error))
                np.save(os.path.join(save_dir, "history_training_variance_estimation"), np.array(history_training_variance_estimation))

                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.plot(np.arange(0,len(history_init_state_var)), history_init_state_var)
                plt.title("init state variance")
                fig.savefig(os.path.join(save_dir,"init-state-var"))
                plt.close()

                fig = plt.figure()
                ax = fig.add_subplot(121)
                for i in range(num_classes):
                    ax.plot(np.arange(0,len(history_generation_error_proactive[i]))*save_interval, history_generation_error_proactive[i])
                ax = fig.add_subplot(122)
                for i in range(num_classes):
                    ax.plot(np.arange(0,len(history_generation_error_reactive[i]))*save_interval, history_generation_error_reactive[i], label=str(i))
                plt.title("generation error (proactive / reactive)")
                plt.legend()
                fig.savefig(os.path.join(save_dir,"generation-error"))
                plt.close()

                plt.figure()
                plt.plot(np.arange(len(all_std_diffs)),all_std_diffs, 'bo',label='std diff')
                plt.plot(np.arange(len(all_mean_diffs)),all_mean_diffs, 'ro',label='mean diff')
                plt.legend()
                plt.savefig(os.path.join(save_dir, 'convergence-condition.png'))
                plt.close()

        history_initial_states.append(model.initial_states.W.array.copy())

        # all_mean_diffs = []
        # all_std_diffs = []
        # m1s = []
        # s1s = []
        # tmp_epoch_marker = 0
        # conv_eval_interval = 500
        # mean_threshold = 1e-3
        # std_threshold = 5e-2
        # for i in range(conv_eval_interval+1,epoch):
        #     m1 = np.mean(likelihood_per_epoch[i-conv_eval_interval-1:i])
        #     s1 = np.std(likelihood_per_epoch[i-conv_eval_interval-1:i])
        #     m1s.append(m1)
        #     s1s.append(s1)
        #     if i > conv_eval_interval*2:
        #         # relative decrease from previous to this timestep (<0 = no improvement)
        #         mean_diff = (m1s[len(m1s)-conv_eval_interval] - m1s[len(m1s)-1]) / np.max([np.abs(m1s[len(m1s)-conv_eval_interval]), np.abs(m1s[len(m1s)-1])])
        #         # absolute variance decrease in comparison to first time step (=> converges against 1 which means that variance is zero)
        #         # std_diff = (s1s[0]-s1) / np.abs(s1s[0])
        #         std_diff = (s1s[len(s1s)-conv_eval_interval] - s1s[len(s1s)-1]) / np.max([np.abs(s1s[len(s1s)-conv_eval_interval]), np.abs(s1s[len(s1s)-1])])
        #         all_mean_diffs.append(mean_diff)
        #         all_std_diffs.append(std_diff)
        #         print(str(i) + ": mean_diff: " + str(mean_diff) + ", std_diff: " + str(std_diff))
        #         if (-mean_threshold < mean_diff < mean_threshold) and (-std_threshold < std_diff < std_threshold):
        #             if tmp_epoch_marker == 0:
        #                 tmp_epoch_marker = i
        #             # put a first tmp_epoch_marker if it is converged, train for 20% more
        #             elif i > tmp_epoch_marker * 1.2:
        #                 break


        if likelihood_stop and epoch > conv_eval_interval:
            m1 = np.mean(likelihood_per_epoch[epoch-conv_eval_interval-1:epoch])
            s1 = np.std(likelihood_per_epoch[epoch-conv_eval_interval-1:epoch])
            m1s.append(m1)
            s1s.append(s1)
            if epoch > conv_eval_interval*2:
                # relative decrease from previous to this timestep (<0 = no improvement)
                mean_diff = (m1s[len(m1s)-conv_eval_interval] - m1s[len(m1s)-1]) / np.max([np.abs(m1s[len(m1s)-2]), np.abs(m1s[len(m1s)-1])])
                #absolute variance decrease in comparison to first time step (=> converges against 1 which means that variance is zero)
                #std_diff = (s1s[0]-s1) / np.abs(s1s[0])
                #std_diff = (s1s[len(s1s)-conv_eval_interval] - s1s[len(s1s)-1]) /
                std_diff = (s1s[len(s1s)-conv_eval_interval] - s1s[len(s1s)-1]) / np.max([np.abs(s1s[len(s1s)-conv_eval_interval]), np.abs(s1s[len(s1s)-1])])

                all_mean_diffs.append(mean_diff)
                all_std_diffs.append(std_diff)
                if plot:
                    print(str(epoch) + ": mean_diff: " + str(mean_diff) + ", std_diff: " + str(std_diff))
                if (-mean_threshold < mean_diff < mean_threshold) and (-std_threshold < std_diff < std_threshold) and (epoch > minimum_epochs): # and (std_diff > 0.95):
                    if tmp_epoch_marker == 0:
                        tmp_epoch_marker = epoch
                    elif epoch > tmp_epoch_marker * 1.2:
                        break


    if plot == True:
        save_network(save_dir, params, model, model_filename = "network-final")

    return model.initial_states, history_initial_states, results, resm, save_dir
