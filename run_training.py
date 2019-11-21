# general imports
import os
import pathlib
import sys
import time
import datetime
import random
from distutils.file_util import copy_file

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
from dtw import dtw
# for evaluation
#from sklearn.manifold import MDS
#from matplotlib.mlab import PCA

# local imports
from nets import SCTRNN, make_initial_state_zero, make_initial_state_random, NetworkParameterSetting, save_network, load_network
from utils.normalize import normalize, range2norm, norm2range
from utils.visualization import plot_results, plot_pca_activations

gpu_id = 0 # -1 for CPU

euclidean_distance = lambda x, y: np.abs(x-y)
def evaluate_generation(target, generation, method='mse'):
    if method == 'mse':
        return chainer.functions.mean_squared_error(results[i,:], x_train_orig[i,:]).data.tolist()
    elif method == 'dtw':
        val = 0
        for i in range(target.shape[1]):
            val += dtw(target[1:,i], generation[:-1,i], dist=euclidean_distance)[0]
        return val

# Determine whether CPU or GPU should be used
xp = np
if gpu_id >= 0 and cuda.available:
   print("Use GPU!")
   #If NVRTCError:
   #$ export CUDA_HOME=/usr/local/cuda-9.1
   #$ export PATH=${CUDA_HOME}/bin:${PATH}
   #$ export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
   cuda.get_device_from_id(gpu_id).use()
   xp = cuda.cupy
else:
   print("Use CPU!")
   gpu_id = -1

# define if position derivatives schould be learned, too
use_derivatives = False

# define parameter values for different trials
# external_contrib_runs = [1]
# runs = len(external_contrib_runs)

# aberrant_sensory_precision_runs = [-8, -4, -2, 0, 2, 4, 8]
# runs = len(aberrant_sensory_precision_runs)

#tau_runs = [1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4]
# runs = len(tau_runs)

#num_c_runs = [1000,900,800,700,600,500,400,300,200,100,70,50,30,10,5] # [200, 100, 70, 50, 30, 10, 5] #[10,30,50,70,100,200]
#runs = len(num_c_runs)

# ini_var_runs = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
# runs = len(ini_var_runs)

#lrate_runs = [1e-7, 1e-6, 1e-5, 5e-5, 1e-4, 1e-3, 0.01, 0.05]
#runs = len(lrate_runs)

# conn_runs = [70, 60, 50, 40, 30, 20, 10, 5, 1]
# runs = len(conn_runs)

#var_int_runs = [0, 1, 2]
#runs = len(var_int_runs)

# # ext_noise_runs = [0.0001, 0.001, 0.003, 0.005, 0.007, 0.01]
# # runs = len(ext_noise_runs)
# ext_noise_runs = []
#
# # hyp_prior_runs = [0.01, 1, 100]
# hyp_prior_runs = [0.5, 1, 3]
# runs = len(hyp_prior_runs)

# Implicit trajectory noise
# implicit_trajectory_noise_runs = [0.001, 0.001, 0.001]
#implicit_trajectory_noise_runs = [0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005]
#implicit_trajectory_noise_runs = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
#implicit_trajectory_noise_runs = [0, 0, 0, 0, 0, 0, 0, 0, 0]
#implicit_trajectory_noise_runs = [0, 0, 0, 1e-5, 1e-5, 1e-5, 0.0001, 0.0001, 0.0001, 0.001, 0.001, 0.001]
implicit_trajectory_noise_runs = [0]#, 0]#, 0, 0, 0]#,0,0,0,0,0]
# runs = len(implicit_trajectory_noise_runs)

# Explicit sensor noise that has influence on BI
# explicit_sensor_variance_runs = [0.0001, 0.0001, 0.0001]
#explicit_sensor_variance_runs = [1e-4, 1e-4, 1e-4, 0.001, 0.001, 0.001, 0.01, 0.01, 0.01]
#explicit_sensor_variance_runs = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
explicit_sensor_variance_runs = [0.001]#, 0.001]#, 0.001, 0.001, 0.001]#, 0.001, 0.001, 0.001, 0.001, 0.001]
# runs = len(explicit_sensor_variance_runs)

# whether to add the explicit_sensor_variance to the training signal
add_external_signal_variance = True

add_smooth_traj_noise = False

# Hypo prior that influences BI
#hyp_prior_runs =  [0.3, 0.5, 0.9, 1, 2, 5]
hyp_prior_runs = [0.001, 0.01, 0.1, 1, 10, 100, 1000] #[0.01, 1, 50]
# hyp_prior_runs= [0.001]

# If True, define init_weight_dir here, otherwise generate new initial weights
reuse_existing_weights = True

# 1000

# init_weight_dir = "/home/anja/github/drawingcompletion/results/training/2019-11-all/2019-11-05_14-34_0970831/initialWeights"
# ==> /home/anja/github/drawingcompletion/results/training/2019-11-18_15-55_0480074
#init_weight_dir = "/home/anja/github/drawingcompletion/results/training/2019-11-all/2019-11-05_14-35_0321645/initialWeights"
# ==> /home/anja/github/drawingcompletion/results/training/2019-11-18_16-02_0579644
# init_weight_dir = "/home/anja/github/drawingcompletion/results/training/2019-11-all/2019-11-05_14-35_0421490/initialWeights"
# ==> /home/anja/github/drawingcompletion/results/training/2019-11-18_16-02_0911090
#init_weight_dir = "/home/anja/github/drawingcompletion/results/training/2019-11-all/2019-11-05_14-35_0786428/initialWeights"
# ==> /home/anja/github/drawingcompletion/results/training/2019-11-18_16-03_0361625
# init_weight_dir = "/home/anja/github/drawingcompletion/results/training/2019-11-all/2019-11-05_14-36_0715535/initialWeights"
# ==> /home/anja/github/drawingcompletion/results/training/2019-11-18_16-03_0552937
# init_weight_dir = "/home/anja/github/drawingcompletion/results/training/2019-11-all/2019-11-08_15-35_0902631/initialWeights"
# ==> /home/anja/github/drawingcompletion/results/training/2019-11-18_16-04_0152981
# init_weight_dir = "/home/anja/github/drawingcompletion/results/training/2019-11-all/2019-11-08_15-36_0061932/initialWeights"
# ==> /home/anja/github/drawingcompletion/results/training/2019-11-18_16-04_0418418
# init_weight_dir = "/home/anja/github/drawingcompletion/results/training/2019-11-all/2019-11-08_15-36_0110189/initialWeights"
# ==> /home/anja/github/drawingcompletion/results/training/2019-11-18_16-05_0887558
# init_weight_dir = "/home/anja/github/drawingcompletion/results/training/2019-11-all/2019-11-08_15-36_0712878/initialWeights"
# ==> /home/anja/github/drawingcompletion/results/training/2019-11-18_16-05_0096949
# init_weight_dir = "/home/anja/github/drawingcompletion/results/training/2019-11-all/2019-11-08_15-36_0818585/initialWeights"
# ==> /home/anja/github/drawingcompletion/results/training/2019-11-18_16-06_0786867



# 0.001

# init_weight_dir = "/home/anja/github/drawingcompletion/results/training/2019-11-all/2019-11-05_14-34_0970831/initialWeights"
# ==> /home/anja/github/drawingcompletion/results/training/2019-11-18_16-09_0845833
#init_weight_dir = "/home/anja/github/drawingcompletion/results/training/2019-11-all/2019-11-05_14-35_0321645/initialWeights"
# ==> /home/anja/github/drawingcompletion/results/training/2019-11-18_16-10_0991327
# init_weight_dir = "/home/anja/github/drawingcompletion/results/training/2019-11-all/2019-11-05_14-35_0421490/initialWeights"
# ==> /home/anja/github/drawingcompletion/results/training/2019-11-18_16-11_0506551
# init_weight_dir = "/home/anja/github/drawingcompletion/results/training/2019-11-all/2019-11-05_14-35_0786428/initialWeights"
# ==> /home/anja/github/drawingcompletion/results/training/2019-11-18_16-11_0852766
# init_weight_dir = "/home/anja/github/drawingcompletion/results/training/2019-11-all/2019-11-05_14-36_0715535/initialWeights"
# ==> /home/anja/github/drawingcompletion/results/training/2019-11-18_16-11_0307238
# init_weight_dir = "/home/anja/github/drawingcompletion/results/training/2019-11-all/2019-11-08_15-35_0902631/initialWeights"
# ==> /home/anja/github/drawingcompletion/results/training/2019-11-18_16-13_0425077
# init_weight_dir = "/home/anja/github/drawingcompletion/results/training/2019-11-all/2019-11-08_15-36_0061932/initialWeights"
# ==> /home/anja/github/drawingcompletion/results/training/2019-11-18_16-13_0558169
# init_weight_dir = "/home/anja/github/drawingcompletion/results/training/2019-11-all/2019-11-08_15-36_0110189/initialWeights"
# ==> /home/anja/github/drawingcompletion/results/training/2019-11-18_16-14_0034660
# init_weight_dir = "/home/anja/github/drawingcompletion/results/training/2019-11-all/2019-11-08_15-36_0712878/initialWeights"
# ==> /home/anja/github/drawingcompletion/results/training/2019-11-18_16-14_0537299
# init_weight_dir = "/home/anja/github/drawingcompletion/results/training/2019-11-all/2019-11-08_15-36_0818585/initialWeights"
# ==> /home/anja/github/drawingcompletion/results/training/2019-11-18_16-15_0238757





# hyp_prior_runs = [1000.01, 1000.1, 1001, 1010, 1100]

runs = len(hyp_prior_runs)

assert(len(implicit_trajectory_noise_runs) == runs)
assert(len(explicit_sensor_variance_runs) == runs)

# which variance to set for the input in the Bayesian inference
ext_var_proactive = 1

same_weights_per_run = True
same_bias_per_run = True
learn_bias = True

#excitation_bias_runs = [0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1] # [0.00000001, 0.0000001, 0.000001, 0.00001, 0.1, 1, 2, 5, 10] #[0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1]
#runs = len(excitation_bias_runs)
#same_weights_per_run = True
#same_bias_per_run = False
#learn_bias = False

prediction_error_type = 'standard' # 'standard' or 'integrated'
# prediction_error_type = 'integrated'

save_interval = 100            # interval for testing the production capability of the network and saving initial state information
save_model_interval = 100      # interval for storing the learned model
epochs = 30000                  # total maximum number of epochs

likelihood_stop = False        # whether to automatically stop learning depending on the likelihood
minimum_epochs = 20000



# stop when there is no new "best" epoch result for proactive generation within the last X epochs
check_best_improvement_stop = True
minimum_epochs = 10000
patience = 5000 # stop if no improvement since X epochs


experiment_info = "train with 6 classes, init weight dir: " + init_weight_dir

training_data_file = "data/drawing-data-sets/drawings-191105-6-drawings.npy"
training_data_file_classes = "data/drawing-data-sets/drawings-191105-6-drawings-classes.npy"

x_train_orig = np.float32(np.load(training_data_file))

# drawings in data file are in order 0 1 2 0 1 2 0 1 2...

num_classes = 6
num_samples_per_class = 10


save_location = "./results/training/"

now = datetime.datetime.now()
expStr = str(now.year).zfill(4) + "-" + str(now.month).zfill(2) + "-" + str(now.day).zfill(2) + "_" + str(now.hour).zfill(2) + "-" + str(now.minute).zfill(2) + "_" + str(now.microsecond).zfill(7)
save_dir = os.path.join(save_location, expStr)
print(save_dir)
pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

if same_weights_per_run or same_bias_per_run:
    if not reuse_existing_weights:
        init_weight_dir = os.path.join(save_dir, "initialWeights")
        pathlib.Path(init_weight_dir).mkdir(parents=True, exist_ok=True)

for r in range(runs):

    best_epoch_error = 1000 # some high initial error value
    best_epoch = 0

    # num_timesteps = 70
    # traj_noise_parts = 7
    num_timesteps = 90
    traj_noise_parts = 10
    num_io = 3
    # corresponds to a period of num_timesteps/num_repetitions

    #noise_variances=[0.003, 0.003, 0.003, 0.003]


    #Implicit input variance between trajectories
    implicit_trajectory_noise = 0.001
    if len(implicit_trajectory_noise_runs) > 0:
        implicit_trajectory_noise = implicit_trajectory_noise_runs[r]

    # Explicit sensor variance for BI
    explicit_sensor_variance = 0.0001
    if len(explicit_sensor_variance_runs) > 0:
        explicit_sensor_variance = explicit_sensor_variance_runs[r]

    # Hypo prior:  variance added for BI
    if len(hyp_prior_runs) > 0:
        hyp_prior = hyp_prior_runs[r]


    final_save_dir = os.path.join(save_dir, str(hyp_prior))
    pathlib.Path(final_save_dir).mkdir(parents=True, exist_ok=True)


    # x_train = create_lissajous_curves_murata_four_positions(num_classes, num_repetitions, num_timesteps, noise_variances, num_samples_per_class = num_samples_per_class, format = 'array')
    x_train_orig = np.float32(np.load(training_data_file))
    x_train = np.copy(x_train_orig)
    if implicit_trajectory_noise > 0:
        print("Adding implicit trajectory noise")
        trajectory_noise_variances = np.tile([implicit_trajectory_noise], (num_classes*num_samples_per_class,x_train.shape[1]))
        x_train += np.sqrt(trajectory_noise_variances) * np.random.randn(x_train.shape[0], x_train.shape[1])
    if add_smooth_traj_noise:
        print("Adding smooth trajectory noise")
        for i in range(x_train.shape[0]):
            x_train[i,:] += np.reshape(get_trajectory_noise(num_timesteps, int(num_timesteps/traj_noise_parts), dim=num_io), (num_io*num_timesteps,))

    # adding sensor variance
    external_signal_variance_vec = xp.ones((x_train.shape)) * explicit_sensor_variance

    explicit_sensor_variance = np.max([np.mean(np.var(x_train[x:num_classes*num_samples_per_class:num_classes,:],axis=0)) for x in range(num_classes)])
    print("Update explicit sensor variance to actual sensor variance " + str(explicit_sensor_variance))

    if use_derivatives == True:
       # add also velocity and acceleration
       for i in range(0, x_train.shape[0]):

           reshaped_line = np.copy(np.reshape(x_train[i, 0:x_train.shape[1]],
                                              (num_timesteps, int(x_train.shape[1] / num_timesteps))))
           reshaped_line_x = reshaped_line[:, 0]
           reshaped_line_y = reshaped_line[:, 1]

           # for velocity and acceleration
           velocity_x = np.copy(np.reshape(np.gradient(reshaped_line_x), (num_timesteps,)))
           velocity_y = np.copy(np.reshape(np.gradient(reshaped_line_y), (num_timesteps,)))

           acceleration_x = np.reshape(np.gradient(velocity_x), (num_timesteps, 1))
           acceleration_y = np.reshape(np.gradient(velocity_y), (num_timesteps, 1))

           velocity_x = np.reshape(velocity_x, (num_timesteps, 1))
           velocity_y = np.reshape(velocity_y, (num_timesteps, 1))

           velocity = np.concatenate((velocity_x, velocity_y), axis=1)

           acceleration = np.concatenate((acceleration_x, acceleration_y), axis=1)

           # normalize input
           reshaped_line = reshaped_line / np.amax(reshaped_line)
           velocity = velocity / np.amax(velocity)
           acceleration = acceleration / np.amax(acceleration)

           reshaped_line = np.concatenate((reshaped_line, velocity, acceleration), axis=1)

           if i == 0:
               x_train_extended = np.reshape(reshaped_line,
                                             (1, 3 * int(x_train.shape[1] / num_timesteps) * num_timesteps))
           else:
               x_train_extended = np.concatenate((x_train_extended, np.reshape(reshaped_line, (
                   1, 3 * int(x_train.shape[1] / num_timesteps) * num_timesteps))), axis=0)

       x_train = x_train_extended


    if add_external_signal_variance:
        print("Add noise for sensor variance")
        if gpu_id >= 0:
            x_train = cuda.to_gpu(x_train) + xp.sqrt(external_signal_variance_vec) * xp.random.randn(x_train.shape[0], x_train.shape[1])
            x_train = xp.float32(cuda.to_cpu(x_train))
        else:
            x_train += xp.sqrt(external_signal_variance_vec) * xp.random.randn(x_train.shape[0], x_train.shape[1])

    # c_train = xp.arange(num_classes)
    # if num_samples_per_class > 1:
    #     c_train = xp.tile(xp.array(c_train), num_samples_per_class)
    c_train = np.load(training_data_file_classes)

    # x_train is of dimensionality (num_classes * num_samples_per_class) x (num_timesteps * num_io)
    # each row of x_train: [t1dim1, t1dim2, t2dim1, t2dim2...]
    # each column of x_train: [batch1class1, batch1class2, batch2class1, batch2class2...]

    batch_size = num_samples_per_class * num_classes

    # infer number of classes from
    # num_classes = max(c_train)+1
    inputDim = int(x_train.shape[1]/num_timesteps)
    plot_results(x_train[0:num_classes], num_timesteps, os.path.join(final_save_dir, 'target_trajectories.png'), inputDim,twoDim=True)

    ####################################################################################
    # batch_size, num_classes, x_train_orig, c_train and noise_variances should be set #
    ####################################################################################

    #################################################
    # x_train_orig => data normalization => x_train # => Not necessary here, because it is already in a valid range
    #################################################

    # Initialize parameter setting

    # training_ext_contrib = external_contrib_runs[r]
    training_ext_contrib = 1

    # training_tau = tau_runs[r]
    training_tau = 2

    # training_context_n = num_c_runs[r]
    training_context_n = 250

    #training_ini_var = ini_var_runs[r]
    training_ini_var = 10

    #aberrant_sensory_precision = aberrant_sensory_precision_runs[r]
    aberrant_sensory_precision = 0

    # excitation_bias = excitation_bias_runs[r]
    excitation_bias = 1/training_context_n # default 0.05

    # lrate = lrate_runs[r]
    lrate = 0.0005

    #conn = conn_runs[r]
    conn = training_context_n

    #var_integration = var_int_runs[r]
    var_integration = 2


    # # fixed initial states: one-of-k coding
    # oneMatrix = np.concatenate((1*np.ones((1,20)), -1*np.ones((1,20))))
    # initMatrix = np.zeros((num_classes, training_context_n),dtype=np.float32)
    # for i in range(num_classes):
    #     initMatrix[i,i] = 1
    #     initMatrix[i,i+num_classes] = 1
    #     initMatrix[i,i+2*num_classes] = 1
    #     initMatrix[i,i+3*num_classes] = 1
    #     initMatrix[i,i+4*num_classes] = 1 # for num_c = 40
    #     # initMatrix[i,i+5*num_classes] = 1
    #     # initMatrix[i,i+6*num_classes] = 1
    #     # initMatrix[i,i+7*num_classes] = 1 # for num_c = 70
    # #     # initMatrix[i,i+8*num_classes] = 1
    # #     # initMatrix[i,i+9*num_classes] = 1
    # #     # initMatrix[i,i+10*num_classes] = 1
    # #     # initMatrix[i,i+11*num_ classes] = 1 # for num_c = 100

    # CREATE PARAMETER SETTING AND NETWORK MODEL

    p = NetworkParameterSetting(epochs = epochs, batch_size = batch_size)

    p.set_network_type('SCTRNN', {'num_io':x_train.shape[1]/num_timesteps, 'num_c':training_context_n, 'lr':lrate, 'num_classes': num_classes,
       'learn_tau':False, 'tau_c':training_tau,
       'learn_init_states':True, 'init_state_init':'zero', 'init_state_var': training_ini_var, 'init_state_loss_scaling':1,
       'learn_weights':True,
       'learn_bias':learn_bias,
       'training_external_contrib':training_ext_contrib,
       'aberrant_sensory_precision':aberrant_sensory_precision,
       'rec_connection_factor':conn,
       'variance_integration_mode':var_integration,
       'hyp_prior':hyp_prior,
       'external_signal_variance':explicit_sensor_variance,
       'excitation_bias':excitation_bias,
       'bias_likelihood':False}) # 'alpha_init':0.001, 'momentum_init':0.9
       #'bias_likelihood':True}) # 'alpha_init':0.001, 'momentum_init':0.9

    connect_likelihood = False
    abs_connectivity = np.float32(0.05)

    with open(os.path.join(final_save_dir,"info.txt"),'w') as f:
       f.write(p.print_parameters())
       f.write("\n")
       # f.write("num_classes = " + str(num_classes) + ", noise_variances=" + str(noise_variances))
       #f.write("3 scaling varied circles, 2 init state " + str(c_train) + ", noise var in patterns 0.001")
       f.write("\n"+experiment_info)
       f.write("\n")
    f.close()

    # create new RNN model
    model = SCTRNN(p.num_io, p.num_c, p.tau_c, p.num_classes, init_state_init = p.init_state_init, init_state_learning = p.learn_init_states, weights_learning = p.learn_weights, bias_learning = p.learn_bias, tau_learning = p.learn_tau, external_contrib = p.training_external_contrib, aberrant_sensory_precision = p.aberrant_sensory_precision, excitation_bias = p.excitation_bias, rec_conn_factor = p.rec_connection_factor, variance_integration_mode = p.variance_integration_mode, hyp_prior = p.hyp_prior, external_signal_variance = p.external_signal_variance)
    # model = SMTRNN(p.num_io, p.num_fh, p.num_sh, p.tau_io, p.tau_fh, p.tau_sh, p.num_classes, init_state_init_sh = p.init_state_init, init_state_learning = p.learn_init_states, external_contrib = p.training_external_contrib, aberrant_sensory_precision = p.aberrant_sensory_precision)
    model.add_BI_variance = True
    model.set_init_state_learning(c_train)

    # store weights of first run, to be reused by next runs
    if runs > 1 and same_weights_per_run or reuse_existing_weights:
        if r == 0 and not reuse_existing_weights:
            xhW=model.x_to_h.W.data
            hhW=model.h_to_h.W.data
            hyW=model.h_to_y.W.data
            hvW=model.h_to_v.W.data
            np.save(os.path.join(init_weight_dir, 'xhW.npy'), xhW)
            np.save(os.path.join(init_weight_dir, 'hhW.npy'), hhW)
            np.save(os.path.join(init_weight_dir, 'hyW.npy'), hyW)
            np.save(os.path.join(init_weight_dir, 'hvW.npy'), hvW)
        else:
            print("Load predefined initial weights from " + init_weight_dir)
            xhW=np.load(os.path.join(init_weight_dir, 'xhW.npy'))
            hhW=np.load(os.path.join(init_weight_dir, 'hhW.npy'))
            hyW=np.load(os.path.join(init_weight_dir, 'hyW.npy'))
            hvW=np.load(os.path.join(init_weight_dir, 'hvW.npy'))
            model.x_to_h.W.data=xhW[:model.num_c, :]
            model.h_to_h.W.data=hhW[:model.num_c, :model.num_c]
            model.h_to_y.W.data=hyW[:, :model.num_c]
            model.h_to_v.W.data=hvW[:, :model.num_c]

    if runs > 1 and same_bias_per_run or reuse_existing_weights:
        if r == 0 and not reuse_existing_weights:
            xhb=model.x_to_h.b.data
            hhb=model.h_to_h.b.data
            hyb=model.h_to_y.b.data
            hvb=model.h_to_v.b.data
            np.save(os.path.join(init_weight_dir, 'xhb.npy'), xhb)
            np.save(os.path.join(init_weight_dir, 'hhb.npy'), hhb)
            np.save(os.path.join(init_weight_dir, 'hyb.npy'), hyb)
            np.save(os.path.join(init_weight_dir, 'hvb.npy'), hvb)
        else:
            print("Load predefined initial bias weights from " + init_weight_dir)
            xhb=np.load(os.path.join(init_weight_dir, 'xhb.npy'))
            hhb=np.load(os.path.join(init_weight_dir, 'hhb.npy'))
            hyb=np.load(os.path.join(init_weight_dir, 'hyb.npy'))
            hvb=np.load(os.path.join(init_weight_dir, 'hvb.npy'))
            model.x_to_h.b.data=xhb[:model.num_c]
            model.h_to_h.b.data=hhb[:model.num_c]
            model.h_to_y.b.data=hyb
            model.h_to_v.b.data=hvb

    save_network(final_save_dir, params=p, model = model, model_filename = "network-initial")

    if gpu_id >= 0:
       model.to_gpu(gpu_id)

    # Optimizer: takes care of updating the model using backpropagation
    optimizer = optimizers.Adam(p.lr)
    optimizer.setup(model)
    # optimizer.add_hook(chainer.optimizer.WeightDecay(0))

    history_init_state_var = np.zeros((epochs+1,))
    history_init_state_var[0] = np.mean(np.var(model.initial_states.W.array,axis=0))
    history_generation_error_proactive = np.empty((p.num_classes,), dtype=object)
    history_generation_error_reactive = np.empty((p.num_classes,), dtype=object)
    history_training_error = np.zeros((epochs+1,))
    history_training_variance_estimation = np.zeros((epochs+1,))

    likelihood_per_epoch = []

    print("actual variance of init_states_0: " + str(history_init_state_var[0]))

    # Evaluate the performance of the untrained network
    test_batch_size = num_classes
    if var_integration == 1:
        res, resv, resm = model.generate(model.initial_states.W.array, num_timesteps, external_contrib = 0, epsilon_disturbance = 0)
    elif var_integration == 2:
        res, resv, resm = model.generate(model.initial_states.W.array, num_timesteps, epsilon_disturbance = 0, external_signal_variance = ext_var_proactive)
    results = cuda.to_cpu(res)

    for i in range(num_classes):
       generation_error = evaluate_generation(x_train_orig[i,:].reshape((-1,num_io)), results[i,:].reshape((-1,num_io)), method='dtw')
       history_generation_error_proactive[i] = [generation_error]
       with open(os.path.join(final_save_dir,"evaluation.txt"),'a') as f:
           f.write("before learning: pattern generation error (proactive): " + str(history_generation_error_proactive[i]) + "\n")
    plot_results(results, num_timesteps, os.path.join(final_save_dir, "proactive_before-learning"), model.num_io, twoDim=True)

    #COPY?
    if var_integration == 1:
        res, resv, resm, pe, wpe = model.generate(model.initial_states.W.array, num_timesteps, external_contrib = 1, external_input = xp.copy(xp.asarray(x_train[:test_batch_size,:])), epsilon_disturbance = 0)
    elif var_integration == 2:
        res, resv, resm, pe, wpe, respos = model.generate(model.initial_states.W.array, num_timesteps, external_input = xp.copy(xp.asarray(x_train[:test_batch_size,:])), epsilon_disturbance = 0, external_signal_variance = explicit_sensor_variance)
    results = cuda.to_cpu(res)

    for i in range(p.num_classes):
       generation_error = evaluate_generation(x_train_orig[i,:].reshape((-1,num_io)), results[i,:].reshape((-1,num_io)), method='dtw')
       history_generation_error_reactive[i] = [generation_error]
       with open(os.path.join(final_save_dir, "evaluation.txt"),'a') as f:
           f.write("before learning: pattern generation error (reactive): " + str(history_generation_error_reactive[i]) + "\n")
    plot_results(results, num_timesteps, os.path.join(final_save_dir, "reactive_before-learning"), model.num_io, twoDim=True)

    # arrays for tracking likelihood and determining stop condition
    all_mean_diffs = []
    all_std_diffs = []
    m1s = []
    s1s = []
    tmp_epoch_marker = 0
    conv_eval_interval = 500 # the length of the interval to consider for determining convergence
    mean_threshold = 1e-3
    std_threshold = 5e-2
    for epoch in range(1, p.epochs + 1):
        epochStart = time.time()

        outv = np.zeros((num_timesteps,))

        # permutate samples in each epoch so that they are randomly ordered
        # x_train = create_lissajous_curves_murata_four_positions(num_classes, num_repetitions, num_timesteps, noise_variances, num_samples_per_class = num_samples_per_class, format = 'array')

        x_train_orig = np.float32(np.load(training_data_file))
        x_train = np.copy(x_train_orig)
        if implicit_trajectory_noise > 0:
            print("Adding implicit trajectory noise")
            x_train += p.sqrt(trajectory_noise_variances) * np.random.randn(x_train.shape[0], x_train.shape[1])
        if add_smooth_traj_noise:
            print("Adding smooth trajectory noise")
            for i in range(x_train.shape[0]):
                x_train[i,:] += np.reshape(get_trajectory_noise(num_timesteps, int(num_timesteps/traj_noise_parts), dim=num_io), (num_io*num_timesteps,))


        if use_derivatives == True:
           # add also velocity and acceleration
           for i in range(0, x_train.shape[0]):

               reshaped_line = np.copy(np.reshape(x_train[i, 0:x_train.shape[1]],
                                                  (num_timesteps, int(x_train.shape[1] / num_timesteps))))
               reshaped_line_x = reshaped_line[:, 0]
               reshaped_line_y = reshaped_line[:, 1]

               # for velocity and acceleration
               velocity_x = np.copy(np.reshape(np.gradient(reshaped_line_x), (num_timesteps,)))
               velocity_y = np.copy(np.reshape(np.gradient(reshaped_line_y), (num_timesteps,)))

               acceleration_x = np.reshape(np.gradient(velocity_x), (num_timesteps, 1))
               acceleration_y = np.reshape(np.gradient(velocity_y), (num_timesteps, 1))

               velocity_x = np.reshape(velocity_x, (num_timesteps, 1))
               velocity_y = np.reshape(velocity_y, (num_timesteps, 1))

               velocity = np.concatenate((velocity_x, velocity_y), axis=1)

               acceleration = np.concatenate((acceleration_x, acceleration_y), axis=1)

               # normalize input
               reshaped_line = reshaped_line / np.amax(reshaped_line)
               velocity = velocity / np.amax(velocity)
               acceleration = acceleration / np.amax(acceleration)

               reshaped_line = np.concatenate((reshaped_line, velocity, acceleration), axis=1)

               if i == 0:
                   x_train_extended = np.reshape(reshaped_line,
                                                 (1, 3 * int(x_train.shape[1] / num_timesteps) * num_timesteps))
               else:
                   x_train_extended = np.concatenate((x_train_extended, np.reshape(reshaped_line, (
                       1, 3 * int(x_train.shape[1] / num_timesteps) * num_timesteps))), axis=0)

           x_train = x_train_extended

        if add_external_signal_variance:
            print("Add noise for sensor variance")
            if gpu_id >= 0:
                x_train = cuda.to_gpu(x_train) + xp.sqrt(external_signal_variance_vec) * xp.random.randn(x_train.shape[0], x_train.shape[1])
                x_train = xp.float32(cuda.to_cpu(x_train))
            else:
                x_train += xp.sqrt(external_signal_variance_vec) * xp.random.randn(x_train.shape[0], x_train.shape[1])

        print(np.max([np.mean(np.var(x_train[x:num_classes*num_samples_per_class:num_classes,:],axis=0)) for x in range(num_classes)]))

        # plt.figure()
        # plt.plot(np.reshape(x_train[0,:], (70,2))[:,0], np.reshape(x_train[0,:], (70,2))[:,1])
        # plt.savefig("train-traj-" + str(epoch))

        perm = np.random.permutation(x_train.shape[0])

        # here, one batch equals the full training set
        x_batch = xp.asarray(x_train[perm])
        model.set_init_state_learning(c_train[perm])

        mean_init_states = chainer.Variable(xp.zeros((),dtype=xp.float32))
        mean_init_states = chainer.functions.average(model.initial_states.W,axis=0) #keepdims=True
        #mean_init_states = xp.mean(c0.array,axis=0) # using this instead causes no difference in resulting gradient of c0

        # compute h_to_h bias mean
        if p.bias_likelihood:
           mean_h_to_h_bias = chainer.Variable(xp.zeros((),dtype=xp.float32))
           mean_h_to_h_bias = chainer.functions.tile(chainer.functions.average(model.h_to_h.b,keepdims=True), (model.h_to_h.b.shape[0],))

        if connect_likelihood:
           mean_h_to_h_connectivity = chainer.Variable(xp.zeros((),dtype=xp.float32))
           mean_h_to_h_connectivity = chainer.functions.tile(chainer.functions.average(chainer.functions.absolute(model.h_to_h.W)), (model.h_to_h.W.shape[0],model.h_to_h.W.shape[1]))

        # initialize error
        acc_loss = chainer.Variable(xp.zeros((),dtype=xp.float32)) # for weight backprop
        acc_init_loss = chainer.Variable(xp.zeros((),dtype=xp.float32)) # for init states backprop
        acc_bias_loss = chainer.Variable(xp.zeros((),dtype=xp.float32)) # for keeping the bias distribution at a desired variance
        acc_conn_loss = chainer.Variable(xp.zeros((),dtype=xp.float32)) # for keeping the network weights at a certain connectivity
        err = xp.zeros(()) # for evaluation only

        # clear gradients from previous batch
        model.cleargrads()
        # clear output and variance estimations from previous batch
        model.reset_current_output()

        t=0 # iterate through time
        x_t = x_batch[:, p.num_io*t:p.num_io*(t+1)]
        # next time step to be predicted (for evaluation)
        x_t1 = x_batch[:, p.num_io*(t+1):p.num_io*(t+2)]
        # x_t = xp.reshape(x_batch[0][t,:], (1, p.num_io))
        # x_t1 = xp.reshape(x_batch[0][t+1,:], (1, p.num_io))
        # for i in range(1, p.batch_size):
        #     x_t = np.concatenate((x_t, xp.reshape(x_batch[i][t,:], (1,p.num_io))),axis=0)
        #     x_t1 = np.concatenate((x_t1, xp.reshape(x_batch[i][t+1,:], (1,p.num_io))),axis=0)

        # execute first forward step
        #COPY?
        u_h, y, v = model(xp.copy(x_t), None) # initial states of u_h are set automatically according to model.classes

        # noisy output estimation
        #y_out = y.array + xp.sqrt(v.array) * xp.random.randn()

        # compute prediction error, averaged over minibatch

        #example_trajectory = np.zeros((num_timesteps, model.num_io))
        #example_trajectory[0,:] = model.current_x.data[0,:]

        if prediction_error_type == 'standard':
            # compare output to ground truth
            loss_i = chainer.functions.gaussian_nll(chainer.Variable(x_t1), y, exponential.log(v))
        elif prediction_error_type == 'integrated':
            # compare output to posterior of perception
           if var_integration == 1:
               integrated_x = p.training_external_contrib * chainer.Variable(x_t1) + (1 - p.training_external_contrib) * (y + chainer.functions.sqrt(v) * xp.random.randn())
               loss_i = chainer.functions.gaussian_nll(integrated_x, y, exponential.log(v))
           elif var_integration == 2:
               loss_i = chainer.functions.gaussian_nll(model.current_x, y, exponential.log(v))

        acc_loss += loss_i

        # compute error for evaluation purposes
        err += chainer.functions.mean_squared_error(chainer.Variable(x_t), y).array.reshape(()) * p.batch_size

        outv[t] = xp.mean(v.array)

        # rollout trajectory
        for t in range(1,num_timesteps-1):
           # current time step
           x_t = x_batch[:, p.num_io*t:p.num_io*(t+1)]
           # next time step to be predicted (for evaluation)
           x_t1 = x_batch[:, p.num_io*(t+1):p.num_io*(t+2)]

           #COPY?
           u_h, y, v = model(xp.copy(x_t), u_h)

           # noisy output estimation
           #y_out = y.array + xp.sqrt(v.array) * xp.random.randn()

           # compute error for backprop for weights
           if prediction_error_type == 'standard':
               loss_i = chainer.functions.gaussian_nll(chainer.Variable(x_t1), y, exponential.log(v))
           elif prediction_error_type == 'integrated':
                if var_integration == 1:
                    integrated_x = p.training_external_contrib * chainer.Variable(x_t1) + (1 - p.training_external_contrib) * (y + chainer.functions.sqrt(v) * xp.random.randn())
                    loss_i = chainer.functions.gaussian_nll(integrated_x, y, exponential.log(v))
                elif var_integration == 2:
                    loss_i = chainer.functions.gaussian_nll(model.current_x, y, exponential.log(v))
           acc_loss += loss_i

           #example_trajectory[t,:] = model.current_x.data[0,:]

           # compute error for evaluation purposes
           err += chainer.functions.mean_squared_error(chainer.Variable(x_t), y).array.reshape(()) * p.batch_size

           outv[t] = xp.mean(v.array)


        # plt.figure()
        # plt.plot(example_trajectory[:,0], example_trajectory[:,1])
        # plt.savefig("blubb-" + str(epoch))

        # for each training sequence of this batch: compute loss for maintaining desired initial state variance
        for s in range(len(c_train)):
           if gpu_id >= 0:
               acc_init_loss += chainer.functions.gaussian_nll(model.initial_states()[model.classes][s], mean_init_states, exponential.log(cuda.to_gpu(p.init_state_var, device=gpu_id)))
               if p.bias_likelihood:
                   acc_bias_loss += chainer.functions.gaussian_nll(model.h_to_h.b, mean_h_to_h_bias, exponential.log(cuda.to_gpu(np.repeat(p.excitation_bias, p.num_c), device=gpu_id)))
               if connect_likelihood:
                   acc_conn_loss += chainer.functions.gaussian_nll(model.h_to_h.W, mean_h_to_h_connectivity, exponential.log(cuda.to_gpu(np.tile(abs_connectivity, (p.num_c, p.num_c)), device=gpu_id)))
           else:
               acc_init_loss += chainer.functions.gaussian_nll(model.initial_states()[model.classes][s], mean_init_states, exponential.log(p.init_state_var))
               if p.bias_likelihood:
                   acc_bias_loss += chainer.functions.gaussian_nll(model.h_to_h.b, mean_h_to_h_bias, exponential.log(np.repeat(p.excitation_bias, p.num_c)))
               if connect_likelihood:
                   acc_conn_loss += chainer.functions.gaussian_nll(model.h_to_h.W, mean_h_to_h_connectivity, exponential.log(np.tile(abs_connectivity, (p.num_c, p.num_c))))


        # compute gradients
        # (gradients from L_out and L_init are summed up)
        # gradient of initial states equals:
        # 1/p.init_state_var * (c0[cl]-mean_init_states).array

        epochBatchProcessed = time.time()
        print("Elapsed time (batch processing): " + str(epochBatchProcessed - epochStart))

        #        if p.learn_init_states:
        acc_init_loss *= p.init_state_loss_scaling
        acc_init_loss.backward()
        #print("acc_init_loss.backward initial_states gradient: " + str(model.initial_states.W.grad[0][0]) + ", " + str(model.initial_states.W.grad[1][0]))
        acc_loss.backward()

        if p.bias_likelihood:
           acc_bias_loss.backward()

        if connect_likelihood:
           acc_conn_loss.backward()

        print("update")
        optimizer.update()

        # printing and testing
        print("Done epoch " + str(epoch))
        print("Elapsed time (error computation): " + str(time.time() - epochBatchProcessed))
        # print("Tau: " + str(model.tau_c.W.array[0]))
        error = err/p.batch_size/num_timesteps
        #minVar_t = np.mean([np.min(np.mean(np.sqrt(outv[b][50:-1]),axis=1)) for b in range(p.batch_size)])
        #maxVar_t = np.mean([np.max(np.mean(np.sqrt(outv[b][50:-1]),axis=1)) for b in range(p.batch_size)])
        mean_estimated_var = xp.mean(outv)
        history_training_error[epoch] = error
        history_training_variance_estimation[epoch] = mean_estimated_var

        print("train MSE = " + str(error) + "\nmean estimated var: " + str(mean_estimated_var))
        print("init_states = [" + str(model.initial_states.W.array[0][0]) + "," + str(model.initial_states.W.array[0][1]) + "...], var: " + str(np.mean(np.var(model.initial_states.W.array,axis=0))) + ", accs: " + str(acc_loss) + " + " + str(acc_init_loss) + " + " + str(acc_bias_loss))

        if p.bias_likelihood:
           likelihood_per_epoch.append(np.float64(acc_loss.array+acc_init_loss.array+acc_bias_loss.array))
        else:
           likelihood_per_epoch.append(np.float64(acc_loss.array+acc_init_loss.array))

        history_init_state_var[epoch] = np.mean(np.var(model.initial_states.W.array,axis=0))

        with open(os.path.join(final_save_dir,"evaluation.txt"),'a') as f:
           f.write("epoch: " + str(epoch)+ "\n")
           f.write("train MSE = " + str(error) + "\nmean estimated var: " + str(mean_estimated_var))
           f.write("initial state var: " + str(history_init_state_var[epoch]) + ", precision loss: " + str(acc_loss) + ", variance loss: " + str(acc_init_loss) + " + " + str(acc_bias_loss) + "\ninit states:\n")
           for i in range(p.num_classes):
           # f.write('init_states = [' + str(model.initial_states.W[0][0]) + ',' + str(model.initial_states.W[0][1]) + '...], [' + str(model.initial_states.W[1][0]) + ',' + str(model.initial_states.W[1][1]) + '...], var: ' + str(np.mean(np.var(model.initial_states.W.array,axis=0))) + ', accs: ' + str(acc_loss) + " + " + str(acc_init_loss)+'\n')
               f.write("\t[" + str(model.initial_states.W[i][0]) + "," + str(model.initial_states.W[i][1]) + "...]\n")
        f.close()

        if epoch%save_interval == 1 or epoch == p.epochs:
           # evaluate proactive generation

           if var_integration == 1:
               res, resv, resm, u_h_history = model.generate(model.initial_states.W.array, num_timesteps, external_contrib = 0, epsilon_disturbance = 0, additional_output='activations')
           elif var_integration == 2:
               res, resv, resm, u_h_history = model.generate(model.initial_states.W.array, num_timesteps, epsilon_disturbance = 0, additional_output='activations', external_signal_variance = ext_var_proactive)
           results = cuda.to_cpu(res)

           plot_results(results, num_timesteps, os.path.join(final_save_dir, "proactive_epoch-" + str(epoch).zfill(len(str(epochs)))), model.num_io, twoDim=True)

           current_generation_error = np.zeros((1,num_classes))
           for i in range(p.num_classes):
               generation_error_pro = evaluate_generation(x_train_orig[i,:].reshape((-1,num_io)), results[i,:].reshape((-1,num_io)), method='dtw')
               history_generation_error_proactive[i].append(generation_error_pro)
               current_generation_error[0,i] = generation_error_pro
               with open(os.path.join(final_save_dir, "evaluation.txt"), 'a') as f:
                   f.write("pattern generation error (proactive): " + str(generation_error_pro) + "\n")
               f.close()

           plot_pca_activations(u_h_history, num_timesteps, os.path.join(final_save_dir, "pca_context_act_proactive_epoch-" + str(epoch).zfill(len(str(epochs)))), p.num_c, p.num_classes)

           # evaluate reactive generation
           #COPY?
           if var_integration == 1:
               res, resv, resm, pe, wpe, u_h_history = model.generate(model.initial_states.W.array, num_timesteps, external_contrib = 1, external_input = xp.copy(xp.asarray(x_train[:test_batch_size,:])), epsilon_disturbance = 0, additional_output='activations')
           elif var_integration == 2:
               res, resv, resm, pe, wpe, u_h_history, respos = model.generate(model.initial_states.W.array, num_timesteps, external_input = xp.copy(xp.asarray(x_train[:test_batch_size,:])), epsilon_disturbance = 0, additional_output='activations', external_signal_variance = explicit_sensor_variance)
           results = cuda.to_cpu(res)

           plot_results(results, num_timesteps, os.path.join(final_save_dir, "reactive_epoch-" + str(epoch).zfill(len(str(epochs)))), model.num_io, twoDim=True)

           for i in range(test_batch_size):
               generation_error_re = evaluate_generation(x_train_orig[i,:].reshape((-1,num_io)), results[i,:].reshape((-1,num_io)), method='dtw')
               history_generation_error_reactive[i].append(generation_error_re)
               with open(os.path.join(final_save_dir, "evaluation.txt"), 'a') as f:
                   f.write("pattern generation error (reactive): " + str(generation_error_re) + "\n")
               f.close()

           plot_pca_activations(u_h_history, num_timesteps, os.path.join(final_save_dir, "pca_context_act_reactive_epoch-" + str(epoch).zfill(len(str(epochs)))), p.num_c, p.num_classes)

        if epoch%save_model_interval == 1 or epoch == p.epochs:
            save_network(final_save_dir, p, model, model_filename="network-epoch-"+str(epoch).zfill(len(str(epochs))))
            np.save(os.path.join(final_save_dir,"history_init_state_var"), np.array(history_init_state_var))
            np.save(os.path.join(final_save_dir,"history_generation_error_proactive"), np.array(history_generation_error_proactive))
            np.save(os.path.join(final_save_dir,"history_generation_error_reactive"), np.array(history_generation_error_reactive))
            np.save(os.path.join(final_save_dir,"history_training_error"), np.array(history_training_error))
            np.save(os.path.join(final_save_dir, "history_training_variance_estimation"), np.array(history_training_variance_estimation[0:epoch]))

            fig = plt.figure()
            ax = fig.add_subplot(121)
            for i in range(p.num_classes):
               ax.plot(np.concatenate(([0,1],np.arange(1,len(history_generation_error_proactive[i])-1)*save_interval)), history_generation_error_proactive[i])
            ax = fig.add_subplot(122)
            for i in range(p.num_classes):
               ax.plot(np.concatenate(([0,1],np.arange(1,len(history_generation_error_proactive[i])-1)*save_interval)), history_generation_error_reactive[i], label=str(i))
            plt.title("generation error (proactive / reactive)")
            plt.legend()
            fig.savefig(os.path.join(final_save_dir,"generation-error"))
            plt.close()

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(np.arange(0,len(history_init_state_var)), history_init_state_var)
            plt.title("init state variance")
            fig.savefig(os.path.join(final_save_dir,"init-state-var"))
            plt.close()

            np.save(os.path.join(final_save_dir, "likelihood-per-epoch"), np.asarray(likelihood_per_epoch))

            plt.figure()
            plt.plot(np.arange(len(all_std_diffs)),all_std_diffs, 'bo',label='std diff')
            plt.plot(np.arange(len(all_mean_diffs)),all_mean_diffs, 'ro',label='mean diff')
            plt.legend()
            plt.savefig(os.path.join(final_save_dir, 'convergence-condition.png'))
            plt.close()

            if np.mean(current_generation_error) < best_epoch_error:
                best_epoch_error = np.mean(current_generation_error)
                best_epoch = epoch
                with open(os.path.join(final_save_dir, "evaluation.txt"), 'a') as f:
                    f.write("New best epoch: " + str(best_epoch) + " with error " + str(best_epoch_error) + "\n")
                f.close()

            if check_best_improvement_stop:
                if epoch-best_epoch > patience:
                    print("No improvement within the last " + str(patience) + " epochs. Stop training! Best epoch: " + str(best_epoch))
                    with open(os.path.join(final_save_dir,"info.txt"), 'a') as f:
                        f.write("Best epoch: " + str(best_epoch) + " with error " + str(best_epoch_error))
                        f.write("\n")
                    f.close()
                    from distutils.file_util import copy_file
                    copy_file(os.path.join(final_save_dir, "network-epoch-"+str(best_epoch).zfill(len(str(epochs))) + ".npz"), os.path.join(final_save_dir, "network-epoch-best.npz"))
                    break

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
               # absolute variance decrease in comparison to first time step (=> converges against 1 which means that variance is zero)
               # std_diff = (s1s[0]-s1) / np.abs(s1s[0])
               # std_diff = (s1s[len(s1s)-conv_eval_interval] - s1s[len(s1s)-1]) /
               std_diff = (s1s[len(s1s)-conv_eval_interval] - s1s[len(s1s)-1]) / np.max([np.abs(s1s[len(s1s)-conv_eval_interval]), np.abs(s1s[len(s1s)-1])])

               all_mean_diffs.append(mean_diff)
               all_std_diffs.append(std_diff)
               print(str(epoch) + ": mean_diff: " + str(mean_diff) + ", std_diff: " + str(std_diff))
               if (-mean_threshold < mean_diff < mean_threshold) and (-std_threshold < std_diff < std_threshold) and (epoch > minimum_epochs): # and (std_diff > 0.95):
                   if tmp_epoch_marker == 0:
                       break
                       tmp_epoch_marker = epoch
                   elif epoch > tmp_epoch_marker * 1.2:
                       break


    save_network(final_save_dir, p, model, model_filename = "network-final")

    plt.figure()
    plt.plot(np.arange(len(all_std_diffs)),all_std_diffs, 'bo',label='std diff')
    plt.plot(np.arange(len(all_mean_diffs)),all_mean_diffs, 'ro',label='mean diff')
    plt.legend()
    plt.savefig(os.path.join(final_save_dir, 'convergence-condition.png'))
    plt.close()

    neuron_acts = u_h_history.reshape((num_classes*num_timesteps,model.num_c))
    neuron_act_std = np.std(neuron_acts,axis=0)
    max_act = np.max(neuron_act_std)

    with open(os.path.join(final_save_dir, "evaluation.txt"), 'a') as f:
       print("Neuron activation distribution:")
       f.write("Neuron activation distribution:")
       for thr in [0.01, 0.05, 0.08, 0.1, 0.2, 0.5, 0.8]:
           print(str((neuron_act_std/np.max(neuron_act_std) < thr).tolist().count(True)) + "/" + str(model.num_c) + " < " + str(thr))
           f.write(str((neuron_act_std/np.max(neuron_act_std) < thr).tolist().count(True)) + "/" + str(model.num_c) + " < " + str(thr))
    f.close()

    np.save(os.path.join(final_save_dir, "likelihood-per-epoch"), np.asarray(likelihood_per_epoch))


