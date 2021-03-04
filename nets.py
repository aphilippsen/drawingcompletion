import os
import sys
import math

import chainer
from chainer.backends import cuda
from chainer import Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

import numpy as np
from matplotlib import pyplot as plt
import pickle

# Stochastic Continuous-Time Recurrent Neural Network

class SCTRNN(chainer.Chain):
    def __init__(self, num_io, num_context, tau_context, num_classes, init_state_init = None, init_state_learning = True, weights_learning = True, bias_learning = True, tau_learning = False, aberrant_sensory_precision = 0, hyp_prior = 1, external_signal_variance = None, pretrained_model = None):
        """
        Implements the Stochastic Continuous-Time Recurrent Neural Network
        num_io: Number of input and output neurons
        num_context: Number of recurrent layer neurons
        tau_context: Time scale (initialization) (if large: slow update;
                    if small: fast update of context neuron activations)
        num_classes: Number of different attractors that the network can
                    maximally represent (=> number of initial states)
        init_state_init (None): Numpy 2-d matrix for initializing the initial
                    states; must be of size (num_classes x num_context). If None
                    initialized with zeros.
        init_state_learning (True): if True initial states are updated during
                    optimization, if False they are fixed to the given initial values.
        weights_learning (True): Whether to update weights.
        bias_learning (True): Whether to update the bias values while learning.
        tau_learning (False): if True, the tau factor is updated during
                    optimization, if False tau is fixed
        hyp_prior (1): Increases (hyp_prior < 1) or decreases (hyp_prior > 1) the reliance
                    on the networks own prediction during learning. Parameter H.
        external_signal_variance (None): If (None), external signal variance is set
                    identical to the estimated variance, otherwise to fixed value (per
                    dimension)
        aberrant_sensory_precision (0): If != 0, the network under- or overestimates
                    its estimated variance during training.
        pretrained_model (None): If given, initial states and network weights
                    are copied from the other (previously trained) SCTRNN instance.
        """
        super(SCTRNN, self).__init__()

        if pretrained_model:
            # reuse initial states from pretraining and add new ones according to number of classes
            if num_classes - pretrained_model.initial_states.W.array.shape[0] < 1:
                init_state_init = np.zeros((num_classes, num_context), dtype=np.float32)
            else:
                init_state_init = np.concatenate((pretrained_model.initial_states.W.array, np.zeros((num_classes - pretrained_model.initial_states.W.array.shape[0], num_context), dtype=np.float32)))
                print("Reuse initial states from pretrained model!")

        elif init_state_init is None:
            init_state_init = np.zeros((num_classes,num_context),dtype=np.float32)
            print("Initialize initial states with zeros!")

        # define learning parameters of net
        with self.init_scope():

            # define initialization methods
            W_init = chainer.initializers.LeCunUniform()
            b_init = chainer.initializers.Normal(scale = 1/num_context)

            if init_state_learning:
                self.initial_states = L.Parameter(init_state_init)
            if weights_learning and bias_learning:
                # Links
                self.x_to_h = L.Linear(num_io, num_context, initialW = W_init, initial_bias = b_init)
                self.h_to_h = L.Linear(num_context, num_context, initialW = W_init, initial_bias = b_init)
                self.h_to_y = L.Linear(num_context, num_io, initialW = W_init, initial_bias = b_init)
                self.h_to_v = L.Linear(num_context, num_io, initialW = W_init, initial_bias = b_init)
            elif weights_learning:
                # Links
                self.x_to_h = L.Linear(num_io, num_context, initialW = W_init, nobias=True)
                self.h_to_h = L.Linear(num_context, num_context, initialW = W_init, nobias=True)
                self.h_to_y = L.Linear(num_context, num_io, initialW = W_init, nobias=True)
                self.h_to_v = L.Linear(num_context, num_io, initialW = W_init, nobias=True)

            if tau_learning:
                self.tau_c = L.Parameter(np.float32(np.asarray([tau_context])))

        if not init_state_learning:
            self.initial_states = L.Parameter(init_state_init)
        if not weights_learning:
            self.x_to_h = L.Linear(num_io, num_context, initialW = W_init, initial_bias = b_init)
            self.h_to_h = L.Linear(num_context, num_context, initialW = W_init, initial_bias = b_init)
            self.h_to_y = L.Linear(num_context, num_io, initialW = W_init, initial_bias = b_init)
            self.h_to_v = L.Linear(num_context, num_io, initialW = W_init, initial_bias = b_init)
        if not tau_learning:
            self.tau_c = L.Parameter(np.float32(np.asarray([tau_context])))

        if not bias_learning:
            # define fixed bias values which are added in each __call__() call and initialize it
            self.x_to_h_bias = L.Bias(axis=1, shape=(num_context,))
            fixed_bias = chainer.Parameter(b_init, (num_context,))
            self.x_to_h_bias.b = fixed_bias

            self.h_to_h_bias = L.Bias(axis=1, shape=(num_context,))
            fixed_bias = chainer.Parameter(b_init, (num_context,))
            self.h_to_h_bias.b = fixed_bias

        # other class properties
        self.num_io = num_io
        self.num_c = num_context
        self.aberrant_sensory_precision = aberrant_sensory_precision

        # store which initial states were determined when using init_state policy 'best' in generate()
        self.used_is_idx = []

        self.reset_current_output()

        self.gpu_id = -1

        self.bias_learning = bias_learning

        # Whether to [True] sample from the resulting distribution of the BI (or [False] just take the mean)
        self.add_BI_variance = True

        self.hyp_prior = hyp_prior
        self.external_signal_variance = external_signal_variance

        if pretrained_model:
            self.x_to_h.W.array=pretrained_model.x_to_h.W.array
            self.x_to_h.b.array=pretrained_model.x_to_h.b.array
            self.h_to_h.W.array=pretrained_model.h_to_h.W.array
            self.h_to_h.b.array=pretrained_model.h_to_h.b.array
            self.h_to_y.W.array=pretrained_model.h_to_y.W.array
            self.h_to_y.b.array=pretrained_model.h_to_y.b.array
            self.h_to_v.W.array=pretrained_model.h_to_v.W.array
            self.h_to_v.b.array=pretrained_model.h_to_v.b.array

        self.classes = []

    def to_gpu(self, device=None):
        """
        Override method: copy also non-updated parameters to GPU
        """
        self.gpu_id = device

        super(SCTRNN, self).to_gpu(device)

        # separately shift variables to GPU which might not be included in superclass, depending on learning condition
        self.initial_states.to_gpu(device)
        self.tau_c.to_gpu(device)
        self.x_to_h.to_gpu(device)
        self.h_to_h.to_gpu(device)
        self.h_to_y.to_gpu(device)
        self.h_to_v.to_gpu(device)
        try:
            self.x_to_h_bias.to_gpu(device)
            self.h_to_h_bias.to_gpu(device)
        except:
            pass

    def to_cpu(self):
        """
        Override method: copy also non-updated parameters to GPU
        """
        self.gpu_id = -1

        super(SCTRNN, self).to_cpu()

        # separately shift variables to GPU which might not be included in superclass, depending on learning condition
        self.initial_states.to_cpu()
        self.tau_c.to_cpu()
        self.x_to_h.to_cpu()
        self.h_to_h.to_cpu()
        self.h_to_y.to_cpu()
        self.h_to_v.to_cpu()
        try:
            self.x_to_h_bias.to_cpu()
            self.h_to_h_bias.to_cpu()
        except:
            pass

    def reset_current_output(self):
        """
        Reset current_y and current_v.
        """
        self.current_y = chainer.Variable()
        self.current_v = chainer.Variable()

    def set_init_state_learning(self, classes):
        """
        Define which initial states belong to which class in the
        batch provided for learning.
        classes: array of size [batch_size]
        """
        self.classes = classes

    def set_initial_states_zero(self):
        self.initial_states.W.array = np.float32(np.zeros((self.initial_states.W.array.shape[0], self.initial_states.W.array.shape[1])))

    def set_initial_states_mean(self):
        self.initial_states.W.array = np.tile(np.mean(self.initial_states.W.array,axis=0),(self.initial_states.W.array.shape[0],1))

    def __call__(self, x_data, u_h):
        """
        Perform one forward step computation given the input x_data,
        and the current states of the input neurons (u_io) and the
        hidden context layer neurons (u_h).
        If u_h is None, the initial states are used instead.
        """

        from chainer.backends import cuda
        xp = cuda.get_array_module(self.initial_states.W)

        x = chainer.Variable(x_data)#.reshape((1,-1)))

        if not self.current_y.array is None:

            # get mean and variance for prior distribution
                        
            pred_mean = self.current_y
            # prediction variance is taken from current variance estimation which expresses ability to predict the variance in the training data
            # hyp_prior multiplication of factor H
            pred_var = self.current_v * (self.hyp_prior)

            if not xp.any(pred_var.array):
                # It should not happen, but just in case check for zero prediction variance
                print("Predicted variance is zero!!!!")
                pred_var.array = xp.tile(np.float32(0.00001), pred_var.array.shape)

            # get mean and variance for input distribution

            input_mean = x
            # if external_signal_variance is not available, set equal to pred_var
            if self.external_signal_variance is None:
                input_var = pred_var
            else:
                input_var = chainer.Variable(xp.tile(xp.asarray(xp.float32([self.external_signal_variance])), (x.shape[0],x.shape[1])))

            ### Bayesian inference ###
            if xp.any(xp.isinf(input_var.array)):
                sigma_BI = xp.sqrt(pred_var.array)
                mu_BI = pred_mean.array
            else:
                # standard deviation of BI signal
                sigma_BI = xp.sqrt(xp.divide(xp.multiply(input_var.array, pred_var.array), (input_var.array + pred_var.array)))
                # mean of BI signal
                mu_BI = xp.power(sigma_BI, 2) * (xp.divide(pred_mean.array, pred_var.array) + xp.divide(input_mean.array, input_var.array))

            # sample from posterior distribution to get new input for the network
            if self.add_BI_variance:
                if cuda.get_array_module(self.initial_states.W) == np:
                    x.array = mu_BI + sigma_BI * xp.float32(xp.random.randn(sigma_BI.shape[0], sigma_BI.shape[1]))
                else:
                    x.array = mu_BI + sigma_BI * (xp.random.randn(sigma_BI.shape[0], sigma_BI.shape[1], dtype=xp.float32))
            else:
                x.array = mu_BI

        self.current_x = x

        # in the first time step no u_h is given => set initial states
        if u_h is None:
            u_h = self.initial_states()[self.classes]

        # forward context mapping
        
        # update recurrent connections
        h = F.tanh(u_h)
        if self.bias_learning:
            recurrent_connections = F.transpose(F.tensordot(self.h_to_h.W, F.transpose(h), axes=1)) + F.tile(self.h_to_h.b, (h.shape[0], 1))
            u_h = chainer.functions.scale(self.x_to_h(x) + recurrent_connections, 1/self.tau_c()) + chainer.functions.scale(u_h, (1 - 1/self.tau_c()))
        else:
            u_h = chainer.functions.scale(self.x_to_h_bias(self.x_to_h(x)) + self.h_to_h_bias(self.h_to_h(h)), 1/self.tau_c()) + chainer.functions.scale(u_h, (1 - 1/self.tau_c()))

        # forward output mapping
        u_out = self.h_to_y(h)
        y = F.tanh(u_out)

        # forward variance mapping
        u_v = self.h_to_v(h)
        v = F.exp(u_v + self.aberrant_sensory_precision) + 0.00001 # [Idei et al. 2017]

        self.current_y = y
        self.current_v = v

        return u_h, y, v

    def generate(self, init_states, num_steps, external_input = None, add_variance_to_output = None, additional_output='none', external_signal_variance = -1, add_BI_variance = True, hyp_prior = None, x_start = None):
        """
        Use the trained model and the given ``init_states`` to generate a sequence of length ``num_steps`` via closed- or open-loop control.

        Args:
            init_states: A numpy array of initial state for initializing the network activations
            num_steps: Integer number of steps to generate
            external_input: A numpy array of external input signals for each initial state.
                One input signal per initial state should be provided. If there are more input signals than initial states, it
                should be a multiple of the initial states, then the initial states array gets copied to match the length of ``external_input``.
            add_variance_to_output (None): Float value, indicating the amount of noise (as standard deviation of a Gaussian distribution) added to the
                network estimated average in each time step. If None, the variance estimated by the network itself is added to the
                trajectory, so to generate the network average without disturbance, it has to be set to 0 explicitly.
            additional_output ('none'): if set to 'activations', the history of the context layer activations is additionally returned.
            external_signal_variance: Define the sensory noise of the environment used for Bayesian inference. If None, input and prediction variance are set the same, if -1, model.external_signal_variance is used.
            add_BI_variance (True): Whether to [True] sample from the resulting distribution of the BI (or [False] just take the mean).
            hyp_prior (None): Which value of parameter H to use to determine reliance on predictions vs. reliance on input. None: Same value as for training is used.
            x_start (None): If given, the generated trajectory will start from this value, instead of from 0.

        Returns:
            results: The generated output trajectory.
            resultsv: The estimated variance for the output trajectory.
            resultsm: The estimated mean of the trajectory (is equal to results if add_variance_to_output==0).
            pred_error: The prediction error (only if ``external_input`` is set).
            weighted_pred_error: The variance-scaled prediction error (only if ``external_input`` is set).            
            u_h_history: The history of context layer activations during trajectory generation (only if ``additional_output`` == 'activations').
            resultspos: The posteriors of the Bayesian inference.
        """

        xp = cuda.get_array_module(self.initial_states.W)

        self.reset_current_output()

        stored_external_signal_variance = self.external_signal_variance
        stored_add_BI_variance = self.add_BI_variance
        stored_hyp_prior = self.hyp_prior

        self.add_BI_variance = add_BI_variance

        if not hyp_prior is None:
            self.hyp_prior = hyp_prior

        # cope with indirect definitions of init states which should be used
        if init_states is 'mean':
            # use the mean of all initial states
            init_states = xp.reshape(xp.mean(self.initial_states.W.array,axis=0), (1, self.num_c))
        elif init_states is 'best':
            # use the init state that best reproduces the trajectory

            # there is only one initial state (or they are all the same, i.e. zero)
            if self.initial_states.W.shape[0] == 1 or not xp.any(self.initial_states.W.array):
                init_states = self.initial_states.W.array
                self.used_is_idx = [0]
            else:
                self.used_is_idx = xp.zeros((len(external_input),), dtype=int)
                traj_idx = 0
                for extinp in external_input:
                    res = np.empty((self.initial_states.W.shape[0],), dtype=object)
                    resv = np.empty((self.initial_states.W.shape[0],), dtype=object)
                    resm = np.empty((self.initial_states.W.shape[0],), dtype=object)
                    pe = np.empty((self.initial_states.W.shape[0],), dtype=object)
                    wpe = np.empty((self.initial_states.W.shape[0],), dtype=object)
                    res_posterior = np.empty((self.initial_states.W.shape[0],), dtype=object)
                    is_idx=0
                    # try all available initial states
                    for inis in self.initial_states.W.array:
                        curr_extinp = xp.reshape(external_input[traj_idx,:], (1,external_input.shape[1]))

                        if self.gpu_id >= 0:
                            res[is_idx], resv[is_idx], resm[is_idx], pe[is_idx], wpe[is_idx], res_posterior[is_idx] = self.generate(cuda.to_gpu([inis]), num_steps, curr_extinp, add_variance_to_output, additional_output='none')
                        else:
                            res[is_idx], resv[is_idx], resm[is_idx], pe[is_idx], wpe[is_idx], res_posterior[is_idx] = self.generate([inis], num_steps, curr_extinp, add_variance_to_output, additional_output='none')
                        is_idx += 1

                    self.used_is_idx[traj_idx] = np.asarray([x[0].mean() for x in pe]).argmin()
                    traj_idx += 1
                init_states = self.initial_states.W.array[self.used_is_idx]
                self.used_is_idx = self.used_is_idx.tolist()
                print("Indices of chosen initial states: " + str(self.used_is_idx))

        if external_input is None:
            print("No external_input available.")
            num_generate = len(init_states)
        else:
            num_generate = len(external_input)

        results = xp.zeros((num_generate,num_steps*self.num_io),dtype=xp.float32)
        resultsv = xp.zeros((num_generate,num_steps*self.num_io),dtype=xp.float32)
        resultsm = xp.zeros((num_generate, num_steps * self.num_io), dtype=xp.float32)
        resultspos = xp.zeros((num_generate, num_steps * self.num_io), dtype=xp.float32)
        pred_error = xp.zeros((num_generate,(num_steps-1)),dtype=xp.float32)
        weighted_pred_error = xp.zeros((num_generate,(num_steps-1)),dtype=xp.float32)
        u_h_history = xp.zeros((num_generate,num_steps*self.num_c),dtype=xp.float32)

        # generate one trajectory for each given initial state matrix
        for i in range(num_generate):
            print("num_generate: " + str(i))
            self.reset_current_output()
            # use corresponding initial states
            is_idx = i%len(init_states)
            # set the initial state for this generation
            initSt = chainer.Variable(init_states[is_idx].reshape((1,self.num_c)))

            # set external input of first time step if existing, zero initialization otherwise
            if external_input is None: # proactive
                if not x_start is None:
                    x_init = x_start
                else:
                    x_init = xp.asarray(make_initial_state_zero(1, self.num_io))
            else: # reactive
                # if x_start is defined, it overwrites the usually taken first time step
                if not x_start is None:
                    x_init = x_start
                else:
                    x_init = xp.asarray(external_input[i,0:self.num_io].reshape((1,self.num_io)))
            print('Use x_init ' + str(x_init))

            # first trajectory generation step
            
            # external_signal_variance may be a single value or defined per time step
            try:
                to_set_external_signal_variance = external_signal_variance[0]
            except:
                to_set_external_signal_variance = external_signal_variance
            # valid values are None or >=0
            if to_set_external_signal_variance is None or to_set_external_signal_variance >= 0:
                self.external_signal_variance = to_set_external_signal_variance

            u_h, y, v = self(x_init, initSt)
            
            # posterior for paper plot
            current_posterior = self.current_x
            
            # optionally, add variance to output
            if add_variance_to_output is None:
                y_out = y.array + xp.sqrt(v.array) * xp.random.randn()
            else:
                y_out = y.array + xp.sqrt(add_variance_to_output) * xp.random.randn()

            results[i,0:self.num_io] = xp.reshape(y_out, (self.num_io,))
            resultsv[i,0:self.num_io] = xp.reshape(v.array, (self.num_io,))
            resultsm[i, 0:self.num_io] = xp.reshape(y.array, (self.num_io,))
            resultspos[i, 0:self.num_io] = xp.reshape(current_posterior.array, (self.num_io,))
            u_h_history[i,0:self.num_c] = u_h.array[0,:]

            # following trajectory generation steps
            for t in range(1, num_steps):

                # set external_signal_variance for this time step
                try:
                    to_set_external_signal_variance = external_signal_variance[t]
                    if to_set_external_signal_variance is None or to_set_external_signal_variance >= 0:
                        self.external_signal_variance = to_set_external_signal_variance
                except:
                    # use the same for all time steps
                    pass

                # determine next timestep of input according to closed-loop or open-loop control
                if external_input is None:
                    new_x = x_init # external_input is not used in network update
                else:
                    # new_x = xp.reshape(external_input[i,self.num_io*t:self.num_io*(t+1)], (1,self.num_io))
                    new_x = external_input[i,self.num_io*t:self.num_io*(t+1)].reshape((1,self.num_io))

                # generate forward step
                u_h, y, v = self(new_x, u_h)

                # posterior for paper plot
                current_posterior = self.current_x

                # optionally, add variance to output
                if add_variance_to_output is None:
                    y_out = y.array + xp.sqrt(v.array) * xp.random.randn()
                else:
                    y_out = y.array + xp.sqrt(add_variance_to_output) * xp.random.randn()

                results[i,self.num_io*t:self.num_io*(t+1)] = xp.reshape(y_out, (self.num_io,))
                resultsv[i,self.num_io*t:self.num_io*(t+1)] = xp.reshape(v.array, (self.num_io,))
                resultsm[i,self.num_io*t:self.num_io*(t+1)] = xp.reshape(y.array, (self.num_io,))
                resultspos[i, self.num_io * t:self.num_io * (t + 1)] = xp.reshape(current_posterior.array, (self.num_io,))
                u_h_history[i,self.num_c*t:self.num_c*(t+1)] = u_h.array[0,:]

            if not external_input is None:
                pred_error[i] = compute_prediction_error(xp.reshape(external_input[i,:],(-1,self.num_io)), xp.reshape(results[i,:],(-1,self.num_io)), axis=1)
                weighted_pred_error[i] = compute_weighted_prediction_error(xp.reshape(external_input[i,:],(-1,self.num_io)), xp.reshape(results[i,:],(-1,self.num_io)), xp.reshape(resultsv[i,:],(-1,self.num_io)), axis=1)

        self.reset_current_output()
        self.external_signal_variance = stored_external_signal_variance
        self.add_BI_variance = stored_add_BI_variance
        self.hyp_prior = stored_hyp_prior

        if additional_output == 'none':
            if external_input is None:
                return results, resultsv, resultsm
            else:
                return results, resultsv, resultsm, pred_error, weighted_pred_error, resultspos
        elif additional_output == 'activations':
            if external_input is None:
                return results, resultsv, resultsm, u_h_history
            else:
                return results, resultsv, resultsm, pred_error, weighted_pred_error, u_h_history, resultspos

class NetworkParameterSetting:
    '''
        Class that collects all training parameters.
    '''
    def __init__(self, epochs = 10, batch_size = 128):
        self.epochs = epochs
        self.batch_size = batch_size

    def set_network_type(self, type_str, parameters):
        self.type_str = type_str
        for key,val in parameters.items():
            print(key)
            # general network parameters
            if key == 'num_io':
                self.num_io = int(val)
                continue
            elif key == 'tau_io':
                self.tau_io = float(val)
                continue
            elif key == 'learn_init_states':
                self.learn_init_states = (str(val) == 'True')
                continue
            elif key == 'learn_tau':
                self.learn_tau = (str(val) == 'True')
                continue
            elif key == 'learn_weights':
                self.learn_weights = (str(val) == 'True')
                continue
            elif key == 'learn_bias':
                self.learn_bias = (str(val) == 'True')
                continue
            elif key == 'lr': # learning rate
                self.lr = float(val)
                continue
            elif key == 'num_classes':
                self.num_classes = int(val)
                continue
            elif key == 'hyp_prior':
                self.hyp_prior = val
                continue
            elif key == 'external_signal_variance':
                self.external_signal_variance = val
                continue
            # normalization factors for input data
            elif key == 'norm_offset':
                self.norm_offset = val
                continue
            elif key == 'norm_range':
                self.norm_range = val
                continue
            elif key == 'minmax':
                self.minmax = val
                continue
            
            # network-type-specific parameters
            if type_str == 'SCTRNN':
                if key == 'num_c': # number of context neurons
                    self.num_c = int(val)
                    continue
                elif key == 'tau_c': # update speed tau of context neurons
                    self.tau_c = float(val)
                    continue
                elif key == 'init_state_var':
                    self.init_state_var = np.float32(val)
                    continue
                elif key == 'init_state_init':
                    self.init_state_init = val
                    continue
                else:
                    print("Unknown parameter " + key + " for network type " + type_str)
                    continue

            else:
                print("Unknown network type " + type_str)

        # initialize initial states from string command
        if type(self.init_state_init) == str:
            self.init_state_var = np.repeat(self.init_state_var, self.num_c)

            if self.init_state_init == 'zero':
                self.init_state_init = make_initial_state_zero(self.num_classes, self.num_c)
            elif self.init_state_init == 'random':
                self.init_state_init = make_initial_state_random(self.num_classes, self.num_c)
            else:
                print('Unknown init_state_init value: ' + str(self.init_state_init))

        else:
            # for fixed initial states, no init state loss should be used, so init_state_var is not required!
            self.init_state_init = np.float32(self.init_state_init)
            print(self.init_state_init)

    def get_network_type(self):
        return self.type_str

    def get_parameter_string(self):
        members = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        string = ""
        for m in members:
            string += m + ": " + str(eval("self." + m)) + "\n"
        return string

# Functions for initializing arrays in the format that chainer expects

def make_initial_state_zero(batch_size, n_hidden):
    return np.array(np.zeros((batch_size, n_hidden)),dtype=np.float32)

def make_initial_state_random(batch_size, n_hidden):
    return np.array(np.random.uniform(-1,1,(batch_size, n_hidden)),dtype=np.float32)


# save networks

def save_network(dir_name, params = None, model = None, model_filename = "network"):
    if not params is None:
        with open(os.path.join(dir_name,"parameters.txt"),'a') as f:
            f.write(params.get_parameter_string())
            f.write("\n")
        f.close()

        with open((os.path.join(dir_name,"parameters.pickle")), 'wb') as f:
            pickle.dump(params, f)
            f.close()

        if not params.learn_init_states:
            try:
                np.save(dir_name+"/init_states.npy", np.array([model.initial_states]))
                print("store initial states of SCTRNN")
            except:
                try:
                    np.save(dir_name+"/init_states.npy", np.array([model.initial_states_sh]))
                    print("store initial states of SMTRNN")
                except:
                    print("Warning: Network type cannot be determined, do not store initial states...")

        if not params.learn_weights:
            weights_array = cuda.to_cpu(np.array([model.x_to_h, model.h_to_h, model.h_to_y, model.h_to_v]))
            np.save(dir_name+"/network_weights.npy", np.array([weights_array]))

        if not params.learn_bias:
            bias_array = cuda.to_cpu(np.array([model.x_to_h_bias, model.h_to_h_bias]))
            np.save(dir_name+"/network_bias.npy", bias_array)

    if not model is None:
        if not model_filename.endswith('.npz'):
            model_filename = model_filename + '.npz'
        serializers.save_npz(os.path.join(dir_name, model_filename), model)


def load_network(dir_name, network_type = 'SCTRNN', model_filename = 'network'):
    params = None
    model = None

    with open(dir_name+"/parameters.pickle", 'rb') as f:
        params = pickle.load(f)

    if network_type == 'SCTRNN':
        model = SCTRNN(params.num_io, params.num_c, params.tau_c, params.num_classes, init_state_init = params.init_state_init, init_state_learning = params.learn_init_states, weights_learning = params.learn_weights, bias_learning = params.learn_bias, hyp_prior = params.hyp_prior, external_signal_variance = params.external_signal_variance)

    if not model_filename.endswith('.npz'):
        model_filename = model_filename + '.npz'
    serializers.load_npz(os.path.join(dir_name, model_filename), model)

    if not params.learn_init_states:
        init_states = np.load(dir_name+"/init_states.npy")
        model.initial_states = init_states[0]

    if not params.learn_weights:
        network_weights = np.load(dir_name+"/network_weights.npy",allow_pickle=True)
        if network_type == 'SCTRNN':
            model.x_to_h = network_weights[0][0]
            model.h_to_h = network_weights[0][1]
            model.h_to_y = network_weights[0][2]
            model.h_to_v = network_weights[0][3]

    try:
        if not params.learn_bias:
            bias_weights = np.load(dir_name+"/network_bias.npy")
            if network_type == 'SCTRNN':
                model.x_to_h_bias = bias_weights[0]
                model.h_to_h_bias = bias_weights[1]
    except:
        pass

    return params, model

def compute_prediction_error(desired, predicted, axis=None, type = 'mse'):
    """
    Compute MSE between desired and predicted trajectory (time_steps x feature_dim).
    The trajectories are shifted against each other to account for the time
    shift due to prediction.
    axis: Along which axis should the result be averaged?
    type: 'mse' is standard mean square error.
    """
    if axis is None:
        axis = tuple(range(0, desired.ndim))

    if predicted.shape[0] == desired.shape[0]:
        predicted = predicted[:-1,:]

    if type == 'mse':
        return np.mean((desired[1:,:] - predicted)**2,axis=axis)
    # elif type == 'kl':
    #     return

def compute_weighted_prediction_error(desired, predicted, variance, axis=None):
    if axis is None:
        axis = tuple(range(0, desired.ndim))

    if predicted.shape[0] == desired.shape[0]:
        predicted = predicted[:-1,:]
    if variance.shape[0] == desired.shape[0]:
        variance = variance[:-1,:]

    return np.mean((desired[1:,:] - predicted)**2/variance,axis=axis)
