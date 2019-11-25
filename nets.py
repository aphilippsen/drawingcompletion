import os
import sys

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

import pdb

# RECURRENT NEURAL NETWORK DEFINITIONS

# Stochastic Continuous-Time Recurrent Neural Network

class SCTRNN(chainer.Chain):
    def __init__(self, num_io, num_context, tau_context, num_classes, init_state_init = None, init_state_learning = True, weights_learning = True, bias_learning = True, tau_learning = False, external_contrib = 1, aberrant_sensory_precision = 0, excitation_bias = 0.05, rec_conn_factor = 0, variance_integration_mode = 1, hyp_prior = 1, external_signal_variance = None, pretrained_model = None):
        """
        Creates a S-CTRNN with input/output layer size num_io, and context layer
        size num_context.
        tau_context: Time scale (initialization) (if large: slow update;
                    if small: fast update of context neuron activations)
        num_classes: Number of different attractors that the network can
                    maximally represent (=> number of initial states)
        init_state_init (None): Numpy 2-d matrix for initializing the initial
                    states; must be of size (num_classes x num_context). If None
                    initialized with zeros.
        external_contrib (1): how much the network uses the
                    external signal as opposed to its own prediction (valid
                    range: 0 <= external_contrib <= 1, for learning to succeed
                    external_contrib must be > 0)
        init_state_learning (True): if True initial states are updated during
                    optimization, if False they are fixed to the initial values
        tau_learning (False): if True, the tau factor is updated during
                    optimization, if False tau is fixed
        pretrained_model (None): If given, initial states and network weights
                    are copied from the other (previously trained) SCTRNN instance.
        aberrant_sensory_precision (0): If != 0, the network under- or overestimates
                    its estimated variance during training


        external_signal_variance (None): If (None), external signal variance is set
                    identical to the estimated variance, otherwise to fixed value (per
                    dimension)
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
            b_init = chainer.initializers.Normal(scale = excitation_bias)

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

        # how much the current input drives the network (1-external_contrib: how much the network uses its own output)
        self.external_contrib = external_contrib
        # if external_contrib < 1, instead the network's own estimation is used
        # with or without adding noise from own estimated variance?
        self.apply_estimated_variance = variance_integration_mode
        self.aberrant_sensory_precision = aberrant_sensory_precision
        self.excitation_bias = excitation_bias

        # store which initial states were determined when using init_state policy 'best' in generate()
        self.used_is_idx = []

        self.reset_current_output()

        self.gpu_id = -1

        self.bias_learning = bias_learning

        # Whether to [True] sample from the resulting distribution of the BI (or [False] just take the mean)
        self.add_BI_variance = True

        self.rec_connectivity = L.Parameter(np.ones((num_context, num_context), dtype = np.float32))
        # number of other neurons that one neuron may have connections
        # =1: only connection to themselves, not to other neurons
        if rec_conn_factor <= 0:
            self.rec_conn_factor = num_context
        else:
            self.rec_conn_factor = rec_conn_factor

        if self.rec_conn_factor < num_context:
            for i in range(num_context):
                for j in range(num_context):
                    if np.abs(i-j) > self.rec_conn_factor - 1:
                        self.rec_connectivity.W.array[i, j] = 0.1

        if not self.bias_learning and self.rec_conn_factor < num_context:
            print("Warning: altered connectivity and bias_learning=False are not yet working together")

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
        self.rec_connectivity.to_gpu(device)
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
        self.rec_connectivity.to_cpu()
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
        Reset current_y and current_v, required if external_contrib < 1.
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

        x = chainer.Variable(x_data)

        if not self.current_y.array is None:
            #for i in range(x.shape[0]):


            if self.apply_estimated_variance == 0:
                # x'_{t+1} = \chi * x_{t+1} + (1-\chi) * y_t
                x = self.external_contrib * x + (1 - self.external_contrib) * self.current_y
            elif self.apply_estimated_variance == 1:
                # print(self.external_contrib)
                # x'_{t+1} = \chi * x_{t+1} + (1-\chi) * N(y_t, v_t)
                x = self.external_contrib * x + (1 - self.external_contrib) * (self.current_y + chainer.functions.sqrt(self.current_v) * xp.random.randn(self.current_v.shape[0], self.current_v.shape[1]))
            elif self.apply_estimated_variance == 2:

                # infere two distributions
                # for i in range(0,x.shape[0]):
                #     pred_var_1 = self.current_v.array[0,0] * (self.hyp_prior)
                #     pred_var_2 = self.current_v.array[0,1] * (self.hyp_prior)
                #     pred_mean_1 = self.current_y.array[0,0]
                #     pred_mean_2 = self.current_y.array[0,1]
                #
                #     if external_signal_variance == None:
                #         input_var_1 = pred_var_1
                #         input_var_2 = pred_var_2
                #     else:
                #         input_var_1 = 0.003#self.external_signal_variance # pred_var_1  # x direction line width
                #         input_var_2 = 0.003#self.external_signal_variance # pred_var_2  # y direction line width
                #
                #     input_1 = x.array[i,0]
                #     input_2 = x.array[i,1]
                #
                #     sigma_1 = xp.sqrt((input_var_1 * pred_var_1) / (input_var_1 + pred_var_1))
                #     sigma_2 = xp.sqrt((input_var_2 * pred_var_2) / (input_var_2 + pred_var_2))
                #
                #     mu_1 = xp.power(sigma_1, 2) * (((pred_mean_1 / pred_var_1) + (input_1 / input_var_1)))
                #     mu_2 = xp.power(sigma_2, 2) * (((pred_mean_2 / pred_var_2) + (input_2 / input_var_2)))
                #
                #     x.array[i,0] = xp.random.normal(mu_1, sigma_1, 1)[0]
                #     x.array[i,1] = xp.random.normal(mu_2, sigma_2, 1)[0]
                #

                # if x.shape[0] > 1:
                #     import pdb
                #     pdb.set_trace()

                # test_input_var = chainer.Variable(xp.reshape(xp.asarray(xp.float32([0.1,0.05,0.1])), (1,3)))
                # test_input = chainer.Variable(xp.tile(xp.asarray(xp.float32([0.3])), (1,3)))
                # test_pred_var = chainer.Variable(xp.reshape(xp.asarray(xp.float32([0.1,0.1,0.05])), (1,3)))
                # test_pred = chainer.Variable(xp.tile(xp.asarray(xp.float32([0.8])), (1,3)))
                # sigma_test = xp.sqrt(xp.divide(xp.multiply(test_input_var.array, test_pred_var.array), (test_input_var.array + test_pred_var.array)))
                # mu_test = xp.power(sigma_test, 2) * (xp.divide(test_pred.array, test_pred_var.array) + xp.divide(test_input.array, test_input_var.array))
                # # => mu_test: array([[0.5500001, 0.4666667, 0.6333334]], dtype=float32)
                # # => sigma_test: array([[0.22360681, 0.1825742 , 0.1825742 ]], dtype=float32)


                if self.hyp_prior > 1000:
                    # (self.hyp_prior - 1000) is a value a
                    pred_var = (self.hyp_prior-1000) * chainer.Variable(xp.tile(xp.asarray(xp.float32([1])), (x.shape[0],x.shape[1])))
                # testing condition, compare ICDL19_Daniel_plots.py
                elif self.hyp_prior == -555:
                    # Set prior variance equal to input variance to have always equal contribution
                    pred_var = chainer.Variable(xp.tile(xp.asarray(xp.float32([self.external_signal_variance])), (x.shape[0],x.shape[1])))
                else:
                    pred_var = self.current_v * (self.hyp_prior)
                # minimum value > 0 to avoid division-by-zero

                if not xp.any(pred_var.array):
                    # if predicted variance is exactly zero (which usually doesn't happen)
                    print("Predicted variance is zero!!!!")
                    pred_var.array = xp.tile(np.float32(0.00001), pred_var.array.shape)
                pred_mean = self.current_y


                if self.external_signal_variance is None:
                    input_var = pred_var
                else:
                    if self.hyp_prior > 1000:
                        input_var = 1 / pred_var
                    else:
                        input_var = chainer.Variable(xp.tile(xp.asarray(xp.float32([self.external_signal_variance])), (x.shape[0],x.shape[1])))
                input_mean = x

                # print("set variances: input " + str(input_var) + " and pred " + str(pred_var))

                # standard deviation of BI signal
                sigma_BI = xp.sqrt(xp.divide(xp.multiply(input_var.array, pred_var.array), (input_var.array + pred_var.array)))
                # mean of BI signal
                mu_BI = xp.power(sigma_BI, 2) * (xp.divide(pred_mean.array, pred_var.array) + xp.divide(input_mean.array, input_var.array))

                # # correct, but very inefficient on GPU
                # for dim in range(x.array.shape[1]):
                #     x.array[i,dim] = xp.random.normal(mu_BI[0,dim], sigma_BI[0,dim], 1)[0]

                # For plotting

                ##CPU###
                """"
                value_range = np.reshape(np.linspace(-1.5, 1.5, 1000), (1, 1000))
                input_pop = pop.transform_to_population_coding(xp.reshape(xp.copy(input_mean.array[0,0]), (1,1)), 1000, xp.copy(input_var.array[0,0]), -1.5, 1.5)
                prior_pop = pop.transform_to_population_coding(xp.reshape(xp.copy(pred_mean.array[0,0]), (1,1)), 1000, xp.copy(pred_var.array[0,0]), -1.5, 1.5)
                posterior_pop = pop.transform_to_population_coding(xp.reshape(xp.copy(mu_BI[0,0]), (1,1)), 1000, xp.copy(sigma_BI[0,0]) ** 2, -1.5, 1.5)

                plt.figure()
                plt.plot(value_range[0, :], input_pop[0, :], 'r')
                plt.plot(value_range[0, :], prior_pop[0, :], 'b')
                plt.plot(value_range[0, :], posterior_pop[0, :], 'g')
                #plt.show()
                plt.savefig('myfig')

                pdb.set_trace()

                value_range = np.reshape(np.linspace(-1.5, 1.5, 1000), (1, 1000))
                input_pop = pop.transform_to_population_coding(xp.reshape(xp.copy(input_mean.array[0, 1]), (1, 1)), 1000, xp.copy(input_var.array[0, 1]), -1.5, 1.5)
                prior_pop = pop.transform_to_population_coding(xp.reshape(xp.copy(pred_mean.array[0, 1]), (1, 1)), 1000, xp.copy(pred_var.array[0, 1]), -1.5, 1.5)
                posterior_pop = pop.transform_to_population_coding(xp.reshape(xp.copy(mu_BI[0, 1]), (1, 1)), 1000, xp.copy(sigma_BI[0, 1]) ** 2, -1.5, 1.5)

                plt.figure()
                plt.plot(value_range[0, :], input_pop[0, :], 'r')
                plt.plot(value_range[0, :], prior_pop[0, :], 'b')
                plt.plot(value_range[0, :], posterior_pop[0, :], 'g')
                #plt.show()
                plt.savefig('myfig')
                """
                ###CPU###

                if self.add_BI_variance:
                    if cuda.get_array_module(self.initial_states.W) == np:
                        x.array = mu_BI + sigma_BI * xp.float32(xp.random.randn(sigma_BI.shape[0], sigma_BI.shape[1]))
                    else:
                        x.array = mu_BI + sigma_BI * (xp.random.randn(sigma_BI.shape[0], sigma_BI.shape[1], dtype=xp.float32))
                else:
                    x.array = mu_BI

        self.current_x = x

        if u_h is None:
            u_h = self.initial_states()[self.classes] # np.zeros((2,self.num_c),dtype=np.float32))

        # forward context mapping
        h = F.tanh(u_h)
        # u_h = 1/self.tau_c() * (self.x_to_h(x) + self.h_to_h(h)) + (1 - 1/self.tau_c()) * u_h

        # recurrent connections
        if self.bias_learning:
            # u_h = chainer.functions.scale(self.x_to_h(x) + self.h_to_h(h), 1/self.tau_c()) + chainer.functions.scale(u_h, (1 - 1/self.tau_c()))
            # recurrent_connections = self.h_to_h(h)
            # import pdb; pdb.set_trace()
            recurrent_connections = F.transpose(F.tensordot(self.h_to_h.W * self.rec_connectivity(), F.transpose(h), axes=1)) + F.tile(self.h_to_h.b, (h.shape[0], 1)) #F.reshape(self.h_to_h.b,(h.shape[0],self.num_c))
            # print(str(recurrent_connections.shape) +  " vs. " + str(self.h_to_h(h).shape))
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


    def generate(self, init_states, num_steps, external_contrib = 1, external_input = None, epsilon_disturbance = None, plotting = False, additional_output='none', external_signal_variance = -1, add_BI_variance = True, hyp_prior = None, x_start = None):
        """
        Use the trained model and the given ``init_states`` to generate a sequence of length ``num_steps`` via closed- or open-loop control.

        Args:
            init_states: A numpy array of initial state for initializing the network activations
            num_steps: Integer number of steps to generate
            external_contrib: Ratio of contribution of the external signal
            external_input: A numpy array of external input signals for each initial state. if ``None``, external_contrib = 0 is set.
                One input signal per initial state should be provided. If there are more input signals than initial states, it
                should be a multiple of the initial states, then the initial states array gets copied to match the length of ``external_input``.
            epsilon_disturbance: Float value, indicating the amount of noise (as standard deviation of a Gaussian distribution) added to the
                network estimated average in each time step. If undefined, the variance estimated by the network itself is added to the
                trajectory, so to generate the network average without disturbance, it has to be set to 0 explicitly.
            plotting: Boolean indicating whether the result should be plotted or not.
            additional_output: if 'none', no effect, if 'activations', the history of the context layer activations is additionally returned.

        Returns:
            The generated output trajectory.
            The estimated variance of the output trajectory.
            The history of context layer activations during trajectory generation (only if ``additional_output`` == 'activations')
            The prediction error (only if ``external_input`` is set).
            The variance-scaled prediction error (only if ``external_input`` is set).
        """

        # external_input = np.float32(np.zeros((1,270)))
        # external_input = None

        xp = cuda.get_array_module(self.initial_states.W)

        self.reset_current_output()

        stored_external_contrib = self.external_contrib
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
                    respost = np.empty((self.initial_states.W.shape[0],), dtype=object)
                    is_idx=0
                    for inis in self.initial_states.W.array:
                        curr_extinp = xp.reshape(external_input[traj_idx,:], (1,external_input.shape[1]))

                        #sarray([inis]))cuda.to_gpu(xp.asarray([inis]))p.asarray([self.initial_states.W.array[0]])
                        if self.gpu_id >= 0:
                            res[is_idx], resv[is_idx], resm[is_idx], pe[is_idx], wpe[is_idx], respost[is_idx] = self.generate(cuda.to_gpu([inis]), num_steps, external_contrib, curr_extinp, epsilon_disturbance, plotting, additional_output='none')
                        else:
                            res[is_idx], resv[is_idx], resm[is_idx], pe[is_idx], wpe[is_idx], respost[is_idx] = self.generate([inis], num_steps, external_contrib, curr_extinp, epsilon_disturbance, plotting, additional_output='none')
                        is_idx += 1

                    #print("Prediction errors for different initial states (trajectory " + str(traj_idx) + "): " + str([np.mean(x[0]) for x in pe]))
                    self.used_is_idx[traj_idx] = np.asarray([x[0].mean() for x in pe]).argmin() #int(xp.argmin([xp.mean(x[0]) for x in pe]))
                    traj_idx += 1
                init_states = self.initial_states.W.array[self.used_is_idx]
                self.used_is_idx = self.used_is_idx.tolist()
                print("Indices of chosen initial states: " + str(self.used_is_idx))

        # Does make sense but at this point it gets overwritten later... so just pass a warning for the moment
        if external_input is None:
            import warnings
            warnings.warn("No external_input available. Set external_contrib to zero / or external_signal_variance high!")
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        #     if self.apply_estimated_variance == 1:
        #         self.external_contrib = 0
#            elif self.apply_estimated_variance == 2:
#                self.external_signal_variance = 100

            num_generate = len(init_states)
        else:
            num_generate = len(external_input)

        # if len(external_input) > len(init_states):
        #     assert(len(external_input)%len(init_states) == 0)
        #     repeat_factor = int(len(external_input)/len(init_states))
        #     init_states = np.array([init_states,]*repeat_factor)

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
            # if external_input is None:
            #     is_idx = i
            # else:
            is_idx = i%len(init_states)

            # set the initial state for this generation
            # initSt = chainer.Variable(xp.reshape(xp.asarray(init_states[is_idx]), (1,self.num_c)))
            initSt = chainer.Variable(init_states[is_idx].reshape((1,self.num_c)))
            # initSt = init_states[is_idx].reshape(1, self.num_c)

            try:
                self.external_contrib = external_contrib[0]
            except:
                self.external_contrib = external_contrib

            # set external input if existing, zero initialization otherwise
            if external_input is None or self.external_contrib == 0:
                if not x_start is None:
                    x_init = x_start
                else:
                    x_init = xp.asarray(make_initial_state_zero(1, self.num_io))
            else:
                # if x_start is defined, it overwrites the usually taken first time step
                if not x_start is None:
                    x_init = x_start
                else:
                    x_init = xp.asarray(external_input[i,0:self.num_io].reshape((1,self.num_io)))
            print('Use x_init ' + str(x_init))

            # if external_signal_variance is None or external_signal_variance >= 0:
                # only set it here if there is a valid value defined for it
            try:
                to_set_external_signal_variance = external_signal_variance[0]
            except:
                to_set_external_signal_variance = external_signal_variance
            if to_set_external_signal_variance is None or to_set_external_signal_variance >= 0:
                self.external_signal_variance = to_set_external_signal_variance

            # perform first trajectory generation step
            u_h, y, v = self(x_init, initSt)
            # optionally, add variance to output

            # posterior for paper plot
            current_posterior = self.current_x

            if epsilon_disturbance is None:
                y_out = y.array + xp.sqrt(v.array) * xp.random.randn()
            else:
                y_out = y.array + xp.sqrt(epsilon_disturbance) * xp.random.randn()

            results[i,0:self.num_io] = xp.reshape(y_out, (self.num_io,))
            resultsv[i,0:self.num_io] = xp.reshape(v.array, (self.num_io,))
            resultsm[i, 0:self.num_io] = xp.reshape(y.array, (self.num_io,))
            resultspos[i, 0:self.num_io] = xp.reshape(current_posterior.array, (self.num_io,))
            u_h_history[i,0:self.num_c] = u_h.array[0,:]

            # following trajectory generation steps
            for t in range(1, num_steps):

                try:
                    self.external_contrib = external_contrib[t]
                except:
                    pass

                try:
                    to_set_external_signal_variance = external_signal_variance[t]
                    if to_set_external_signal_variance is None or to_set_external_signal_variance >= 0:
                        self.external_signal_variance = to_set_external_signal_variance
                except:
                    pass


                # determine next timestep of input according to closed-loop or open-loop control
                if external_input is None  or self.external_contrib == 0:
                    new_x = x_init # external_input is not used in network update
                else:
                    new_x = xp.reshape(external_input[i,self.num_io*t:self.num_io*(t+1)], (1,self.num_io))

                # generate forward step
                u_h, y, v = self(new_x, u_h)

                # posterior for paper plot
                current_posterior = self.current_x

                # optionally, add variance to output
                if epsilon_disturbance is None:
                    y_out = y.array + xp.sqrt(v.array) * xp.random.randn()
                else:
                    y_out = y.array + xp.sqrt(epsilon_disturbance) * xp.random.randn()

                results[i,self.num_io*t:self.num_io*(t+1)] = xp.reshape(y_out, (self.num_io,))
                resultsv[i,self.num_io*t:self.num_io*(t+1)] = xp.reshape(v.array, (self.num_io,))
                resultsm[i,self.num_io*t:self.num_io*(t+1)] = xp.reshape(y.array, (self.num_io,))
                resultspos[i, self.num_io * t:self.num_io * (t + 1)] = xp.reshape(current_posterior.array, (self.num_io,))
                u_h_history[i,self.num_c*t:self.num_c*(t+1)] = u_h.array[0,:]

            if plotting:
                if self.num_io == 1:
                    x = np.arange(len(results[i]), dtype=np.float32)
                    plt.figure()
                    plt.plot(np.reshape(x, (len(results[i]), 1)), results[i])
                    plt.title("Pattern " + str(i) + ", generated mean")
                    plt.show(block=False)

                    x = np.arange(len(resultsv[i]), dtype=np.float32)
                    plt.figure()
                    plt.plot(np.reshape(x[1:], (len(resultsv[i][1:]), 1)), resultsv[i][1:])
                    plt.title("Pattern " + str(i) + ", variance")
                    plt.show(block=False)

                elif self.num_io == 2:
                    plt.figure()
                    plt.plot(results[i][1:,0], results[i][1:,1])
                    plt.title("Pattern " + str(i) + ", generated mean")
                    plt.show(block=False)

                    x = np.arange(len(resultsv[i]), dtype=np.float32)
                    plt.figure()
                    leaveOutTimeSteps = int(num_steps * 0.05)
                    plt.plot(np.reshape(x[leaveOutTimeSteps:], (len(resultsv[i][leaveOutTimeSteps:,0]), 1)), resultsv[i][leaveOutTimeSteps:,0], label="dim1")
                    plt.plot(np.reshape(x[leaveOutTimeSteps:], (len(resultsv[i][leaveOutTimeSteps:,0]), 1)), resultsv[i][leaveOutTimeSteps:,1], label="dim2")
                    plt.title("Pattern " + str(i) + ", variance")
                    plt.show(block=False)
                    plt.legend()

            if not external_input is None:
                pred_error[i] = compute_prediction_error(xp.reshape(external_input[i,:],(-1,self.num_io)), xp.reshape(results[i,:],(-1,self.num_io)), axis=1) # np.mean((results[i] - external_input[i])**2,axis=1) # chainer.functions.mean_squared_error(results[i], external_input[i]).array
                weighted_pred_error[i] = compute_weighted_prediction_error(xp.reshape(external_input[i,:],(-1,self.num_io)), xp.reshape(results[i,:],(-1,self.num_io)), xp.reshape(resultsv[i,:],(-1,self.num_io)), axis=1)
                if plotting:
                    plt.figure()
                    plt.plot(np.arange(0,len(pred_error[i])), pred_error[i])
                    plt.title("prediction error")
                    plt.show(block = False)
                    plt.figure()
                    plt.plot(np.arange(0,len(weighted_pred_error[i])), weighted_pred_error[i])
                    plt.title("weighted prediction error")
                    plt.show(block = False)

        self.reset_current_output()
        self.external_contrib = stored_external_contrib
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
        Class that collects all training parameters for different network
        types.
        TODO: load/save parameter settings
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
            # elif key == 'teaching_signal_train':
            #     self.teaching_signal_train = float(val)
            #     continue
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
            # elif key == 'alpha_init':
            #     self.alpha_init = float(val)
            #     continue
            # elif key == 'momentum_init':
            #     self.momentum_init = float(val)
            #     continue
            elif key == 'lr': # learning rate
                self.lr = float(val)
                continue
            elif key == 'num_classes':
                self.num_classes = int(val)
                continue
            elif key == 'training_external_contrib':
                self.training_external_contrib = val
                continue
            elif key == 'init_state_loss_scaling': # scaling factor for the contribution of the variance criterion on init state learning (usually 1)
                self.init_state_loss_scaling = float(val)
                continue
            elif key == 'aberrant_sensory_precision':
                self.aberrant_sensory_precision = float(val)
                continue
            elif key == 'excitation_bias':
                self.excitation_bias = np.float32(val)
                continue
            elif key == 'rec_connection_factor':
                self.rec_connection_factor = int(val)
            elif key == 'variance_integration_mode':
                self.variance_integration_mode = int(val)
            elif key == 'hyp_prior':
                self.hyp_prior = val
            elif key == 'external_signal_variance':
                self.external_signal_variance = val

            elif key == 'norm_offset':
                self.norm_offset = val
                continue
            elif key == 'norm_range':
                self.norm_range = val
                continue
            elif key == 'minmax':
                self.minmax = val
                continue
            elif key == 'teaching_order':
                self.teaching_order = val
                continue
            elif key == 'teach_duration':
                self.teach_duration = val
                continue
            elif key == 'bias_likelihood':
                self.bias_likelihood = val

            # network-type-specific parameters
            if type_str == 'CTRNN' or type_str == 'SCTRNN':
                if key == 'num_c':
                    self.num_c = int(val)
                    continue
                elif key == 'tau_c':
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

            elif type_str == 'MTRNN' or type_str == 'SMTRNN':
                if key == 'num_fh':
                    self.num_fh = int(val)
                    continue
                elif key == 'tau_fh':
                    self.tau_fh = float(val)
                    continue
                elif key == 'num_sh':
                    self.num_sh = int(val)
                    continue
                elif key == 'tau_sh':
                    self.tau_sh = float(val)
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
            elif type_str == 'PB_SMTRNN':
                if key == 'num_c':
                    self.num_c = int(val)
                    continue
                elif key == 'num_pb':
                    self.num_pb = int(val)
                elif key == 'tau_c':
                    self.tau_c = float(val)
                    continue
                elif key == 'tau_pb':
                    self.tau_pb = float(val)
                    continue
                elif key == 'init_state_var':
                    self.init_state_var = np.float32(val)
                    continue
                elif key == 'init_state_init':
                    self.init_state_init = val
                    continue
                elif key == 'pb_init':
                    self.pb_init = val
                    continue
                else:
                    print("Unknown parameter " + key + " for network type " + type_str)
                    continue
            else:
                print("Unknown network type " + type_str)


        if type(self.init_state_init) == str:
            if self.type_str == 'SCTRNN' or self.type_str == 'PB_SMTRNN':
                num_neurons = self.num_c
            elif self.type_str == 'SMTRNN' or network_type == 'MTRNN':
                num_neurons = self.num_sh

            self.init_state_var = np.repeat(self.init_state_var, num_neurons)

            if self.init_state_init == 'zero':
                self.init_state_init = make_initial_state_zero(self.num_classes, num_neurons)
            elif self.init_state_init == 'random':
                self.init_state_init = make_initial_state_random(self.num_classes, num_neurons)
            else:
                print('Unknown init_state_init value: ' + str(self.init_state_init))

            if self.type_str == 'PB_SMTRNN':
                if self.pb_init == 'zero':
                    self.pb_init = np.array(np.zeros((self.num_classes, self.num_pb)),dtype=np.float32)
                elif self.pb_init == 'random':
                    self.pb_init = np.array(np.random.uniform(-1,1,(self.num_classes, self.num_pb)),dtype=np.float32)
                else:
                    print('Unknown init_state_init value: ' + str(self.pb_init))

        else:
            # for fixed initial states, no init state loss should be used, so init_state_var is not required!
            self.init_state_init = np.float32(self.init_state_init)
            print(self.init_state_init)


    def get_network_type(self):
        return self.type_str


    def print_parameters(self):
        #members = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        #for m in members:
        #    print(m + ": " + str(eval("self." + m)))
        print(vars(self))
        return str(vars(self))



# Functions for initializing arrays in the format that chainer expects

def make_initial_state_zero(batch_size, n_hidden):
    return np.array(np.zeros((batch_size, n_hidden)),dtype=np.float32)

def make_initial_state_random(batch_size, n_hidden):
    return np.array(np.random.uniform(-1,1,(batch_size, n_hidden)),dtype=np.float32)



# store

# saveDir = "results/"+time.asctime(time.localtime(time.time()))+"/"
# pathlib.Path(saveDir).mkdir(parents=True, exist_ok=True)
def save_network(dir_name, params = None, model = None, model_filename = "network"):
    if not params is None:
        with open(os.path.join(dir_name,"parameters.txt"),'a') as f:
            f.write(params.print_parameters())
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
        model = SCTRNN(params.num_io, params.num_c, params.tau_c, params.num_classes, init_state_init = params.init_state_init, external_contrib = params.training_external_contrib, init_state_learning = params.learn_init_states, weights_learning = params.learn_weights, bias_learning = params.learn_bias, aberrant_sensory_precision = params.aberrant_sensory_precision, excitation_bias = params.excitation_bias, rec_conn_factor = params.rec_connection_factor, variance_integration_mode = params.variance_integration_mode, hyp_prior = params.hyp_prior, external_signal_variance = params.external_signal_variance)
        print("set variance integration mode to " + str(params.variance_integration_mode))

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
                #cuda.get_array_module(model.h_to_h_bias.b.data)
                model.x_to_h_bias = bias_weights[0]
                model.h_to_h_bias = bias_weights[1]
                #cuda.get_array_module(model.h_to_h_bias.b.data)
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
