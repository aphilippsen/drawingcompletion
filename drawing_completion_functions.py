import chainer
import numpy as np
from inference import infer_initial_states_sctrnn
from utils.visualization import plot_multistroke

xp = np

def complete_drawing(model, params, input_traj, reduced_time_steps, is_selection_mode = 'best', x_start = None, hyp_prior = None,  high_sensory_variance = 50, plottingFile = None, add_BI_variance = True, inference_epochs = 2000, inference_network_path = '', gpu_id = 0):
    """
        - model: The trained SCTRNN model
        - param: Parameters of the model
        - input_traj: The given input trajectory for completion
        - reduced_time_steps: Number of time steps which should be visible to the network, input_traj will be accordingly preprocessed
        - is_selection_mode:
            'best' for using best initial state
            'inference' for inferring the initial state from the input_traj
            'mean' for using the mean of all training initial states
            'zero' for using zero initial states
            matrix (num_patterns x num_neurons): set the given values as initial states

        - x_start (optional): starting point for trajectory generation, otherwise set to 0
        - hyp_prior (optional): Prior parameter to be used for completion if given, otherwise inferred from model
        - high_sensory_variance (optional): which value for σ2_sensor should be assumed if no external input is available (affects amount of randomness of drawing in hypo-prior condition)
        - plottingFile (optional): whether to plot the generated trajectory, default False
        - add_BI_variance (optional): whether in each time step during generation noise with the amplitude of the estimated variance should be added to the trajectory
        - inference_epochs (optional): how much epochs to perform for inference, default 2000
        - inference_network_path (optional): define this if is_selection_mode=='inference' is set, but no new inference should be performed, instead an existing network file should be used


    """
    time_steps = int(input_traj.shape[1]/model.num_io)

    if hyp_prior is None:
        hyp_prior = model.hyp_prior
    if x_start is None:
        x_start = np.float32(np.tile(0, (1, model.num_io)))

    external_signal_var_testing = np.tile(model.external_signal_variance, (time_steps,))
    # after the initial time steps, external input becomes unavailable=unreliable!
    external_signal_var_testing[reduced_time_steps:] = high_sensory_variance


    # use input only until ... time steps
    delete_from = reduced_time_steps+1
    if gpu_id > -1:
        input_traj_zero = chainer.cuda.to_gpu(xp.copy(chainer.cuda.to_cpu(input_traj.reshape((-1,model.num_io)))))
    else:
        input_traj_zero = xp.copy(input_traj.reshape((-1,model.num_io)))
    for t in range(0, input_traj_zero.shape[0], time_steps):
        input_traj_zero[t+delete_from:t+time_steps,:] = 0
    input_traj_zero = input_traj_zero.reshape((input_traj.shape[0],-1))

    results_path = '.'
    if not isinstance(is_selection_mode, str):
        # assume that a fixed IS is given
        init_state = is_selection_mode

    elif is_selection_mode == 'mean':
        init_state = np.mean(model.initial_states.W.array,axis=0).reshape((1, model.num_c))

    elif is_selection_mode == 'zero':
        init_state = np.float32(np.zeros((1, model.num_c)))

    elif is_selection_mode == 'best':
        # init_state = np.reshape(model.initial_states.W.array[0,:], (1, model.num_c))
        
        input_traj_cut = input_traj_zero[:,0:model.num_io*reduced_time_steps]
        res, resv, resm, pe, wpe, respost = model.generate('best', reduced_time_steps, external_input = input_traj_cut, add_variance_to_output = 0, hyp_prior = 1, external_signal_variance = external_signal_var_testing, x_start = x_start)

        init_state = model.initial_states.W.array[model.used_is_idx,:] #xp.reshape(model.initial_states.W.array[model.used_is_idx,:], (len(model.used_is_idx), model.num_c))

    elif is_selection_mode == 'inference':
        if inference_network_path == '':
            # perform inference
            inferred_is, is_history, res, resm, results_path = infer_initial_states_sctrnn(params, model, input_traj, epochs=inference_epochs, start_is='mean', num_timesteps = reduced_time_steps, hyp_prior = hyp_prior, external_signal_variance = model.external_signal_variance, x_start = x_start)
            init_state = inferred_is.W.array
        else:
            # load network that has the inference results
            params, inf_model = load_network(inference_network_path, network_type = 'SCTRNN', model_filename="network-final.npz")
            if gpu_id >= 0:
                inf_model.to_gpu()
            init_state = inf_model.initial_states.W.array


    model.add_BI_variance = add_BI_variance

    # generation with the inferred initial states, first 30 timesteps with input, after that without input
    xp.random.seed(seed=1)
    res, resv, resm, pe, wpe, u_h_history, respost = model.generate(init_state, time_steps, external_input = input_traj_zero, add_variance_to_output = 0, hyp_prior = hyp_prior, external_signal_variance = external_signal_var_testing, additional_output='activations', x_start = x_start)


    if not plottingFile is None:
        for i in range(res.shape[0]):
            plot_multistroke(res[i,:].reshape((1,-1)), time_steps, plottingFile + "pattern-" + str(i) + ".png", model.num_io, given_part=reduced_time_steps)


    return init_state, res, results_path, u_h_history

