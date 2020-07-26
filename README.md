# drawingcompletion: A predictive coding account for drawing ability of human children and chimpanzees #

## Overview ##

This code implements the completion of representational drawings based on trajectory learning, using the predictive coding paradigm.
As a model, stochastic continuous-time recurrent neural networks are combined with a Bayesian inference module which integrates sensory information with own predictions.
The effect of strong or weak reliance on priors is investigated to evaluate potential cognitive mechanisms for differences in the drawing behavior between human children and chimpanzees.

More details about the experiment and the scientific foundations can be found in the following papers:

* Anja Philippsen and Yukie Nagai. "A predictive coding account for cognition in human children and chimpanzees: A case study of drawing." IEEE Transactions on Cognitive and Developmental Systems (2020).

* Anja Philippsen and Yukie Nagai. "A predictive coding model of representational drawing in human children and chimpanzees." 2019 Joint IEEE 9th International Conference on Development and Learning and Epigenetic Robotics (ICDL-EpiRob). IEEE, 2019.

The model was also used in the following publication:

* Daniel Oliva, Anja Philippsen, and Yukie Nagai. "How development in the bayesian brain facilitates learning." 2019 Joint IEEE 9th International Conference on Development and Learning and Epigenetic Robotics (ICDL-EpiRob). IEEE, 2019.

## Documentation ##

The code runs with Python3 and is implemented using the [CHAINER deep learning framework](https://chainer.org/).

### Installation ###

The following packages are required (install e.g. via pip):

* chainer
* matplotlib
* numpy
* scikit-learn
* pandas
* dtw

### How to run ###

The code can be used to generate results for the  given training trajectories or other type of training data. Steps 1 to 4 describe the main flow for generating results.

#### 1. Data creation ####

**data_generation/generate_training_data.py:**
A GUI which allows you to draw (left mouse button) and store the created drawings (right mouse button).

**data_generation/trajectory_multistroke_preprocessing.py:**
Script that was used to preprocess the trajectories to equal length and to summarize them in a data sets file to be used for training etc. Code has to be adapted when adding new drawings.
Two *npy* files are generated, one containing the trajectories and one containing the corresponding class labels.

Drawing sets which are ready to use can be found in *data_generation/drawing-data-sets/*.

#### 2. Training ####

**run_training.py:**
Main file to start the network training. Running it generates a folder in *results/training/* with name *data_set_name* in with subfolders are created for each call of run_training.py named after the current date and time. Inside these folder, for each parameter condition one folder will be generated in which the trained network is stored.

#### 3. Completion ####

**run_completion.py:**
Loads the trained networks (adjust the directory if necessary), then the networks are presented with the first part of the training trajectories and asked to complete them. The values *condition_directories* and *test_hyp_priors* have to be adjusted to decide which parameter condition to use.

Results are stored in a folder in *results/completion/* determined by *data_set_name* variable, using the following structure
*results/completion/[data_set_name]/[training run]/[training parameter]/[inference method]/test-[testing parameter]*
For *inference* as inference method, the subfolder *inference_networks* contains the inference results (networks with updated initial states) for this network.

#### 4. Evaluation ####

**run_evaluation_training.py:**
Evaluation of trained networks. Results are stored in the corresponding folder in *results/training/*.

**run_evaluation.py**
Evaluation of the performance of the network for completing the trajectories.
(By default, evaluation is done assuming that the same value of H is used for training and testing, results are stored in the *evaluation* folder as *inference-corresponding*.)

**run_evaluation_representation.py**
Evaluation of the internal representations of the networks for completing the trajectories. This evaluation is performed individually for each H parameter value, and stored in the *evaluation* folder as *inference-1* etc.

**run_evaluation_attractors.py**
Completion is performed using initial states linearly interpolated between the trained initial states. Results are stored in the *evaluation* directory, in a subfolder determined via the *plot_dir* variable.

#### Other source code files ####

**nets.py:**
The implementation of the recurrent neural network with Bayesian inference.

**inference.py** and **drawing_completion_functions.py:**
Implementation of the completion of trajectories via backpropagation.

**utils/visualization.py:**
Code for plotting the trajectories.

**utils/normalize.py:**
Code for normalizing the trajectory data dimension-wise.

**utils/distance_measures.py:**
Measuring distance between trajectories via Dynamic Time Warping.

**error_statistics_analysis.R**
R script for testing statistical significances of the error differences in different H conditions.

