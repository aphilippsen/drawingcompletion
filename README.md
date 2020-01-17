# drawingcompletion
Computational model replicating the performance of how human children and chimpanzees complete drawings


Requirements:
-------------

The code runs with Python3 and is implemented using the CHAINER deep learning framework.

The following packages are required (install e.g. via pip):
chainer
matplotlib
numpy
dtw


Contents:
---------

1.-4. describe the main flow for how to generate results using the source code.

1. Data creation

data_generation/generate_training_data.py:
A GUI which allows you to draw (left mouse button) and store the created drawings (right mouse button).

data_generation/trajectory_multistroke_preprocessing.py:
Preprocesses the trajectories (equal length) and summarizes them in data sets to be used for training etc. Code has to be adapted when adding new drawings.
Generates a ...drawings.npy containing the drawings and ...drawings-classes.npy containing the corresponding class labels.

Existing drawing data can be found in the subfolders, containing the raw drawings and the preprocessed drawing data sets.

2. Training

run_training.py:
Main file to start the network training. Running it generates a folder in results/training/ with name data_set_name in with subfolders are created for each call of run_training.py named after the current date and time. Inside these folder, for each parameter condition one folder will be generated in which the trained network is stored.

3. Completion

run_completion.py:
Loads the trained networks (adjust the directory if necessary), then the networks are presented with the first 30% of the training trajectories and asked to complete them.

condition_directories and test_hyp_priors values have to be adjusted to decide which H_train and H_test condition to use.

Results are stored in a folder in results/completion/ determined by data_set_name variable, using the following structure
results/completion/[data_set_name]/[training run]/[training parameter]/[inference method]/test-[testing parameter]
For "inference" as inference method, the subfolder inference_networks contains the inference results (networks with updated initial states) of the X performed inferences for this network.

4. Evaluation

run_evaluation_training.py:
Evaluation of trained networks. Results are stored in the corresponding folder in results/training/

run_evaluation.py
Evaluation of the performance of the network for completing the trajectories.

run_evaluation_representation.py
Evaluation of the internal representations of the networks for completing the trajectories.


* Other source code files:

nets.py: the neural network model implementation

inference.py and drawing_completion_functions.py: completion implementation

utils/visualization.py: plotting code

utils/normalize.py: code for normalizing data dimension-wise

utils/distance_measures.py: dtw call

