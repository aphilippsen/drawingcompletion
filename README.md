# drawingcompletion
Computational model of drawing completion in human children and chimpanzees


Contents:

train_drawings.py:
Main file to start the network training.
Running it generates a folder in results/training/ with the current date and time. Inside this folder, one folder will be generated containing the trained network for each of the parameter conditions, and the initial weights are stored that were used for all networks in all conditions.

complete_drawings.py:
Loads the trained networks and the beginning of the trajectory data and asks network to complete via inference process.

data: drawing data

model: the neural network model implementation

visualization: plotting code

analysis: code for evaluating the trained networks

preprocessing: for recording drawings from a touchscreen (or via mouse input), and saving them, and for combining multiple drawings into a training data set


