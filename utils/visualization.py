from chainer.backends import cuda
import matplotlib
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import numpy as np

from utils.normalize import normalize

def plot_results(res, num_timesteps, save_filename, inputDim, twoDim = False, allInOne = False, title = None):

    res = cuda.to_cpu(res)

    if twoDim:
        plt.figure()
        for i in range(res.shape[0]):
            if not allInOne:
                plt.subplot(res.shape[0],1,i+1)
            toShow = np.reshape(res[i,:], (num_timesteps,inputDim))
            plt.plot(toShow[:,0], toShow[:,1])
        if not title is None:
            plt.title(title)
        plt.savefig(save_filename)
        plt.close()

    else:
        plt.figure()
        for i in range(res.shape[0]):
            if not allInOne:
                plt.subplot(res.shape[0],1,i+1)
            toShow = np.reshape(res[i,:], (num_timesteps,inputDim))
            for param in range(inputDim):
                plt.plot(np.arange(num_timesteps), toShow[:,param])
        if not title is None:
            plt.title(title)
        plt.savefig(save_filename)
        plt.close()

def plot_pca_activations(u_h_history, num_timesteps, save_filename, num_original_dims, num_classes):
    allContextActivations = np.reshape(cuda.to_cpu(u_h_history), (-1,num_original_dims))

    pca = PCA(n_components=2)
    pcaContextActivations = pca.fit_transform(StandardScaler().fit_transform(allContextActivations))
    Y, offset, dataRange, minmax = normalize(pcaContextActivations)

    pcaComp1 = 0
    pcaComp2 = 1
    if num_classes > 4:
        split = int(np.ceil(num_classes/2))
    else:
        split = len(u_h_history)
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, split))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(split):
        ax.plot(Y[i*(num_timesteps+1),pcaComp1], Y[i*(num_timesteps+1),pcaComp2], color=colors[i%split], label=str(i), marker='o', markersize=5)
        ax.plot(Y[i*(num_timesteps+1):(i+1)*(num_timesteps+1),pcaComp1], Y[i*(num_timesteps+1):(i+1)*(num_timesteps+1),pcaComp2], color=colors[i], label=str(i))
    plt.legend()
    fig.savefig(save_filename + "-a")
    plt.close()
    if len(u_h_history) > split:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(split,len(u_h_history)):
            ax.plot(Y[i*(num_timesteps+1),pcaComp1], Y[i*(num_timesteps+1),pcaComp2], color=colors[i%split], label=str(i), marker='o', markersize=5)
            ax.plot(Y[i*(num_timesteps+1):(i+1)*(num_timesteps+1),pcaComp1], Y[i*(num_timesteps+1):(i+1)*(num_timesteps+1),pcaComp2], color=colors[i%split], label=str(i))
        plt.legend()
        fig.savefig(save_filename + "-b")
        plt.close()

def plot_multistroke(res, num_timesteps, save_filename, input_dim, given_part = 0, title = ""):

    res = cuda.to_cpu(res)

    # 3rd dim == 1 means that this point is connected to the previous one, dim == 0 that not!
    plt.figure()
    for i in range(res.shape[0]):
        for t in range(1, num_timesteps):
            traj = res[i,:].reshape((num_timesteps, input_dim))
            if int(np.round(traj[t,2])) == 1:
                if t < given_part:
                    plt.plot(traj[t-1:t+1,0], traj[t-1:t+1,1], 'g')
                else:
                    plt.plot(traj[t-1:t+1,0], traj[t-1:t+1,1], 'k')
            else:
                if t < given_part:
                    plt.plot(traj[t-1:t+1,0], traj[t-1:t+1,1], '#11ff11')
                else:
                    plt.plot(traj[t-1:t+1,0], traj[t-1:t+1,1], 'lightgray')
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    plt.title(title)
    plt.savefig(save_filename)
    plt.close()

