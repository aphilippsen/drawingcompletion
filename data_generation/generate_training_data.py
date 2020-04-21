import numpy as np
import os
import matplotlib.pyplot as plt
from chainer import cuda
import scipy.interpolate
from drawing_interface import DrawingGenerationInterface, process_drawing, process_stroke

def get_new_points(X,Y,Z,steps):
    array = []
    intervals = np.linspace(0,1,steps)
    w=[0,1]
    x = np.interp(intervals,w,X)
    y = np.interp(intervals,w,Y)
    z = np.interp(intervals,w,Z)
    for i in range(steps):
        array.append([x[i],y[i],z[i]])
    return array

def get_new_face_training_data(temp,index,training_data):
    input_dim = 3
    average_distance_per_point = np.mean([np.mean(np.sum(np.abs(np.asarray(temp)[1:,:] - np.asarray(temp)[0:-1,:]),axis=1)) ])
    print(average_distance_per_point)
    first_point = temp[31]
    last_point = temp[0]
    distance_to_travel = np.sqrt(np.sum(np.abs(first_point-last_point)**2))
    print(distance_to_travel)
    num_points_inbetween = int(distance_to_travel / average_distance_per_point)

    X = [temp[31][0],temp[36][0]];Y=[temp[31][1],temp[36][1]];Z=[temp[31][2],temp[36][2]]
    array1 = get_new_points(X,Y,Z,num_points_inbetween)
    first_point = temp[36]
    last_point = temp[46]
    distance_to_travel = np.sqrt(np.sum(np.abs(first_point-last_point)**2))
    print(distance_to_travel)
    num_points_inbetween = int(distance_to_travel / average_distance_per_point)
    X = [temp[28][0],temp[46][0]];Y=[temp[28][1],temp[46][1]];Z=[1,temp[46][2]]
    array = get_new_points(X,Y,Z,num_points_inbetween)
    face_drawing=np.empty((90,3), dtype=object)
    for n in range(3):
        j=0
        for i in range(27,31):
          face_drawing[j][n]=temp[i][n]
          j=j+1
        for i in range(len(array1)):
          face_drawing[j][n]=array1[i][n]
          j=j+1
        for i in range(0,27):
          face_drawing[j][n]=temp[i][n]
          j=j+1
        for i in range(len(array)):
          if n!= 2:
             face_drawing[j][n]=array[i][n]
          else:
             face_drawing[j][n]=0.127
          j=j+1
        for i in range(46,90):
          face_drawing[j][n]=temp[i][n]
          j=j+1



    training_data[index]=face_drawing.reshape(1,-1)
    np.save('data/drawings/multi-stroke/drawings-190215-faces-houses-flowers-diff-initial-points.npy', training_data)

def plot_training_images_for_video(index,training_data,plot_dir):
    num_timesteps = 90
    input_dim = 3
    given_part = 30
    num_classes = 3
    run_idx = [index]
    # plot final drawing only
    # # for video animation
    fileformat = '.png'
    traj_lengths_to_plot= np.arange(num_timesteps)
    for l in traj_lengths_to_plot:
        fig = plt.figure('Training_data', figsize=(10,3)) #figsize=(10, 11.0))
        plt.rcParams.update({'font.size': 35, 'legend.fontsize': 30})
        curr_subplot = 0
        for pat in range(3):
            for r in range(len(run_idx)):
                input_traj_idx = run_idx[r]*num_classes+pat
                print(str(run_idx[r]) + ", " + str(1) + ", " + str(input_traj_idx))

                ax = fig.add_subplot(1,3,1 + curr_subplot)
                # ax.set_xlabel('$x_0$')
                # ax.set_ylabel('$x_1$')
                ax.set_xlim([-0.9, 0.9])
                ax.set_ylim([-0.9, 1])
                # if run_idx[r] > 0:
                plt.yticks([], [])
                # if pat < 2:
                plt.xticks([], [])
                #ax.plot(training_data[input_traj_idx,:].reshape((-1,input_dim))[0:30,0], training_data[input_traj_idx,:].reshape((-1,input_dim))[0:30,1], 'orange', linewidth = 5)
                for t in range(1, l):
                    if int(np.round(training_data[input_traj_idx, :].reshape((-1, input_dim))[t, 2])) == 1:
                       ax.plot(training_data[input_traj_idx, :].reshape((-1, input_dim))[t - 1:t + 1, 0],training_data[input_traj_idx, :].reshape((-1, input_dim))[t - 1:t + 1, 1], 'black', linewidth=2)
                    else:
                       ax.plot(training_data[input_traj_idx, :].reshape((-1, input_dim))[t - 1:t + 1, 0],
                            training_data[input_traj_idx, :].reshape((-1, input_dim))[t - 1:t + 1, 1], 'lightgray', linewidth=2)

                curr_subplot += 1
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'Training_data' + str(l) + fileformat))
        plt.close()

if __name__ == '__main__':

    fig, ax = plt.subplots(nrows=1, ncols=1)

    # Use in case you want to display existing drawings in the background.
    WHICH_STIMULUS = -1
    training_data_directory = 'data_generation/drawing-data-sets')
    orig_training_data = np.load(os.path.join(training_data_directory, 'drawings-191105-6-drawings.npy'),allow_pickle=True)
    index= 0
    num_timesteps=90
    display_timesteps=90
    input_dim = 3
    given_part = 0
    num_classes = 6

    if WHICH_STIMULUS >= 0:
        for t in range(1, num_timesteps):
            for stim in range(WHICH_STIMULUS,len(orig_training_data),num_classes):
                traj = orig_training_data[stim].reshape((num_timesteps, input_dim))
                if t >= display_timesteps:
                    continue
                if int(np.round(traj[t,2])) == 1:
                    if t < given_part:
                        ax.plot(traj[t-1:t+1,0], traj[t-1:t+1,1], 'g', linewidth = 0.5)
                    else:
                        ax.plot(traj[t-1:t+1,0], traj[t-1:t+1,1], 'k', linewidth = 0.5)
                else:
                    if t < given_part:
                        ax.plot(traj[t-1:t+1,0], traj[t-1:t+1,1], '#11ff11', linewidth = 0.5)
                    else:
                        ax.plot(traj[t-1:t+1,0], traj[t-1:t+1,1], 'lightgray', linewidth = 0.5)

    interface = DrawingGenerationInterface(ax, process_stroke, process_drawing)
    plt.show(block=False)
