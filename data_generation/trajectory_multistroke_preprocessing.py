import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate

faces = np.load('data/drawings/multi-stroke/drawing-2019-02-14_faces.npy')
# data-specific preprocessing (merging wrong ones)
# 35 and 36 should be the same
pen_status_faces = [1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]
new_traj_index_faces = [0, 7, 14, 21, 28, 35, 43, 50, 57, 64]

houses = np.load('data/drawings/multi-stroke/drawing-2019-02-15_houses.npy')
houses = houses[:-1]
pen_status_houses = [1,0,1,0,1, 1,0,1,0,1, 1,0,1,0,1, 1,0,1,0,1, 1,0,1,0,1, 1,0,1,0,1, 1,0,1,0,1, 1,0,1,0,1, 1,0,1,0,1, 1,0,1,0,1]
new_traj_index_houses = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]

# stars = np.load("data/drawings/multi-stroke/drawing-2019-02-17_stars.npy")
# pen_status_stars = [1,1,1,1,1,1,1,1,1,1]
# new_traj_index_stars = [0,1,2,3,4,5,6,7,8,9]

# circles = np.load("data/drawings/multi-stroke/drawing-2019-02-18_circles.npy")
# pen_status_circles = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
# new_traj_index_circles = [0,3,6,9,12,15,18,21,24,27]

cars = np.load('data/drawings/multi-stroke/drawing-2019-11-05_12-03_car-10.npy')
pen_status_cars = [1,0,1,0,1,0,1, 1,0,1,0,1,0,1,1,0,1,0,1,0,1,1,0,1,0,1,0,1,1,0,1,0,1,0,1,1,0,1,0,1,0,1,1,0,1,0,1,0,1,1,0,1,0,1,0,1,1,0,1,0,1,0,1,1,0,1,0,1,0,1,]
new_traj_index_cars = [0, 7, 14, 21, 28, 35, 42, 49, 56, 63]

flowers = np.load("/home/anja/repos/cognitivemirroring/ChainerRNN/data/drawings/multi-stroke/drawing-2019-02-18_flowers.npy")
pen_status_flowers = [1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1]
new_traj_index_flowers = [0,5,10, 15, 20, 25, 30, 35, 40, 45]

human = np.load('data/drawings/multi-stroke/drawing-2019-11-05_10-53_human-figure-10.npy')
human = np.concatenate((human[0:20], human[21:71]),axis=0)
pen_status_human = [1,0,1,0,1,0,1, 1,0,1,0,1,0,1, 1,0,1,0,1,0,1, 1,0,1,0,1,0,1, 1,0,1,0,1,0,1, 1,0,1,0,1,0,1, 1,0,1,0,1,0,1, 1,0,1,0,1,0,1, 1,0,1,0,1,0,1, 1,0,1,0,1,0,1,]
new_traj_index_human = [0, 7, 14, 21, 28, 35, 42, 49, 56, 63]

rockets = np.load('data/drawings/multi-stroke/drawing-2019-11-05_14-08_rocket.npy')
pen_status_rockets = [1,0,1,0,1,0,1,0,1, 1,0,1,0,1,0,1,0,1, 1,0,1,0,1,0,1,0,1, 1,0,1,0,1,0,1,0,1, 1,0,1,0,1,0,1,0,1, 1,0,1,0,1,0,1,0,1, 1,0,1,0,1,0,1,0,1, 1,0,1,0,1,0,1,0,1, 1,0,1,0,1,0,1,0,1, 1,0,1,0,1,0,1,0,1]
new_traj_index_rockets = [0, 9, 18, 27, 36, 45, 54, 63, 72, 81]

num_classes = 6

# all_trajs = np.empty((len(new_traj_index_faces) + len(new_traj_index_flowers)),dtype=object)
# all_trajs = np.empty((len(new_traj_index_faces) + len(new_traj_index_houses) + len(new_traj_index_flowers)),dtype=object)
all_trajs = np.empty((len(new_traj_index_faces) + len(new_traj_index_houses) + len(new_traj_index_cars) + len(new_traj_index_flowers) + len(new_traj_index_human) + len(new_traj_index_rockets)), dtype=object)

assert(len(new_traj_index_faces) == len(new_traj_index_houses) == len(new_traj_index_cars) == len(new_traj_index_flowers) == len(new_traj_index_human) == len(new_traj_index_rockets))
resIdx = 0
for i in range(len(new_traj_index_faces)):
    # APPEND NEXT FACE
    first = new_traj_index_faces[i]
    if i < len(new_traj_index_faces)-1:
        final = new_traj_index_faces[i+1]
    else:
        final = len(faces)
    print(str(first) + " - " + str(final))
    # create pen status column:
    all_pen_status = []
    for j in range(first, final):
        all_pen_status.append(np.repeat(pen_status_faces[j], len(faces[j])))
    all_pen_status = np.concatenate(all_pen_status).reshape((-1,1))
    all_trajs[resIdx] = np.concatenate((np.concatenate(faces[first:final]), all_pen_status), axis=1)
    resIdx += 1

    # APPEND NEXT HOUSE
    first = new_traj_index_houses[i]
    if i < len(new_traj_index_houses)-1:
        final = new_traj_index_houses[i+1]
    else:
        final = len(houses)
    print(str(first) + " - " + str(final))
    # create pen status column:
    all_pen_status = []
    for j in range(first, final):
        all_pen_status.append(np.repeat(pen_status_houses[j], len(houses[j])))
    all_pen_status = np.concatenate(all_pen_status).reshape((-1,1))
    all_trajs[resIdx] = np.concatenate((np.concatenate(houses[first:final]), all_pen_status), axis=1)
    resIdx += 1

    # APPEND NEXT CAR
    first = new_traj_index_cars[i]
    if i < len(new_traj_index_cars)-1:
        final = new_traj_index_cars[i+1]
    else:
        final = len(cars)
    print(str(first) + " - " + str(final))
    # create pen status column:
    all_pen_status = []
    for j in range(first, final):
        all_pen_status.append(np.repeat(pen_status_cars[j], len(cars[j])))
    all_pen_status = np.concatenate(all_pen_status).reshape((-1,1))
    all_trajs[resIdx] = np.concatenate((np.concatenate(cars[first:final]), all_pen_status), axis=1)
    resIdx += 1

    # APPEND NEXT FLOWER
    first = new_traj_index_flowers[i]
    if i < len(new_traj_index_flowers)-1:
        final = new_traj_index_flowers[i+1]
    else:
        final = len(flowers)
    print(str(first) + " - " + str(final))
    # create pen status column:
    all_pen_status = []
    for j in range(first, final):
        all_pen_status.append(np.repeat(pen_status_flowers[j], len(flowers[j])))
    all_pen_status = np.concatenate(all_pen_status).reshape((-1,1))
    all_trajs[resIdx] = np.concatenate((np.concatenate(flowers[first:final]), all_pen_status), axis=1)
    resIdx += 1

    # APPEND NEXT HUMAN
    first = new_traj_index_human[i]
    if i < len(new_traj_index_human)-1:
        final = new_traj_index_human[i+1]
    else:
        final = len(human)
    print(str(first) + " - " + str(final))
    # create pen status column:
    all_pen_status = []
    for j in range(first, final):
        all_pen_status.append(np.repeat(pen_status_human[j], len(human[j])))
    all_pen_status = np.concatenate(all_pen_status).reshape((-1,1))
    all_trajs[resIdx] = np.concatenate((np.concatenate(human[first:final]), all_pen_status), axis=1)
    resIdx += 1

    # APPEND NEXT ROCKET
    first = new_traj_index_rockets[i]
    if i < len(new_traj_index_rockets)-1:
        final = new_traj_index_rockets[i+1]
    else:
        final = len(rockets)
    print(str(first) + " - " + str(final))
    # create pen status column:
    all_pen_status = []
    for j in range(first, final):
        all_pen_status.append(np.repeat(pen_status_rockets[j], len(rockets[j])))
    all_pen_status = np.concatenate(all_pen_status).reshape((-1,1))
    all_trajs[resIdx] = np.concatenate((np.concatenate(rockets[first:final]), all_pen_status), axis=1)
    resIdx += 1


plt.figure()
colors=['lightgray', 'black']
for i in range(0, len(all_trajs), num_classes):
    for j in range(all_trajs[i].shape[0]):
        plt.plot(all_trajs[i][j,0], all_trajs[i][j,1], color=colors[int(all_trajs[i][j,2])], marker='*')
plt.show()

plt.figure()
colors=['lightgray', 'black']
for i in range(1, len(all_trajs), num_classes):
    for j in range(all_trajs[i].shape[0]):
        plt.plot(all_trajs[i][j,0], all_trajs[i][j,1], color=colors[int(all_trajs[i][j,2])], marker='*')
plt.show()

if num_classes > 2:
    plt.figure()
    colors=['lightgray', 'black']
    for i in range(2, len(all_trajs), num_classes):
        for j in range(all_trajs[i].shape[0]):
            plt.plot(all_trajs[i][j,0], all_trajs[i][j,1], color=colors[int(all_trajs[i][j,2])], marker='*')
    plt.show()

    plt.figure()
    colors=['lightgray', 'black']
    for i in range(3, len(all_trajs), num_classes):
        for j in range(all_trajs[i].shape[0]):
            plt.plot(all_trajs[i][j,0], all_trajs[i][j,1], color=colors[int(all_trajs[i][j,2])], marker='*')
    plt.show()

    plt.figure()
    colors=['lightgray', 'black']
    for i in range(4, len(all_trajs), num_classes):
        for j in range(all_trajs[i].shape[0]):
            plt.plot(all_trajs[i][j,0], all_trajs[i][j,1], color=colors[int(all_trajs[i][j,2])], marker='*')
    plt.show()

    plt.figure()
    colors=['lightgray', 'black']
    for i in range(5, len(all_trajs), num_classes):
        for j in range(all_trajs[i].shape[0]):
            plt.plot(all_trajs[i][j,0], all_trajs[i][j,1], color=colors[int(all_trajs[i][j,2])], marker='*')
    plt.show()

# same length
# min_length = np.min([len(all_trajs[x]) for x in range(len(all_trajs))])
time_steps = 90
num_io = 3

x_train = np.zeros((len(all_trajs), time_steps*num_io))

for d in range(len(all_trajs)):
    current_shape = np.asarray(all_trajs[d])

    for i in range(num_io):
        data_x = np.linspace(0, current_shape.shape[0]-1, num=current_shape.shape[0])
        data_y = current_shape[:,i]

        interp_fct = scipy.interpolate.interp1d(data_x, data_y, 'cubic')

        new_x = np.linspace(0, current_shape.shape[0]-1, num=time_steps)
        new_y = interp_fct(new_x)

        # print('fill ' + str(i*time_steps) + " bis " + str((i*time_steps)+time_steps))
        # x_train[d,i*time_steps:(i*time_steps)+time_steps] = np.reshape(new_y, (1,-1))
        x_train[d,i:(num_io*time_steps):num_io] = np.reshape(new_y, (1,-1))

classes = np.tile(np.arange(num_classes), int(len(all_trajs)/num_classes))

data_name = 'drawings-191105-6-drawings'
np.save(data_name + '.npy', x_train)
np.save(data_name + '-classes.npy', classes)

