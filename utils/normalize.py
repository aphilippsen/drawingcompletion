import numpy as np

def normalize(data_orig, axis = 0, threshold = 0, minmax = [], feature_dim = 0):
    """
    Normalize 'data_orig' for each data dimensionality along 'axis', ignoring
    differences in each modality smaller than 'threshold'. Desired data range
    is by default [-1, 1], but can be adjusted via 'minmax' (2d-array of
    2 x num_dimensions).
    Returns the normalized output, offset and the data range
    """

    if axis == 0:
        param_dim = np.size(data_orig, 1)
        if feature_dim == 0:
            feature_dim = param_dim
        if param_dim > feature_dim:
            data = np.reshape(data_orig, (-1, feature_dim))
        else:
            data = data_orig
    else:
        param_dim = np.size(data_orig, 0)
        if feature_dim == 0:
            feature_dim = param_dim
        if param_dim > feature_dim:
            data = np.reshape(data_orig, (feature_dim, -1))
        else:
            data = data_orig

    offset = np.mean(data, axis=axis)

    shiftedData = data - offset

    mins = np.min(shiftedData, axis=axis)
    maxs = np.max(shiftedData, axis=axis)
    ranges = maxs - mins
    ranges[ranges < threshold] = 0
    mins = maxs - ranges
    data_range = np.array([mins, maxs])

    if len(minmax) == 0:
        minmax = np.array([np.repeat([-1], feature_dim), np.repeat([1], feature_dim)])

    oldBase = data_range[0,:]
    newBase = minmax[0,:]

    oldRange = data_range[1,:] - data_range[0,:]
    oldRange[oldRange == 0] = 1
    newRange = minmax[1,:] - minmax[0,:]

    y = newRange * (shiftedData - oldBase)  / oldRange + newBase

    if param_dim > feature_dim:
        y = np.reshape(y, data_orig.shape)
        num_timesteps = int(param_dim/feature_dim)
        offset = np.tile(offset, num_timesteps)
        data_range = np.tile(data_range, num_timesteps)

    return y, offset, data_range

def range2norm(data, offset, data_range, axis = 0, minmax = []):
    if np.ndim(data) == 1:
        data = np.reshape(data, (1, len(data)))
    if axis == 0:
        param_dim = np.size(data, 1)
    else:
        param_dim = np.size(data, 0)

    if minmax.shape[1] < param_dim:
        minmax = np.tile(minmax, int(param_dim/minmax.shape[1]))

    shiftedData = data - offset

    if len(minmax) == 0:
        minmax = np.array([np.repeat([-1], param_dim), np.repeat([1], param_dim)])

    oldBase = data_range[0,:]
    newBase = minmax[0,:]

    oldRange = data_range[1,:] - data_range[0,:]
    oldRange[oldRange == 0] = 1
    newRange = minmax[1,:] - minmax[0,:]

    y = newRange * (shiftedData - oldBase)  / oldRange + newBase
    return y

def norm2range(data, offset, data_range, axis = 0, minmax = []):
    if axis == 0:
        param_dim = np.size(data, 1)
    else:
        param_dim = np.size(data, 0)

    if minmax.shape[1] < param_dim:
        minmax = np.tile(minmax, int(param_dim/minmax.shape[1]))

    if len(minmax) == 0:
        minmax = np.array([np.repeat([-1], param_dim), np.repeat([1], param_dim)])

    oldBase = data_range[0,:]
    newBase = minmax[0,:]

    oldRange = data_range[1,:] - data_range[0,:]
    newRange = minmax[1,:] - minmax[0,:]
    newRange[newRange == 0] = 1

    x = oldRange * (data - newBase) / newRange + oldBase + offset
    return x

