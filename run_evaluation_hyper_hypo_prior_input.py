import numpy as np
import matplotlib
import matplotlib.pyplot as plt

[final_res, final_uh_history, final_err_vis_corr, final_err_vis_best, final_err_new_corr, final_err_new_best, final_err_vis_largest, final_err_new_largest, final_vis_best_class, final_new_best_class] = np.load('/home/anja/github/drawingcompletion/results/hyper_hypo_prior_input/final_0.01-100_6x7/use_init_state_loss_input-class-[0, 1, 2, 3, 4, 5]_inference-100_start_from_mean/all-drawing-style-evals.npy', allow_pickle=True)

def count_confusions(matrix):
    norm_factor = matrix.shape[2]*len(matrix[0,0,0])
    confusions = np.zeros((matrix.shape[0], matrix.shape[1]))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            for k in range(matrix.shape[2]):
                confusions[i,j] += np.sum([x!=k for x in matrix[i,j,k]])
    return confusions/norm_factor

def average_error(matrix):
    avg_error = np.zeros((matrix.shape[0], matrix.shape[1]))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            avg_error[i,j] = np.mean([np.mean(x) for x in matrix[i,j]])
    return avg_error


count_confusions(final_vis_best_class)
#             sensor precise  sensor normal  sensor imprecise
# prior precise     
# prior normal      
# prior imprecise   

#array([[0.01666667, 0.1       , 0.13888889],
#       [0.01111111, 0.03888889, 0.16111111],
#       [0.        , 0.00555556, 0.77222222]])


count_confusions(final_new_best_class)
#array([[0.12777778, 0.21666667, 0.24444444],
#       [0.07777778, 0.15555556, 0.19444444],
#       [0.84444444, 0.8       , 0.86666667]])


average_error(final_err_vis_corr)
#array([[0.01531427, 0.02964912, 0.03455991],
#       [0.01095854, 0.01969333, 0.03921373],
#       [0.01010892, 0.01356861, 0.47022551]])


average_error(final_err_new_corr)
#array([[0.05866375, 0.08118785, 0.09272107],
#       [0.05754315, 0.06990458, 0.08788162],
#       [0.59007427, 0.60336977, 0.62176742]])

def plotMat(X, vmin=0, vmax=1, is_percentage=False):
    fig = plt.figure(figsize = (10,7))
    parameters = {'xtick.labelsize': 25,
                  'ytick.labelsize': 25,
                  'axes.labelsize': 25,
                  'axes.titlesize': 25}
    plt.rcParams.update(parameters)
    
    current_cmap = matplotlib.cm.get_cmap('viridis')
    current_cmap.set_bad(color='gray')
    
    ax = fig.add_subplot()
    #imshow portion
    if is_percentage:
        s = ax.imshow(X*100, interpolation='nearest', cmap = current_cmap, vmin=vmin, vmax=vmax)
    else:
        s = ax.imshow(X, interpolation='nearest', cmap = current_cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(s, ax=ax)
    #text portion
    diff = 1.
    min_val = 0.
    rows = X.shape[0]
    cols = X.shape[1]
    col_array = np.arange(min_val, cols, diff)
    row_array = np.arange(min_val, rows, diff)
    x, y = np.meshgrid(col_array, row_array)
    for col_val, row_val in zip(x.flatten(), y.flatten()):
        if np.isnan(X[row_val.astype(int),col_val.astype(int)]):
            continue # skip this for nan values
            
        # display color either white or black for better contrast
        if X[row_val.astype(int),col_val.astype(int)] < 0.17:
            textcolor = 'white'
        else:
            textcolor = 'black'
        
        # generate string that should be displayed in tile
        if is_percentage:
            c = str(int(np.round(X[row_val.astype(int),col_val.astype(int)]*100))) + "%"
            X[row_val.astype(int),col_val.astype(int)] = np.round(X[row_val.astype(int),col_val.astype(int)]*100)
        else:
            c = str(np.round(X[row_val.astype(int),col_val.astype(int)]*100)/100)
            
        ax.text(col_val, row_val, c, va='center', ha='center', color=textcolor, fontsize=30)

    #set tick marks for grid
    ax.set_xticks(np.arange(cols)) #np.arange(min_val+diff/2, cols+diff/2))#np.arange(min_val-diff/2, cols-diff/2))
    ax.set_yticks(np.arange(rows)) #np.arange(min_val+diff/2, cols+diff/2))#np.arange(min_val-diff/2, rows-diff/2))
    ax.set_xticklabels(['low', 'middle', 'high'])
    ax.set_yticklabels(['low', 'middle', 'high'])
    ax.set_xlim(min_val-diff/2, cols-diff/2)
    ax.set_ylim(min_val-diff/2, rows-diff/2)

    #plt.show()

error_vis_corr = average_error(final_err_vis_corr)
plotMat(np.flipud(np.fliplr(error_vis_corr)),vmax=0.65)
plt.ylabel('precision of prior')
plt.xlabel('precision of sensor')
plt.title('Average error to corresponding shape (presented part)')
plt.tight_layout()
plt.savefig('error-corr-vis.pdf')
plt.close()

error_vis_best = average_error(final_err_vis_best)
plotMat(np.flipud(np.fliplr(error_vis_best)),vmax=0.65)
plt.ylabel('precision of prior')
plt.xlabel('precision of sensor')
plt.title('Average error to closest shape (presented part)')
plt.tight_layout()
plt.savefig('error-best-vis.pdf')
plt.close()

error_new_corr = average_error(final_err_new_corr)
plotMat(np.flipud(np.fliplr(error_new_corr)),vmax=0.65)
plt.ylabel('precision of prior')
plt.xlabel('precision of sensor')
plt.title('Average error to corresponding shape (completed part)')
plt.tight_layout()
plt.savefig('error-corr-new.pdf')
plt.close()

error_new_best = average_error(final_err_new_best)
plotMat(np.flipud(np.fliplr(error_new_best)),vmax=0.65)
plt.ylabel('precision of prior')
plt.xlabel('precision of sensor')
plt.title('Average error to closest shape (completed part)')
plt.tight_layout()
plt.savefig('error-best-new.pdf')
plt.close()

confusions_vis_part = count_confusions(final_vis_best_class)
confusions_new_part = count_confusions(final_new_best_class)

# those with a high error are invalid, mark as NaN
cut_off_threshold = 0.3
confusions_vis_part[error_vis_best>cut_off_threshold] = np.nan
confusions_new_part[error_new_best>cut_off_threshold] = np.nan

plotMat(np.copy(np.flipud(np.fliplr(confusions_new_part))), vmax = 25, is_percentage=True)
plt.ylabel('precision of prior')
plt.xlabel('precision of sensor')
plt.title('Percentage of misinterpreted shapes\n (completed part)')
plt.tight_layout()
plt.savefig('confusions-new.pdf')
plt.close()

plotMat(np.copy(np.flipud(np.fliplr(confusions_vis_part))), vmax = 25, is_percentage=True)
plt.ylabel('precision of prior')
plt.xlabel('precision of sensor')
plt.title('Percentage of misinterpreted shapes\n (presented part)')
plt.tight_layout()
plt.savefig('confusions-vis.pdf')
plt.close()



