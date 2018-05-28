# -*- coding: utf-8 -*-
"""
Created on Wed May  2 15:42:18 2018

This script compares the chronological age prediction errors (the absolute value 
of the difference between the true age and the predicted age in years for each 
exemplar) between the baseline and transfer learning conditions using a paired 
t-test. It uses the predictions stored in 'results_ccnn_regr_baseline.npz' and
'results_ccnn_regr_transfer.npz', generated by the scripts 'ccnn_regr_baseline.py'
and 'ccnn_regr_transfer.py', respectively.

This script was used for the baseline and transfer learning regression conditions 
in the manuscript 'Transfer learning improves resting-state functional connectivity 
pattern analysis using convolutional neural networks' by Vakli, Deák-Meszlényi, 
Hermann, & Vidnyánszky.

@author: Pál Vakli & Regina J. Deák-Meszlényi (RCNS-HAS-BIC)
"""
############################### File names ####################################
regr_1 = 'results_ccnn_regr_baseline.npz'
regr_2 = 'results_ccnn_regr_transfer.npz'

###################### Importing necessary libraries ##########################
import numpy as np
from scipy.stats import ttest_rel

########################### Function definition ###############################
# r_squared computes the cofficient of determination (R^2) for the predicted 
# chronological age values
# INPUT: labels: 1D vector (np.array) storing actual labels
#        predictions: 1D vector (np.array) storing predicted labels
# OUTPUT: rsq: R^2
def r_squared(labels, predictions):
    
    ss_res = np.mean(np.square(labels-predictions))
    ss_tot = np.mean(np.square(labels-np.mean(labels)))
    rsq = 1-(ss_res/ss_tot)
    
    return rsq

# Loading labels and predictions, computing accuracy, and performing the t-test

# Loading result files
with np.load(regr_1) as data:
    true_labels1 = data['labels']
    pred_labels1 = data['predictions']
    splits1 = data['splits']
    
with np.load(regr_2) as data:
    true_labels2 = data['labels']
    pred_labels2 = data['predictions']
    splits2 = data['splits']   

# Calculating accuracies
print('R2 in '+regr_1+' is '+str(r_squared(true_labels1, pred_labels1)))
print('R2 in '+regr_2+' is '+str(r_squared(true_labels2, pred_labels2)))

# Sorting subject IDs
ids1 = np.sort(splits1, axis=0)
ids1 = ids1.reshape((ids1.size, -1), order='F')
ids1 = ids1[~np.logical_not(ids1)]

ids2 = np.sort(splits2, axis=0)
ids2 = ids2.reshape((ids2.size, -1), order='F')
ids2 = ids2[~np.logical_not(ids2)]

# If ids1 ~= ids2, rearranging the elements of ids1 to match those of ids2, 
# pred_labels1 to match pred_labels2, true_labels1 to match true_labels2
if not(np.array_equal(ids1, ids2)):
    idx = np.zeros((ids1.size))
    for i in range(ids1.size):
        idx[i] = (np.array(np.where(ids1 == ids2[i])))
    idx.transpose()
    idx = idx.astype('int32')
    ids1 = ids1.take(idx, axis=0)

    pred_labels1 = pred_labels1[idx]
    true_labels = true_labels1[idx]
else:
    true_labels = true_labels1

# Assembling true labels and the corresponding absolute values of the errors
# in the predictions into a single array, then testing whether the average of
# the prediction errors differs significantly across the two conditions via a
# paired t-test.
compare = np.concatenate((true_labels, abs(true_labels-pred_labels1), \
                          abs(true_labels-pred_labels2)), axis=1)
t_statistic, p_value = ttest_rel(compare[:, 1], compare[:, 2])
print('t = '+str(t_statistic))
print('p = '+str(p_value))