# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 10:25:44 2018

This script compares binary classification performance between two conditions 
(two classifiers) using a binomial test according to Salzberg SL. On Comparing 
Classifiers: Pitfalls to Avoid and a Recommended Approach. Data Min Knowl 
Discov. 1997;1:317–28.
Variables 'class_1' and 'class_2' specify the path/name of the files containing 
the true and predicted labels (file name format: 'results_ccnn_class_*.npz').

This script was used to evaluate classification performance in the baseline and
transfer learning conditions in the manuscript 'Transfer learning improves 
resting-state functional connectivity pattern analysis using convolutional 
neural networks' by Vakli, Deák-Meszlényi, Hermann, & Vidnyánszky.

@author: Pál Vakli & Regina J. Deák-Meszlényi (RCNS-HAS-BIC)
"""
############################### File names ####################################
class_1 = 'results_ccnn_class_CONVinitFULLtrain_inhouse.npz'
class_2 = 'results_ccnn_class_CONVtrainFULLtrain_inhouse.npz'

###################### Importing necessary libraries ##########################
import numpy as np
from scipy.special import comb

########################### Function definition ###############################

# accuracy calculates classification accuracy from one-hot encoded labels and 
# predictions
# INPUT: predictions: 2D tensor (np.array), storing predicted labels 
#                     (calculated with soft-max in our case) of instances with 
#                     one-hot encoding  
#       labels: 2D tensor (np.array), storing actual labels with one-hot 
#               encoding
# OUTPUT: accuracy in %
def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])

############ Loading labels and predictions, computing accuracy ###############
    
# Loading result files
with np.load(class_1) as data:
    true_labels1 = data['labels']
    pred_labels1 = data['predictions']
    # There was no cross-validation in CONVconstFULLconst
    if (class_1 != 'results_ccnn_class_CONVconstFULLconst_inhouse.npz') | (class_1 != 'results_ccnn_class_CONVconstFULLconst_NKI-RS_subset.npz'):
        splits1 = data['splits']
    
with np.load(class_2) as data:
    true_labels2 = data['labels']
    pred_labels2 = data['predictions']
    # There was no cross-validation in CONVconstFULLconst
    if (class_2 != 'results_ccnn_class_CONVconstFULLconst_inhouse.npz') | (class_2 != 'results_ccnn_class_CONVconstFULLconst_NKI-RS_subset.npz'):
        splits2 = data['splits']

# Calculating accuracies
print('Classification accuracy in '+class_1+' is '+str(accuracy(pred_labels1, true_labels1)))
print('Classification accuracy in '+class_2+' is '+str(accuracy(pred_labels2, true_labels2)))

# Standard binary labels
true_labels1 = true_labels1[:, 1]
true_labels2 = true_labels2[:, 1]
pred_labels1 = np.round(pred_labels1[:, 1])
pred_labels2 = np.round(pred_labels2[:, 1])

# Sorting subject IDs
if (class_1 != 'results_ccnn_class_CONVconstFULLconst_inhouse.npz') | (class_1 != 'results_ccnn_class_CONVconstFULLconst_NKI-RS_subset.npz'):
    ids1 = np.sort(splits1, axis=0)
    ids1 = ids1.reshape((ids1.size, -1), order='F')
    ids1 = ids1[~np.logical_not(ids1)]
else:
    if class_1[-11] == "inhouse.npz":
        labels = np.loadtxt('labels_inhouse.txt', delimiter=',')
    elif class_1[-11] == "_subset.npz":
        labels = np.loadtxt("labels_NKI-RS_subset.csv", delimiter=',')
    ids1 = labels[:, 0]

if (class_2 != 'results_ccnn_class_CONVconstFULLconst_inhouse.npz') | (class_2 != 'results_ccnn_class_CONVconstFULLconst_NKI-RS_subset.npz'):
    ids2 = np.sort(splits2, axis=0)
    ids2 = ids2.reshape((ids2.size, -1), order='F')
    ids2 = ids2[~np.logical_not(ids2)]
else:
    if class_2[-11] == "inhouse.npz":
        labels = np.loadtxt('labels_inhouse.txt', delimiter=',')
    elif class_2[-11] == "_subset.npz":
        labels = np.loadtxt("labels_NKI-RS_subset.csv", delimiter=',')
    ids2 = labels[:, 0]

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

# Assembling true labels and predictions into a single array
pred_labels1 = np.reshape(pred_labels1, (pred_labels1.shape[0], -1))
pred_labels2 = np.reshape(pred_labels2, (pred_labels2.shape[0], -1))
true_labels = np.reshape(true_labels, (true_labels.shape[0], -1))

compare = np.concatenate((true_labels, pred_labels1, pred_labels2), axis=1)

######################## Performing the binomial test #########################

# Indices of where the predictions differ
id_diff = np.array(np.where(np.not_equal(compare[:, 1], compare[:, 2]))).reshape(-1, 1) 
# Retaining only the instances where the predictions differ
compare = compare.take(id_diff.transpose(), axis=0).squeeze()

# Indices of where the labels predicted by classifier 2 do not equal the true
# labels (and hence those predicted by classifier 1) Naturally, these are the 
# indices of where classifier1 outperformed classifier2. 
id_firstbett = np.array(np.where(np.not_equal(compare[:, 0], compare[:, 2]))).reshape(-1, 1) 

# Number of different predictions
diff = id_diff.shape[0]
# Number of times classifier1 got it right and classifier2 got it wrong
first_better = id_firstbett.shape[0]

# Checking whether classifier2 actually outperformed classifier1
if first_better < diff/2:
    first_better = diff-first_better

# Computing the probability p that out of 'diff' number of examples classifier 
# A was correct while classifier B was wrong at least 'first_better' times, 
# given the assumption that the two classifiers perform equally well.
prob = 0
for s in range(first_better, diff+1):
    prob += comb(diff, s)

prob = prob*0.5**diff

print('p = '+'{:f}'.format(prob))