# -*- coding: utf-8 -*-
"""
Created on Mon Nov 06 17:52:34 2017

This script implements a connectome-convolutional neural network to classify age category
on the in-house dataset based on resting-state functional connectivity matrices.
The weights and bias terms of the convolutional layers are initialized based on
the values learned previously on the public dataset, stored in 'weights_public.pickle'. 
The weights and bias terms of the fully connected layers are either randomly 
initialized (set 'initmode' to 1 below) or initialized based on the values learned 
previously on the public dataset (set 'initmode' to 2) and trained in each fold 
of the cross-validation on the in-house dataset (folds are stored in 
'folds_inhouse.npy). Results are saved into 'results_ccnn_class_CONVinitFULLtrain.npz' 
and 'results_ccnn_class_CONVinitFULLtrain.npz', respectively.  

This script was used for the conditions 'CONVinitFULLtrain' and 'CONVinitFULLinit' 
in the manuscript 'Transfer learning improves resting-state functional connectivity 
pattern analysis using convolutional neural networks' by Vakli, Deák-Meszlényi, Hermann,
& Vidnyánszky.

This script is partially based on code from Deep learning course by Udacity: 
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/udacity/4_convolutions.ipynb

@author: Pál Vakli & Regina J. Deák-Meszlényi (RCNS-HAS-BIC)
"""
# %% ############## Selecting the initialization mode and folds ###############
# To evaluate the condition 'CONVinitFULLtrain' you have to run this script
# with 'initmode' set to 1. To evaluate the condition 'CONVinitFULLinit', you
# have to set 'initmode' to 2. 
initmode = 2 # 1 = Randomly initializing the weights and bias terms of the fully 
             # connected layers in each fold of the cross-validation (weights and
             # biases of the convolutional layers are initialized based on
             # the previously learned values and then trained.). 
             # 2 = Initializing the weights and bias terms of the fully connected 
             # layers based on the values learned previously on the public dataset
             # in each fold of the cross-validation (weights and
             # biases of the convolutional layers are initialized based on
             # the previously learned values and then trained.).

# %% ############################ Loading data ################################

# Importing necessary libraries
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle

# loading the correlation matrices
picklefile = "CORR_tensor_inhouse.pickle"
    
with open(picklefile, 'rb') as f:
    save = pickle.load(f)
    data_tensor = save['data_tensor']
    del save

# Loading labels
labels_csv = np.loadtxt("labels_inhouse.txt", delimiter=',')
labels = labels_csv[:, 1]
    
subjectIDs = labels_csv[:, 0]

# Loading weights
picklefile = "weights_public.pickle"
with open(picklefile, 'rb') as f:
    save = pickle.load(f)
    layer1_weights_age = save['layer1_weights']
    layer1_biases_age = save['layer1_biases']
    layer2_weights_age = save['layer2_weights']
    layer2_biases_age = save['layer2_biases']
    if initmode == 2:
        layer3_weights_age = save['layer3_weights']
        layer3_biases_age = save['layer3_biases']
        layer4_weights_age = save['layer4_weights']
        layer4_biases_age = save['layer4_biases']
    del save    

# %% ####################### Function definitions #############################
# Define functions for cross-validation, tensor randomization and normalization 
# and performance calculation

# create_train_and_test_folds randomly divides subjectIDs stored in subjects to 
# num_folds sets
# INPUT: num_folds: number of folds in cross-validation (integer)
#        subjects: list of unique subject IDs
# OUTPUT: IDs: array storing unique subject IDs with num_folds columns: 
#              each column contains IDs of test subjects of the given fold
def create_train_and_test_folds(num_folds, subjects):                          
    n = np.ceil(len(subjects)/num_folds).astype(np.int)
    np.random.shuffle(subjects)
    if len(subjects) != n*num_folds:
        s = np.zeros(n*num_folds)
        s[:len(subjects)] = subjects
        subjects = s
    IDs = subjects.reshape((n, num_folds))
    return IDs

# normalize_tensor standardizes an n dimesional np.array to have zero mean and 
# standard deviation of 1
def normalize_tensor(data_tensor):
    data_tensor -= np.mean(data_tensor)
    data_tensor /= np.max(np.abs(data_tensor))
    return data_tensor

# randomize_tensor generates a random permutation of instances and the 
# corresponding labels before training
# INPUT: dataset: 4D tensor (np.array), instances are concatenated along the 
#                 first (0.) dimension
#        labels: 2D tensor (np.array), storing labels of instances in dataset,
#                instances are concatenated along the first (0.) dimension, 
#                number of columns corresponds to the number of classes, i.e. 
#                labels are stored in one-hot encoding 
# OUTPUT: shuffled_dataset: 4D tensor (np.array), instances are permuted along 
#                           the first (0.) dimension
#         shuffled_labels: 2D tensor (np.array), storing labels of instances in 
#                           shuffled_dataset
def randomize_tensor(dataset, labels):                                         
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation,:,:,:]
    shuffled_labels = labels[permutation,:]
    return shuffled_dataset, shuffled_labels

# create_train_and_test_data creates and prepares training and test datasets and 
# labels for a given fold of cross-validation
# INPUT: fold: number of the given fold (starting from 0)
#        IDs: array storing unique subject IDs with num_folds columns: each 
#             column contains IDs of test subjects of the given fold 
#             (output of reate_train_and_test_folds)
#        subjectIDs: list of subject IDs corresponding to the order of instances 
#                    stored in the dataset (ID of the same subject might appear 
#                    more than once)
#        labels: 1D vector (np.array) storing instance labels as integers 
#                (label encoding)
#        data_tensor: 4D tensor (np.array), instances are concatenated along the 
#                     first (0.) dimension
# OUTPUT: train_data: 4D tensor (np.array) of normalized and randomized train 
#                     instances of the given fold
#         train_labels: 2D tensor (np.array), storing labels of instances in 
#                       train_data in one-hot encoding
#         test_data: 4D tensor (np.array) of normalized (but not randomized) 
#                    test instances of the given fold
#         test_labels: 2D tensor (np.array), storing labels of instances in 
#                      test_data in one-hot encoding
def create_train_and_test_data(fold, IDs, subjectIDs, labels, data_tensor):    
    #create one-hot encoding of labels
    num_labels = len(np.unique(labels))
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    
    #identify the IDs of test subjects
    testIDs = np.in1d(subjectIDs, IDs[:,fold])
        
    test_data = normalize_tensor(data_tensor[testIDs,:,:,:]).astype(np.float32)
    test_labels = labels[testIDs]
    
    train_data = normalize_tensor(data_tensor[~testIDs,:,:,:]).astype(np.float32)
    train_labels = labels[~testIDs]
    train_data, train_labels = randomize_tensor(train_data, train_labels)
    
    return train_data, train_labels, test_data, test_labels

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
  
# %% ####### Preparing the data and initializing network parameters ###########

numROI = 111
num_channels = 1
num_labels = 2
image_size = numROI
batch_size = 4
patch_size = image_size
keep_pr = 0.6   # the probability that each element is kept during dropout
num_folds = 10

# Replacing NaNs with 0s and normalizing data
data_tensor[np.isnan(data_tensor)] = 0
data_tensor = normalize_tensor(data_tensor)

# Loading folds
IDs = np.load('folds_inhouse.npy')

# Variables to store test labels and predictions later on
test_labs = []
test_preds = []

# %% ######################### launch TensorFlow ##############################

# Preallocating arrays to save weights & biases
layer1_weights_save = np.zeros([num_folds, 1, patch_size, num_channels, 64])
layer2_weights_save = np.zeros([num_folds, patch_size, 1, 64, 256])
layer3_weights_save = np.zeros([num_folds, 256, 96])
layer4_weights_save = np.zeros([num_folds, 96, num_labels])

layer1_biases_save = np.zeros([num_folds, 64])
layer2_biases_save = np.zeros([num_folds, 256])
layer3_biases_save = np.zeros([num_folds, 96])
layer4_biases_save = np.zeros([num_folds, num_labels])

# Iterating over folds
for i in range(num_folds):
    
    # Creating train and test data for the given fold
    train_data, train_labels, test_data, test_labels = \
    create_train_and_test_data(i, IDs, subjectIDs, labels, data_tensor)
        
    train_data = train_data[:, :image_size, :image_size, :]
    test_data = test_data[:, :image_size, :image_size, :]
    
    # Defining the computational graph
    graph = tf.Graph()
    
    with graph.as_default():
    
        # Input data placeholders
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
      
        # Test data is a constant
        tf_test_dataset = tf.constant(test_data)
      
        # Network weight constants and variables: weights and biases of the 
        # convolutional layers are initialized based on the previously learned
        # values.
        layer1_weights = tf.Variable(layer1_weights_age, name="layer1_weights")
        layer1_biases = tf.Variable(layer1_biases_age, name="layer1_biases")
        layer2_weights = tf.Variable(layer2_weights_age, name="layer2_weights")
        layer2_biases = tf.Variable(layer2_biases_age, name="layer2_biases")
        # Weights and biases of the fully connected layers are trainable: 
        if initmode == 1:
            # Xavier initialization for better convergence in deep layers
            layer3_weights = tf.get_variable("layer3_weights", shape=[256, 96],
                                             initializer=tf.contrib.layers.xavier_initializer())
            layer3_biases = tf.Variable(tf.constant(0.01, shape=[96]))
            layer4_weights = tf.get_variable("layer4_weights", shape=[96, num_labels],
                                             initializer=tf.contrib.layers.xavier_initializer())
            layer4_biases = tf.Variable(tf.constant(0.01, shape=[num_labels]))
        elif initmode == 2:
            # Initialization based on the previously learned values.
            layer3_weights = tf.Variable(layer3_weights_age, name="layer3_weights")
            layer3_biases = tf.Variable(layer3_biases_age, name="layer3_biases")
            layer4_weights = tf.Variable(layer4_weights_age, name="layer4_weights")
            layer4_biases = tf.Variable(layer4_biases_age, name="layer4_biases")
            
        # Convolutional network architecture
        def model(data, keep_pr):
            # First layer: line-by-line convolution with ReLU and dropout
            conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='VALID')
            hidden = tf.nn.dropout(tf.nn.relu(conv+layer1_biases), keep_pr)
            # Second layer: convolution by column with ReLU and dropout
            conv = tf.nn.conv2d(hidden, layer2_weights, [1, 1, 1, 1], padding='VALID')
            hidden = tf.nn.dropout(tf.nn.relu(conv+layer2_biases), keep_pr)
            # Third layer: fully connected hidden layer with dropout and ReLU
            shape = hidden.get_shape().as_list()
            reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
            hidden = tf.nn.dropout(tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases), keep_pr)
            # Fourth (output) layer: fully connected layer with logits as output
            return tf.matmul(hidden, layer4_weights) + layer4_biases
      
        # Calculate loss-function (cross-entropy) in training
        logits = model(tf_train_dataset, keep_pr)
        loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
        
        # Optimizer definition
        learning_rate = 0.001
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss) 

        # Calculate predictions from training data
        train_prediction = tf.nn.softmax(logits)
      
        # Calculate predictions from test data (keep_pr of dropout is 1!)
        test_prediction = tf.nn.softmax(model(tf_test_dataset, 1))
      
        # Number of iterations
        num_steps = 5001
    
    # Start TensorFlow session
    with tf.Session(graph=graph) as session:
        
        # Initializing variables
        tf.global_variables_initializer().run()
        print('\nVariables initialized for fold %d ...' % (i+1))
        
        # Iterating over the training set
        for step in range(num_steps):
          
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            
            # If we have seen all training data at least once, re-randomize the order 
            # of instances
            if (offset == 0 ):
                train_data, train_labels = randomize_tensor(train_data, train_labels)
        
            # Create batch    
            batch_data = train_data[offset:(offset + batch_size), :, :, :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
        
            # Feed batch data to the placeholders
            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
            _, l, predictions = session.run(
                    [optimizer, loss, train_prediction], feed_dict=feed_dict)
        
            # At every 500. step give some feedback on the progress
            if (step % 500 == 0):
                print('Minibatch loss at step %d: %f' % (step, l))
                print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
                
        # Evaluate the trained model on the test data in the given fold
        test_pred = test_prediction.eval()
        print('Test accuracy at fold %d: %.1f%%' % (i+1, accuracy(test_pred, test_labels)))
          
        # Save test predictions and labels of this fold to a list
        test_labs.append(test_labels)
        test_preds.append(test_pred)

        # Storing weights & biases
        layer1_weights_save[i, :, :, :, :] = layer1_weights.eval()
        layer2_weights_save[i, :, :, :, :] = layer2_weights.eval()
        layer3_weights_save[i, :, :] = layer3_weights.eval()
        layer4_weights_save[i, :, :] = layer4_weights.eval()
        
        layer1_biases_save[i, :] = layer1_biases.eval()
        layer2_biases_save[i, :] = layer2_biases.eval()
        layer3_biases_save[i, :] = layer3_biases.eval()
        layer4_biases_save[i, :] = layer4_biases.eval()

# Create np.array to store all predictions and labels
l = test_labs[0]
p = test_preds[0]   
# Iterate through the cross-validation folds    
for i in range(1, num_folds):
    l = np.vstack((l, test_labs[i]))
    p = np.vstack((p, test_preds[i]))
    
# Calculate final accuracy    
print('\nOverall test accuracy: %.1f%%' % accuracy(p, l))
    
# Save data
if initmode == 1:
    np.savez("results_ccnn_class_CONVinitFULLtrain.npz", \
        labels=l, predictions=p, splits=IDs)
elif initmode == 2:
    np.savez("results_ccnn_class_CONVinitFULLinit.npz", \
        labels=l, predictions=p, splits=IDs)
    
# Saving weights and biases
if initmode == 1:
    pickle_file = "weights_ccnn_class_CONVinitFULLtrain.pickle"
elif initmode == 2:
    pickle_file = "weights_ccnn_class_CONVinitFULLinit.pickle"

try:
    f = open(pickle_file, 'wb')
    save = {
            'layer1_weights': layer1_weights_save,
            'layer1_biases': layer1_biases_save,
            'layer2_weights': layer2_weights_save,
            'layer2_biases': layer2_biases_save,
            'layer3_weights': layer3_weights_save,
            'layer3_biases': layer3_biases_save,
            'layer4_weights': layer4_weights_save,
            'layer4_biases': layer4_biases_save,
            }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise