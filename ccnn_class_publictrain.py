# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 14:59:51 2017

This script trains a connectome-convolutional neural network on the public resting-state
functional connectivity data to classify age category and saves the resulting 
weights and bias terms into 'weights_public.pickle'.

This script was used for generating weights for the conditions 'CONVconstFULLconst', 
'CONVconstFULLtrain', 'CONVconstFULLinit', 'CONVinitFULLtrain', 'CONVinitFULLinit',
and the regression transfer learning condition in the manuscript 'Transfer learning 
improves resting-state functional connectivity pattern analysis using 
convolutional neural networks' by Vakli, Deák-Meszlényi, Hermann, & Vidnyánszky.

This script is partially based on code from Deep learning course by Udacity: 
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/udacity/4_convolutions.ipynb

@author: Pál Vakli & Regina J. Deák-Meszlényi (RCNS-HAS-BIC)
"""
# %% ########################### Loading data #################################

# Importing necessary libraries
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle

# loading the correlation matrices
picklefile = "CORR_tensor_public.pickle"
    
with open(picklefile, 'rb') as f:
    save = pickle.load(f)
    data_tensor = save['data_tensor']
    del save

data_tensor = data_tensor.astype(np.float32)

# Loading labels
labels_csv = np.loadtxt("labels_public.csv", delimiter=',')                      
labels = labels_csv[:, 4]

# %% ####################### Function definitions #############################
# Define functions for tensor randomization, normalization, and performance 
# calculation

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
#                 instances are concatenated along the first (0.) dimension, 
#                 number of columns corresponds to the number of classes, i.e. 
#                 labels are stored in one-hot encoding 
# OUTPUT: shuffled_dataset: 4D tensor (np.array), instances are permuted along 
#                           the first (0.) dimension
#         shuffled_labels: 2D tensor (np.array), storing labels of instances in 
#                           shuffled_dataset
def randomize_tensor(dataset, labels):                                         # sorrendcsere
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation,:,:,:]
    shuffled_labels = labels[permutation,:]
    return shuffled_dataset, shuffled_labels

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
  
# %% ### Preparing the training set and initializing network parameters #######

numROI = 111
num_channels = 1
num_labels = 2
image_size = numROI
batch_size = 4 
patch_size = image_size
keep_pr = 0.6 # the probability that each element is kept during dropout

# Replacing NaNs with 0s and normalizing training data
data_tensor[np.isnan(data_tensor)] = 0
train_data = normalize_tensor(data_tensor)

# One-hot encoded train labels
num_labels = len(np.unique(labels))
train_labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)

# %% ####################### launching TensorFlow #############################

# Drawing the computational graph    
graph = tf.Graph()
    
with graph.as_default():
    
    # Input data placeholders and constants
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
      
    # Network weight and bias variables: Xavier initialization for better 
    # convergence in deep layers
    layer1_weights = tf.get_variable("layer1_weights", shape=[1, patch_size, num_channels, 64],
                                     initializer=tf.contrib.layers.xavier_initializer())
    layer1_biases = tf.Variable(tf.constant(0.001, shape=[64]))
    layer2_weights = tf.get_variable("layer2_weights", shape=[patch_size, 1, 64, 256],
                                     initializer=tf.contrib.layers.xavier_initializer())
    layer2_biases = tf.Variable(tf.constant(0.001, shape=[256]))
    layer3_weights = tf.get_variable("layer3_weights", shape=[256, 96],
                                     initializer=tf.contrib.layers.xavier_initializer())
    layer3_biases = tf.Variable(tf.constant(0.01, shape=[96]))
    layer4_weights = tf.get_variable("layer4_weights", shape=[96, num_labels],
                                     initializer=tf.contrib.layers.xavier_initializer())
    layer4_biases = tf.Variable(tf.constant(0.01, shape=[num_labels]))
        
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
            
    # Number of iterations
    num_steps = 5001
    
# Start TensorFlow session
with tf.Session(graph=graph) as session:
    
    # Initializing variables    
    tf.global_variables_initializer().run()
    print('\nVariables initialized ...')
    
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
        
    # Retrieving final weights and bias terms
    layer1_weights_final = layer1_weights.eval()
    layer1_biases_final = layer1_biases.eval()
    layer2_weights_final = layer2_weights.eval()
    layer2_biases_final = layer2_biases.eval()
    layer3_weights_final = layer3_weights.eval()
    layer3_biases_final = layer3_biases.eval()
    layer4_weights_final = layer4_weights.eval()
    layer4_biases_final = layer4_biases.eval()
    
# Saving weights and biases
pickle_file = "weights_public.pickle"
    
try:
    f = open(pickle_file, 'wb')
    save = {
            'layer1_weights': layer1_weights_final,
            'layer1_biases': layer1_biases_final,
            'layer2_weights': layer2_weights_final,
            'layer2_biases': layer2_biases_final,
            'layer3_weights': layer3_weights_final,
            'layer3_biases': layer3_biases_final,
            'layer4_weights': layer4_weights_final,
            'layer4_biases': layer4_biases_final,
            }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise