# transfer_learning_ccnn

This repository contains the Tensorflow implementation of a connectome-convolutional neural network (CCNN) designed to classify age 
category/predict chronological age on the basis of resting-state functional magnetic resonance imaging (fMRI) connectivity matrices. 
The network can be trained first on a source dataset and then the learn weights and bias terms can be used to fine-tune the network
to a target dataset. These scripts were used in:
Pál Vakli, Regina J Deák-Meszlényi, Petra Hermann, Zoltán Vidnyánszky, Transfer learning improves resting-state functional connectivity pattern analysis using convolutional neural networks, GigaScience, Volume 7, Issue 12, December 2018, giy130, https://doi.org/10.1093/gigascience/giy130

* 'ccnn_class_CONVtrainFULLtrain.py' implements the training of the CCNN from scratch to classify age category and the evaluation of classification performance using 10-fold cross-validation.
* 'ccnn_class_publictrain.py' and 'ccnn_class_inhousetrain.py' implement the training of the CCNN from scratch and save the resulting weights and bias terms. 
* 'ccnn_class_CONVconstFULLconst.py' and 'ccnn_class_backtransfer.py' implement the CCNN-based classification of age category on the target dataset using the weights and bias terms as constants whose value correspond to those learn previously on the source dataset.
* 'ccnn_class_CONVconstFULLtrain_FULLinit.py' implements the CCNN-based classification of age category on the target dataset and evaluates classification performance using a 10-fold cross-validation scheme. The weights and bias terms of the convolutional layers as constants whose value correspond to those learn previously on the source dataset, while the weights and biases of the fully connected layers are either initialized randomly, or initialized based on the previously learn values in each fold of the cross-validation.
* 'ccnn_class_CONVinitFULLtrain_FULLinit.py' implements the CCNN-based classification of age category on the target dataset and evaluates classification performance using a 10-fold cross-validation scheme. The weights and bias terms of the convolutional layers are initilaized based on the values learn previously on the source dataset in each fold of the cross-validation, while the weights and biases of the  fully connected layers are either initialized randomly, or initialized based on the previously learn values.
* 'ccnn_regr_baseline.py' implements the training of the CCNN from scratch to regress chronological age and the evaluation of regression performance using 10-fold cross-validation.
* 'ccnn_regr_public.py' implements the training of the fully connected layers of the CCNN to regress chronological age. Weights and biases of the fully connected layers are randomly initialized using Xavier initialization. The weights and biases of the convolutional layers are constants corresponding to those learn previously to classify age category. The learn weights and biases of the fully connected layers can then be used for adapting the network to the target dataset.
* 'ccnn_regr_transfer.py' implements the training of the fully connected layers of the CCNN on the target dataset to regress chronological age and the evaluation of regression performance using 10-fold crossvalidation. The weights and  biases of the convolutional layers are constants corresponding to those learn previously on the source dataset to classify age category. Weights and biases of the fully connected layers are initialized using the values learn on the source dataset to regress chronological age against functional connectivity matrices.
