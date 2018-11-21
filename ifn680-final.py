# -*- coding: utf-8 -*-
"""


+----------------------------------------+
   
+------------------+---------------------+  

"""

import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
import keras
import math
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import *
from keras import backend as K
from keras.models import Model
from keras.optimizers import RMSprop
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def euclidean_distance(vects):
  
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def eucl_dist_output_shape(shapes):

    shape1, shape2 = shapes
    return (shape1[0], 1)



def compute_accuracy(y_true, y_pred):

    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)

def accuracy(y_true, y_pred):
   
    
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

def contrastive_loss(y_true, y_pred):
 
    margin = 1
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)


def create_pairs(x, digit_indices, digits):
    '''
    Positive and negative pair creation.
    Alternates between positive and negative pairs.
    
    @params
                   x:   training dateset
       digit_indices:   category array of all digits, the amount of different digits for each dimension
                  b1:   the minimum digits
                  b2:   the maximun digits
             percent:   the percentage of a digit's amount
    '''
    #store pairs
    pairs = []
    #store labels
    labels = []
    
    # Loop for all digits in the digits_index
    n = min([len(digit_indices[d]) for d in range(len(digits))])-1

    # Sets n as 1 less than the smallest number in the min_sample
    # Loop for all digits in the digits_index
    for d in range(len(digits)):
        #for each digit amount, generate percentage% pairs from total amount n
        for i in range(n):
            #positive pairs
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            #negative pairs
            rand = random.randrange(1,len(digits))
            dn = (d + rand) % (len(digits))
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)

def create_cnn(input_shape):
    #input = keras.layers.Input(shape=input_shape)
    # 16, 16, 32, D:0.25, 0.5, 0.5
    cnn_model = keras.models.Sequential()
    
    # Adds layers to the sequential model
    cnn_model.add(keras.layers.Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    cnn_model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    cnn_model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    cnn_model.add(keras.layers.Dropout(0.25))
    cnn_model.add(keras.layers.Flatten())
    cnn_model.add(keras.layers.Dense(128, activation='relu'))
    cnn_model.add(keras.layers.Dropout(0.5))
    cnn_model.add(keras.layers.Dense(10, activation='softmax'))
    
    return cnn_model

def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = keras.layers.Input(shape=input_shape)
    
    
    
    x = keras.layers.Flatten()(input)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    
    return keras.models.Model(input, x)


#############################################################################
#                            Load and Process Data                          #
#############################################################################
     # Load the Minist dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# the data, shuffled and split between tran and test sets
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

    
# Concatenate the X and y data
x_all = np.append(x_train, x_test, axis = 0)
y_all = np.append(y_train, y_test, axis = 0)

#Creates a mask with all numbers used training and testing, digits in [2,3,4,5,6,7].
mask = np.logical_and(y_all>1, y_all<8)

#Initiating two new arrays with data  for  testing and traning,digits in [2,3,4,5,6,7].
x_2_7_dataset = x_all[mask,:,:]
y_2_7_dataset = y_all[mask]

#Initiating two new arrays with data only for testing ,digits in [0,1,8,9].
x_0_9_dataset = x_all[~mask,:,:]
y_0_9_dataset = y_all[~mask]

modified_x_train, modified_x_test,  modified_y_train,  modified_y_test = train_test_split(x_2_7_dataset,
                                                                                                          y_2_7_dataset,
                                                                                                          test_size=0.2,random_state=42)
final_x_test = np.append(modified_x_test,x_0_9_dataset,axis = 0)
final_y_test = np.append(modified_y_test,y_0_9_dataset,axis = 0)
###########################################
img_rows, img_cols = modified_x_train.shape[1:3]
# reshape the input arrays to 4D (batch_size, rows, columns, channels)
modified_x_train = modified_x_train.reshape(modified_x_train.shape[0], img_rows, img_cols, 1)
#exp_2_pairs = exp_2_pairs.reshape(exp_2_pairs.shape[0], exp_2_pairs.shape[1], img_rows, img_cols, 1)

img_rows, img_cols = modified_x_test.shape[1:3]
# reshape the input arrays to 4D (batch_size, rows, columns, channels)
modified_x_test = modified_x_test.reshape(modified_x_test.shape[0], img_rows, img_cols, 1)
#exp_2_pairs = exp_2_pairs.reshape(exp_2_pairs.shape[0], exp_2_pairs.shape[1], img_rows, img_cols, 1)

img_rows, img_cols = x_0_9_dataset.shape[1:3]
# reshape the input arrays to 4D (batch_size, rows, columns, channels)
x_0_9_dataset = x_0_9_dataset.reshape(x_0_9_dataset.shape[0], img_rows, img_cols, 1)
#exp_2_pairs = exp_2_pairs.reshape(exp_2_pairs.shape[0], exp_2_pairs.shape[1], img_rows, img_cols, 1)

img_rows, img_cols = final_x_test.shape[1:3]
# reshape the input arrays to 4D (batch_size, rows, columns, channels)
final_x_test = final_x_test.reshape(final_x_test.shape[0], img_rows, img_cols, 1)
#exp_2_pairs = exp_2_pairs.reshape(exp_2_pairs.shape[0], exp_2_pairs.shape[1], img_rows, img_cols, 1)
###############################################################################
#                            Create Pairs                                #
###############################################################################

first_digits_set = [2,3,4,5,6,7]
second_digits_set = [0, 1, 8, 9]
third_digits_set = [0,1,2,3,4,5,6,7,8,9]

# The specific digits that are used for the different sets of image pairs can be found at the top of this document.
# Creates pairs of images that will be used to train the model with digits in set1_digits
digit_indices = [np.where(modified_y_train == i)[0] for i in first_digits_set]
training_pairs, training_set = create_pairs(modified_x_train, digit_indices, first_digits_set)

# Creates pairs of images that will be used to test the model with digits in set1_digits
digit_indices = [np.where(modified_y_test == i)[0] for i in first_digits_set]
test_1_pairs, test_set1_y = create_pairs(modified_x_test, digit_indices, first_digits_set)

# Creates pairs of images that will be used to test the model with digits in set2_digits
digit_indices = [np.where(y_0_9_dataset == i)[0] for i in second_digits_set]
test_2_pairs, test_set2_y = create_pairs(x_0_9_dataset, digit_indices, second_digits_set)

# Creates pairs of images that will be used to test the model with digits in set3_digits
digit_indices = [np.where(final_y_test == i)[0] for i in third_digits_set]
test_3_pairs, test_set3_y = create_pairs(final_x_test, digit_indices, third_digits_set)


###############################################################################
 

#input data should be in the form of 28 by 28 matrix
input_shape = modified_x_train.shape[1:]

epochs = 10
verbose=1

model = train_model_with_validation(input_shape=input_shape,
                                    training_pairs=training_pairs,
                                    training_set=training_set,
                                    test_pairs=test_1_pairs,
                                    test_set=test_set1_y,
                                    epochs=epochs,
                                    verbose=verbose
                                    )
print('##################################################################################')
y_pred = model.predict([training_pairs[:, 0], training_pairs[:, 1]])
print('* Accuracy on training set: %0.2f%%' % (100 * compute_accuracy(training_set, y_pred)))
y_pred = model.predict([test_1_pairs[:, 0], test_1_pairs[:, 1]])
print('* Accuracy on test set 1: %0.2f%%' % (100 * compute_accuracy(test_set1_y, y_pred)))
print('##################################################################################')
            
           





def train_model_with_validation(input_shape,
                                training_pairs,
                                training_set,
                                test_pairs,
                                test_set,
                                epochs,
                                verbose
                               ):
    '''
    args:
        Basically same as above.

    returns:
        Siamese model to be avaluated.
    '''

    # Use a CNN network as the shared network.
    cnn_network_model = create_cnn(input_shape)

    # Initiates inputs with the same amount of slots to keep the image arrays sequences to be used as input data when processing the inputs.
    image_vector_shape_1 = Input(shape=input_shape)
    image_vector_shape_2 = Input(shape=input_shape)

    # The CNN network model will be shared including weights
    output_cnn_1 = cnn_network_model(image_vector_shape_1)
    output_cnn_2 = cnn_network_model(image_vector_shape_2)

    # Concatenates the two output vectors into one.
    distance = keras.layers.Lambda(euclidean_distance,
                                   output_shape=eucl_dist_output_shape)([output_cnn_1, output_cnn_2])

    # We define a trainable model linking the two different image inputs to the distance between the        processed input by the cnn network.
    model = Model([image_vector_shape_1, image_vector_shape_2],
                  distance
                 )
    # Specifying the optimizer for the netwrok model
    rms = keras.optimizers.RMSprop()

    # Compiles the model with the contrastive loss function.
    model.compile(loss=contrastive_loss,
                  optimizer=rms,
                  metrics=[accuracy])

    # Number of epochs is defined in the beginning of the document as a static variable.
    # Validating and printing data using the test data with index i.
    model.fit([training_pairs[:, 0], training_pairs[:, 1]], training_set,
              batch_size=128,
              epochs=epochs,
              verbose=verbose,
              validation_data=([test_pairs[:, 0], test_pairs[:, 1]], test_set)
             )

    return model



