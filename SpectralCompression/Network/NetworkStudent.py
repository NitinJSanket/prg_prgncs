import tensorflow as tf
import sys
import numpy as np
# Don't generate pyc codes
sys.dont_write_bytecode = True

def CIFAR10Student(Img, ImageSize, MiniBatchSize):
    """
    Inputs: 
    Img is a MiniBatch of the current image
    ImageSize - Size of the Image
    Outputs:
    pr1 is the predicted output of homing vector for a MiniBatch
    """
    # Img is of size MxNx3
    
    # conv1 output is of size M x N x 8
    conv1 = tf.layers.conv2d(inputs=Img, filters=8, kernel_size=[7, 7], strides=(1, 1), padding="same", activation=None, name='conv1S')

    # bn1 output is of size M x N x 8
    bn1 = tf.layers.batch_normalization(inputs=conv1, name='bn1S')
    bn1 = tf.nn.relu(bn1, name='relu1S')

    # conv2 output is of size M/2 x N/2 x 16
    conv2 = tf.layers.conv2d(inputs=bn1, filters=16, kernel_size=[5, 5], strides=(2, 2), padding="same", activation=None, name='conv2S')

    # bn2 output is of size M/2 x N/2 x 16
    bn2 = tf.layers.batch_normalization(inputs=conv2, name='bn2S')
    bn2 = tf.nn.relu(bn2, name='relu2S')
    
    # conv3 output is of size M/4 x N/4 x 32
    conv3 = tf.layers.conv2d(inputs=bn2, filters=32, kernel_size=[5, 5], strides=(2, 2), padding="same", activation=None, name='conv3S')

    # bn3 output is of size M/4 x N/4 x 32
    bn3 = tf.layers.batch_normalization(inputs=conv3, name='bn3S')
    bn3 = tf.nn.relu(bn3, name='relu3S')

    # fc1
    fc1_flat = tf.reshape(bn3, [-1, ImageSize[0]*ImageSize[1]*32/(4*4)], name='fc1_flatS')
    fc1 = tf.layers.dense(fc1_flat, 32, activation=tf.nn.relu, name='fc2S')
 
    # prLogits
    prLogitsS = tf.layers.dense(fc1, 10, activation=None, name='prLogitsS')

    # softmax
    prSoftMaxS = tf.nn.softmax(prLogitsS, name='prSoftMaxS')

    return prLogitsS, prSoftMaxS
