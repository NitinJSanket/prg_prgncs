import tensorflow as tf
import sys
import numpy as np
# Don't generate pyc codes
sys.dont_write_bytecode = True

def CIFAR10Teacher(Img, ImageSize, MiniBatchSize):
    """
    Inputs: 
    Img is a MiniBatch of the current image
    ImageSize - Size of the Image
    Outputs:
    pr1 is the predicted output of homing vector for a MiniBatch
    """
    # Img is of size MxNx3
    
    # conv1 output is of size M x N x 96
    conv1 = tf.layers.conv2d(inputs=Img, filters=96, kernel_size=[5, 5], strides=(1, 1), padding="same", activation=None, name='conv1T')

    # bn1 output is of size M x N x 96
    bn1 = tf.layers.batch_normalization(inputs=conv1, name='bn1T')
    bn1 = tf.nn.relu(bn1, name='relu1T')

    bn1 = tf.layers.dropout(bn1, rate=0.2, name='dropout1T')

    # conv2 output is of size M/2 x N/2 x 128
    conv2 = tf.layers.conv2d(inputs=bn1, filters=128, kernel_size=[3, 3], strides=(2, 2), padding="same", activation=None, name='conv2T')

    # bn2 output is of size M/2 x N/2 x 128
    bn2 = tf.layers.batch_normalization(inputs=conv2, name='bn2T')
    bn2 = tf.nn.relu(bn2, name='relu2T')

    bn2 = tf.layers.dropout(bn2, rate=0.5, name='dropout2T')
    
    # conv3 output is of size M/4 x N/4 x 256
    conv3 = tf.layers.conv2d(inputs=bn2, filters=256, kernel_size=[3, 3], strides=(2, 2), padding="same", activation=None, name='conv3T')

    # bn3 output is of size M/4 x N/4 x 256
    bn3 = tf.layers.batch_normalization(inputs=conv3, name='bn3T')
    bn3 = tf.nn.relu(bn3, name='relu3T')

    bn3 = tf.layers.dropout(bn3, rate=0.5, name='dropout3T')

    # fc1
    fc1_flat = tf.reshape(bn3, [-1, ImageSize[0]*ImageSize[1]*256/(4*4)], name='fc1_flatT')
    fc1 = tf.layers.dense(fc1_flat, 4096, activation=tf.nn.relu, name='fc1T')

    fc1 = tf.layers.dropout(fc1, rate=0.5, name='dropout4T')

    # fc2
    # fc2 = tf.layers.dense(fc1, 1024, activation=tf.nn.relu, name='fc2T')
 
    # prLogits
    prLogitsT = tf.layers.dense(fc1, 10, activation=None, name='prLogitsT')

    # softmax
    prSoftMaxT = tf.nn.softmax(prLogitsT, name='prSoftMaxT')

    return prLogitsT, prSoftMaxT

