import tensorflow as tf
import sys
import numpy as np
# Don't generate pyc codes
sys.dont_write_bytecode = True

def CIFAR10Identity(Img, ImageSize, MiniBatchSize):
    """
    Inputs: 
    Img is a MiniBatch of the current image
    ImageSize - Size of the Image
    Outputs:
    pr1 is the predicted output of homing vector for a MiniBatch
    """
    # Img is of size MxNx3
    
    # conv1 output is of size M/2 x N/2 x 8
    conv1 = tf.layers.conv2d(inputs=Img, filters=8, kernel_size=[5, 5], strides=(1, 1), padding="same", activation=None, name='conv1T')

    # bn1 output is of size M x N x 8
    bn1 = tf.nn.relu(conv1, name='relu1T')


    conv2 = tf.layers.conv2d(inputs=bn1, filters=16, kernel_size=[5, 5], strides=(1, 1), padding="same", activation=None, name='conv2T')

    # bn1 output is of size M x N x 8
    bn2 = tf.nn.relu(conv2, name='relu2T')

    conv3 = tf.layers.conv2d(inputs=bn2, filters=16, kernel_size=[5, 5], strides=(2, 2), padding="same", activation=None, name='conv3T')

    # bn1 output is of size M x N x 8
    bn3 = tf.nn.relu(conv3, name='relu3T')

    # deconv1 output is of size MxNx3
    prImgCol = tf.layers.conv2d_transpose(inputs=bn3, filters=2, kernel_size=[5, 5], strides=(4, 4), padding="valid", activation=None, name='final')

    # conv2 output is of size MxNx3
    # prImgCol = tf.layers.conv2d(inputs=deconv1, filters=1, kernel_size=[5, 5], strides=(1, 1), padding="same", activation=None, name='final')

    # final = tf.slice(conv2, [0, 0, 0, 0], [MiniBatchSize, ImageSize[0], ImageSize[1], 1], name='final')

    # print(np.shape(final))
    # input('z')
    # prImgCol = tf.reshape(deconv1, [-1, ImageSize[0]*ImageSize[1]*1*(2*2)/(1*1)], name='prImgCol')
    # print(np.shape(prImgCol))
    # input('z')
    
    return prImgCol

