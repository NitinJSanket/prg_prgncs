import tensorflow as tf
import sys
# Don't generate pyc codes
sys.dont_write_bytecode = True

def HomingNetTrainee2(Img, ImageSize, MiniBatchSize):
    """
    Inputs: 
    Img is a MiniBatch of the current image
    ImageSize - Size of the Image
    Outputs:
    pr1 is the predicted output of homing vector for a MiniBatch
    """
    # Img is of size MxNx3
    
    # conv1 output is of size M/2 x N/2 x 8
    conv1 = tf.layers.conv2d(inputs=Img, filters=8, kernel_size=[5, 5], strides=(2, 2), padding="same", activation=None, name='conv1')

    # bn1 output is of size M/2 x N/2 x 8
    bn1 = tf.layers.batch_normalization(inputs=conv1, name='bn1')
    bn1 = tf.nn.relu(bn1, name='relu1')
    
    # conv2 output is of size M/4 x N/4 x 16
    conv2 = tf.layers.conv2d(inputs=bn1, filters=16, kernel_size=[5, 5], strides=(2, 2), padding="same", activation=None, name='conv2')

    # bn2 output is of size M/4 x N/4 x 16
    bn2 = tf.layers.batch_normalization(inputs=conv2, name='bn2')
    bn2 = tf.nn.relu(bn2, name='relu2')
    
    # conv3 output is of size M/8 x N/8 x 8
    conv3 = tf.layers.conv2d(inputs=bn2, filters=8 , kernel_size=[3, 3], strides=(2, 2), padding="same", activation=None, name='conv3')

    # bn3 output is of size M/8 x N/8 x 8
    bn3 = tf.layers.batch_normalization(inputs=conv3, name='bn3')
    bn3 = tf.nn.relu(bn3, name='relu3')
    
    
    # fc1
    fc1_flat = tf.reshape(bn3, [-1, ImageSize[0]*ImageSize[1]*8/(8*8)])
    pr1 = tf.layers.dense(fc1_flat, 2, activation=None)
    
    return pr1
