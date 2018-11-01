import tensorflow as tf
import sys
# Don't generate pyc codes
sys.dont_write_bytecode = True

def HomingNetTrainer2(Img, ImageSize, MiniBatchSize):
    """
    Inputs: 
    Img is a MiniBatch of the current image
    ImageSize - Size of the Image
    Outputs:
    pr1 is the predicted output of homing vector for a MiniBatch
    """
    # Img is of size MxNx3
    
    # conv1 output is of size M/2 x N/2 x 64
    conv1 = tf.layers.conv2d(inputs=Img, filters=64, kernel_size=[11, 11], strides=(2, 2), padding="same", activation=None, name='conv1T')

    # bn1 output is of size M/2 x N/2 x 64
    bn1 = tf.layers.batch_normalization(inputs=conv1, name='bn1T')
    bn1 = tf.nn.relu(bn1, name='relu1T')
    
    # conv2 output is of size M/4 x N/4 x 128
    conv2 = tf.layers.conv2d(inputs=bn1, filters=128, kernel_size=[5, 5], strides=(2, 2), padding="same", activation=None, name='conv2T')

    # bn2 output is of size M/4 x N/4 x 128
    bn2 = tf.layers.batch_normalization(inputs=conv2, name='bn2T')
    bn2 = tf.nn.relu(bn2, name='relu2T')
    
    # conv3 output is of size M/8 x N/8 x 256
    conv3 = tf.layers.conv2d(inputs=bn2, filters=256, kernel_size=[3, 3], strides=(2, 2), padding="same", activation=None, name='conv3T')

    # bn3 output is of size M/8 x N/8 x 256
    bn3 = tf.layers.batch_normalization(inputs=conv3, name='bn3T')
    bn3 = tf.nn.relu(bn3, name='relu3T')
    
    # conv4 output is of size M/16 x N/16 x 512
    conv4 = tf.layers.conv2d(inputs=bn3, filters=512, kernel_size=[3, 3], strides=(2, 2), padding="same", activation=None, name='conv4T')

    # bn4 output is of size M/16 x N/16 x 512
    bn4 = tf.layers.batch_normalization(inputs=conv4, name='bn4T')
    bn4 = tf.nn.relu(bn4, name='relu4T')
    
    # conv5 output is of size M/32 x N/32 x 1024
    conv5 = tf.layers.conv2d(inputs=bn4, filters=1024, kernel_size=[3, 3], strides=(2, 2), padding="same", activation=None, name='conv5T')

    # bn5 output is of size M/32 x N/32 x 1024
    bn5 = tf.layers.batch_normalization(inputs=conv5, name='bn5T')
    bn5 = tf.nn.relu(bn5, name='relu5T')

    # fc1
    fc1_flat = tf.reshape(bn5, [-1, ImageSize[0]*ImageSize[1]*1024/(32*32)])
    fc1 = tf.layers.dense(fc1_flat, 1024, activation=tf.nn.relu)

    # fc2
    fc2 = tf.layers.dense(fc1, 1024, activation=tf.nn.relu)

    # pr1
    pr1 = tf.layers.dense(fc2, 2, activation=None)
    
    return pr1
