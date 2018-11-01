import tensorflow as tf
import sys
# Don't generate pyc codes
sys.dont_write_bytecode = True

def HomingNet(Img, ImageSize, MiniBatchSize):
    """
    Inputs: 
    Img is a MiniBatch of the current image
    ImageSize - Size of the Image
    Outputs:
    pr1 is the predicted output of homing vector for a MiniBatch
    """
    # Img is of size MxNx3
    
    # conv1 output is of size M/2 x N/2 x 64
    conv1 = tf.layers.conv2d(inputs=Img, filters=64, kernel_size=[11, 11], strides=(2, 2), padding="same", activation=None, name='conv1')

    # bn1 output is of size M/2 x N/2 x 64
    bn1 = tf.layers.batch_normalization(inputs=conv1, name='bn1')
    bn1 = tf.nn.relu(bn1, name='relu1')
    
    # conv2 output is of size M/4 x N/4 x 128
    conv2 = tf.layers.conv2d(inputs=bn1, filters=128, kernel_size=[5, 5], strides=(2, 2), padding="same", activation=None, name='conv2')

    # bn2 output is of size M/4 x N/4 x 128
    bn2 = tf.layers.batch_normalization(inputs=conv2, name='bn2')
    bn2 = tf.nn.relu(bn2, name='relu2')
    
    # conv3 output is of size M/8 x N/8 x 256
    conv3 = tf.layers.conv2d(inputs=bn2, filters=256, kernel_size=[3, 3], strides=(2, 2), padding="same", activation=None, name='conv3')

    # bn3 output is of size M/8 x N/8 x 256
    bn3 = tf.layers.batch_normalization(inputs=conv3, name='bn3')
    bn3 = tf.nn.relu(bn3, name='relu3')
    
    # conv4 output is of size M/16 x N/16 x 512
    conv4 = tf.layers.conv2d(inputs=bn3, filters=512, kernel_size=[3, 3], strides=(2, 2), padding="same", activation=None, name='conv4')

    # bn4 output is of size M/16 x N/16 x 512
    bn4 = tf.layers.batch_normalization(inputs=conv4, name='bn4')
    bn4 = tf.nn.relu(bn4, name='relu4')
    
    # conv5 output is of size M/32 x N/32 x 256
    conv5 = tf.layers.conv2d(inputs=bn4, filters=256, kernel_size=[3, 3], strides=(2, 2), padding="same", activation=None, name='conv5')

    # bn5 output is of size M/32 x N/32 x 256
    bn5 = tf.layers.batch_normalization(inputs=conv5, name='bn5')
    bn5 = tf.nn.relu(bn5, name='relu5')

    # conv6 output is of size M/64 x N/64 x 128
    conv6 = tf.layers.conv2d(inputs=bn5, filters=128, kernel_size=[3, 3], strides=(2, 2), padding="same", activation=None, name='conv6')

    # bn6 output is of size M/64 x N/64 x 128
    bn6 = tf.layers.batch_normalization(inputs=conv6, name='bn6')
    bn6 = tf.nn.relu(bn6, name='relu6')

    # conv7 output is of size M/128 x N/128 x 64
    conv7 = tf.layers.conv2d(inputs=bn6, filters=64, kernel_size=[3, 3], strides=(2, 2), padding="same", activation=None, name='conv7')

    # bn6 output is of size M/128 x N/128 x 64
    bn7 = tf.layers.batch_normalization(inputs=conv7, name='bn7')
    bn7 = tf.nn.relu(bn7, name='relu7')

    # conv7 output is of size M/256 x N/256 x 2
    conv7 = tf.layers.conv2d(inputs=bn7, filters=2, kernel_size=[3, 3], strides=(2, 2), padding="same", activation=None, name='conv8')

    # fc1
    fc1_flat = tf.reshape(conv7, [-1, ImageSize[0]*ImageSize[1]*2/(256*256)])
    pr1 = tf.layers.dense(fc1_flat, 2, activation=None)
    
    return pr1


#def loss(SegPred, SegReal):
    """
    Inputs: 
    SegPred is the predicted segmentation
    SegReal is the ground truth segmentation
    Outputs:
    Loss value computed as absolute average difference
    """
#    return tf.reduce_mean(tf.abs(SegPred-SegReal))

