from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np


def ConvBNReLUBlock(inputs = None, filters = None, kernel_size = None, strides = None, padding = None):
    conv =  Conv(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, padding = padding)
    bn = BN(conv)
    Output = ReLU(bn)
    return Output

def ConvTransposeBNReLUBlock(inputs = None, filters = None, kernel_size = None, strides = None, padding = None):
    conv =  ConvTranspose(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, padding = padding)
    bn = BN(conv)
    Output = ReLU(bn)
    return Output

def Conv(inputs = None, filters = None, kernel_size = None, strides = None, padding = None, activation=None, name=None):
    Output = tf.layers.conv2d(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, padding = padding, activation=activation, name=name) # kernel_constraint=tf.keras.constraints.UnitNorm(axis=0),
    return Output

def ConvTranspose(inputs = None, filters = None, kernel_size = None, strides = None, padding = None, activation=None, name=None):
    Output = tf.layers.conv2d_transpose(inputs = inputs, filters = filters, kernel_size = kernel_size, \
                              strides = strides, padding = padding, activation=activation, name=name) # kernel_constraint=tf.keras.constraints.UnitNorm(axis=0),
    return Output

def BN(inputs = None):
    # inputs = tf.layers.batch_normalization(inputs = inputs) 
    return inputs
    
def ReLU(inputs = None):
    Output = tf.nn.relu(inputs)
    return Output

def ResBlock(inputs = None, filters = None, kernel_size = None, strides = None, padding = None):
    Net = ConvBNReLUBlock(inputs = inputs, filters = filters, kernel_size = kernel_size, padding = padding, strides=(1,1))
    Net = Conv(inputs = Net, filters = filters, kernel_size = kernel_size, padding = padding, strides=(1,1), activation=None)
    Net = BN(inputs = Net)
    if Net.shape == inputs.shape:
        Net = tf.add(Net, inputs)
    else:
        Net1 = Conv(inputs = inputs, filters = filters, kernel_size = (1,1), padding = padding, strides=strides, activation=None)
        Net = tf.add(Net, Net1)
    Net = ReLU(inputs = Net)
    return Net

def ResBlockTranspose(inputs = None, filters = None, kernel_size = None, strides = None, padding = None):
    Net = ConvTransposeBNReLUBlock(inputs = inputs, filters = filters, kernel_size = kernel_size, padding = padding, strides=strides)
    Net = ConvTranspose(inputs = Net, filters = filters, kernel_size = kernel_size, padding = padding, strides=(1,1), activation=None)
    Net = BN(inputs = Net)
    # if Net.shape[3] == inputs.shape[3]:
    #     Net = tf.add(Net, inputs)
    # else:
    #     Net1 = Conv(inputs = inputs, filters = filters, kernel_size = (1,1), padding = padding, strides=strides, activation=None)
    #     Net = tf.add(Net1, Net)
    Net = ReLU(inputs = Net)
    return Net

def model(img, layers, args):
    print(img.shape)
    net = img
    for filters in layers:
        net = ResBlock(inputs = net, filters = filters, kernel_size = [3,3], strides = (1,1), padding="same")
        net = tf.layers.max_pooling2d(net, 2, strides=(2,2), padding="same")
        print(net.shape)

    if args.upsample:
      print("Upsampling..")
      layers.reverse()
      for filters in layers:
          net = ResBlockTranspose(inputs = net, filters = filters, kernel_size = [3,3], strides = (2,2), padding="same")
          print(net.shape)
      net = Conv(inputs = net, filters = args.outchannels, kernel_size = [3,3], strides = (1,1), padding = "same")
    out = tf.identity(net, name="output")
    print("Output shape is : ",out.shape)
    return out

# def small_model(Img):
#     net = Conv(inputs = Img, filters = 16, kernel_size = [3,3], padding = "same", strides=(1,1), activation=None)
#     net = Conv(inputs = net, filters = 1, kernel_size = [3,3], padding = "same", strides=(1,1), activation=None)
#     outs1 = tf.identity(net, name="output")
#     return outs1

# def model_(Img):
#     """
#     Inputs: 
#     Img is a MiniBatch of the current image
#     ImageSize - Size of the Image
#     Outputs:
#     pr1 is the predicted output of homing vector for a MiniBatch
#     """
#     # Img is of size MxNx3
    
#     # conv1 output is of size M/2 x N/2 x 8
#     conv1 = tf.layers.conv2d(inputs=Img, filters=16, kernel_size=[5, 5], strides=(1, 1), padding="same", activation=None, name='conv1T')    
#     # bn1 output is of size M x N x 8
#     bn1 = tf.nn.relu(conv1, name='relu1T')
    
#     conv2_1 = tf.layers.conv2d(inputs=bn1, filters=16, kernel_size=[5, 5], strides=(1, 1), padding="same", activation=None, name='conv2T')
#     # bn1 output is of size M x N x 8
#     bn2_1 = tf.nn.relu(conv2_1, name='relu2T')

#     conv2_2 = tf.layers.conv2d(inputs=bn2_1, filters=16, kernel_size=[5, 5], strides=(1, 1), padding="same", activation=None, name='conv2T_1')
#     # bn1 output is of size M x N x 8
#     shortcut = tf.layers.conv2d(inputs=conv1, filters=16, kernel_size=1, strides=1, padding="same")
#     bn2_2 = tf.nn.relu(conv2_2 + shortcut, name='relu2T_1')

#     conv3 = tf.layers.conv2d(inputs=bn2_2, filters=16, kernel_size=[5, 5], strides=(2, 2), padding="same", activation=None, name='conv3T')

#     # bn1 output is of size M x N x 8
#     bn3 = tf.nn.relu(conv3, name='relu3T')
    
#     # deconv1 output is of size MxNx3
#     up1 = tf.layers.conv2d_transpose(inputs=bn3, filters=2, kernel_size=[5, 5], strides=(4, 4), padding="valid", activation=None, name='up1')
#     up2 = tf.layers.conv2d_transpose(inputs=up1, filters=2, kernel_size=[5, 5], strides=(4, 4), padding="valid", activation=None, name='up2')
    
#     outs1 = tf.identity(up2, name="output")
#     return outs1

def main(args):
  # input_size = [1, 512//args.InScale, 256//args.InScale, 1]
  input_size = [int(i) for i in args.Shape[1:-1].split(",")]
  
  print("Input size is -- ",input_size)
  x = tf.placeholder(tf.float32, input_size, name="input")
  layers = [int(i) for i in args.layers[1:-1].split(",")]
  print("Layers are : ",layers)
  
  # output_ = model_(x)
  output_ = model(x, layers, args)
  # output_ = small_model(x)

  graph_location = "tf_model"
  print('Saving graph to: %s' % graph_location)
  in_np = np.ones(input_size, dtype=np.float32)
  ins = tf.convert_to_tensor(in_np)
  saver = tf.train.Saver()
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    feedDict = {x:in_np}
    outp = sess.run([output_], feed_dict=feedDict)
    print("Output shape : ",outp[0].shape)
    print("Output data type : ",outp[0].dtype)
    save_path = saver.save(sess, graph_location + "/mnist_model")

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-s','--Shape', type=str, default="[1,256,256,1]", help='Scaling of input dimention (no space between , and numbers)')
  parser.add_argument('-l','--layers', type=str, default="[4,8,16]", help='Number of encoding layers')
  parser.add_argument('-oc','--outchannels', type=int, default=4, help='Number of output channels')
  parser.add_argument('-u','--upsample', type=bool, default=True, help="Upsampling of the network")
  args = parser.parse_args()
  main(args)
  # FLAGS, unparsed = parser.parse_known_args()
  # tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
  