#!/usr/bin/env python

# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)


# TODO:
# Add Train and Test accuracy

import tensorflow as tf
import cv2
import os
import sys
import glob
import Misc.ImageUtils as iu
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from Network.NetworkTeacher import CIFAR10Teacher
from Network.NetworkStudent import CIFAR10Student
from Misc.MiscUtils import *
import numpy as np
import time
import argparse
import shutil
from StringIO import StringIO
import string
import math as m
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# Don't generate pyc codes
sys.dont_write_bytecode = True

def SetupAll(BasePath):
    """
    Inputs: 
    BasePath - Path to images
    Outputs:
    ImageSize - Size of the Image
    DataPath - Paths of all images where testing will be run on
    """   
    # Image Input Shape
    ImageSize = [32, 32]
    DataPath = []
    NumImages = len(glob.glob(BasePath+'*.png'))
    SkipFactor = 1
    for count in range(1,NumImages+1,SkipFactor):
        DataPath.append(BasePath + str(count) + '.png')

    return ImageSize, DataPath
    
def ReadImages(ImageSize, DataPath):
    """
    Inputs: 
    ImageSize - Size of the Image
    DataPath - Paths of all images where testing will be run on
    Outputs:
    I1Combined - I1 image after standardization and cropping/resizing to ImageSize
    I1 - Original I1 image for visualization purposes only
    """
    
    ImageName = DataPath
    
    I1 = cv2.imread(ImageName)
    
    if(I1 is None):
        # OpenCV returns empty list if image is not read! Like WTF!
        print('ERROR: Image I1 cannot be read')
        sys.exit()
        
        
    # Always get a random crop to fit the size of the network
    I1 = iu.RandomCrop(I1, ImageSize)

    # Resize Image to fit size of the network
    # I1 = iu.Resize(I1, ImageSize)
        
    # Standardize Inputs as given by Inception v3 paper
    # MAYBE: Find Mean of Dataset or use from ImageNet
    # MAYBE: Normalize Dataset
    # https://stackoverflow.com/questions/42275815/should-i-substract-imagenet-pretrained-inception-v3-model-mean-value-at-inceptio
    I1S = iu.StandardizeInputs(np.float32(I1))

    I1Combined = np.expand_dims(I1S, axis=0)

    return I1Combined, I1
                

def TestOperation(ImgPH, ImageSize, ModelPath, DataPath):
    """
    Inputs: 
    ImgPH is the Input Image placeholder
    HomingVecPH is the ground truth  homing vector placeholder
    NumTrainSamples - length(Train)
    ModelPath - Path to load trained model from
    DataPath - Paths of all images where testing will be run on
    Outputs:
    Predictions written to ./TxtFiles/PredOut.txt
    """
    Length = ImageSize[0]
    # Predict output with forward pass, MiniBatchSize for Test is 1
    _, prSoftMaxT = CIFAR10Teacher(ImgPH, ImageSize, 1)
    _, prSoftMaxS = CIFAR10Student(ImgPH, ImageSize, 1)

    # Setup Saver
    Saver = tf.train.Saver()

    
    with tf.Session() as sess:
        Saver.restore(sess, ModelPath)
        print('Number of parameters in this model are %d ' % np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
        
        OutSaveT = open('./TxtFiles/PredOutT.txt', 'w')
        OutSaveS = open('./TxtFiles/PredOutS.txt', 'w')

        for count in tqdm(range(np.size(DataPath))):
            Timer1 = tic()
            
            DataPathNow = DataPath[count]
            Img, ImgOrg = ReadImages(ImageSize, DataPathNow)
            FeedDict = {ImgPH: Img}
            PredT = np.argmax(sess.run(prSoftMaxT, FeedDict))
            PredS = np.argmax(sess.run(prSoftMaxS, FeedDict))

            OutSaveT.write(str(PredT)+'\n')
            OutSaveS.write(str(PredS)+'\n')
            
            TimePerPass = toc(Timer1)
            
            # print('Single Pass took ' + str(TimePerPass) + ' secs')
            
        OutSaveT.close()
        OutSaveS.close()

def Accuracy(Pred, GT):
    """
    Inputs: 
    HomingVecPred is the output of the neural network
    HomingVecReal is the ground truth homing vector
    Outputs:
    NOT IMPLEMENTED YET! 
    """
    return (np.sum(np.array(Pred)==np.array(GT))*100.0/len(Pred))

def ReadLabels(LabelsPathTest, LabelsPathPred):
    if(not (os.path.isfile(LabelsPathTest))):
        print('ERROR: Test Labels do not exist in '+LabelsPathTest)
        sys.exit()
    else:
        LabelTest = open(LabelsPathTest, 'r')
        LabelTest = LabelTest.read()
        LabelTest = map(float, LabelTest.split())

    if(not (os.path.isfile(LabelsPathPred))):
        print('ERROR: Pred Labels do not exist in '+LabelsPathPred)
        sys.exit()
    else:
        LabelPred = open(LabelsPathPred, 'r')
        LabelPred = LabelPred.read()
        LabelPred = map(float, LabelPred.split())
        
    return LabelTest, LabelPred

def ConfusionMatrix(LabelTest, LabelPred):
    # LabelPred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=LabelTest,  # True class for test-set.
                          y_pred=LabelPred)  # Predicted class.

    # Print the confusion matrix as text.
    for i in range(10):
        print(str(cm[i, :]) + ' ({0})'.format(i))

    # Print the class-numbers for easy reference.
    class_numbers = [" ({0})".format(i) for i in range(10)]
    print("".join(class_numbers))

    print('Accuracy: '+ str(Accuracy(LabelPred, LabelTest)), '%')

        
def main():
    """
    Inputs: 
    None
    Outputs:
    Runs Testing code
    """
    # TODO: Make LogDir
    # TODO: Display time to end and cleanup other print statements with color
    # TODO: Make logging file a parameter

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ModelPath', dest='ModelPath', default='/home/nitin/Research/Checkpoints5/199model.ckpt', help='Path to load latest model from, Default:ModelPath')
    Parser.add_argument('--BasePath', dest='BasePath', default='/home/nitin/Research/SpectralCompression/CIFAR10/Test/', help='Path to load images from, Default:BasePath')
    Args = Parser.parse_args()
    ModelPath = Args.ModelPath
    BasePath = Args.BasePath

    # Setup all needed parameters including file reading
    ImageSize, DataPath = SetupAll(BasePath)

    # Define PlaceHolder variables for Input and Predicted output
    ImgPH = tf.placeholder('float', shape=(1, ImageSize[0], ImageSize[1], 3))

    TestOperation(ImgPH, ImageSize, ModelPath, DataPath)

    # Plot Confusion Matrix
    LabelsPathTest = './TxtFiles/LabelsTest.txt'
    LabelsPathPredT = './TxtFiles/PredOutT.txt'
    LabelsPathPredS = './TxtFiles/PredOutS.txt'
    LabelTest, LabelPredT = ReadLabels(LabelsPathTest, LabelsPathPredT)
    _, LabelPredS = ReadLabels(LabelsPathTest, LabelsPathPredS)
    print('Teacher')
    ConfusionMatrix(LabelTest, LabelPredT)
    print('Student')
    ConfusionMatrix(LabelTest, LabelPredS)
 
if __name__ == '__main__':
    main()
 
