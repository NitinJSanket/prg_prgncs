#!/usr/bin/env python

# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (sudo)
# termcolor, do (pip install termcolor)
# tqdm, do (pip install tqdm)

# TODO:
# Clean print statements
# Global step only loss/epoch on tensorboard
# Print Num parameters in model as a function
# Clean comments
# Confusion Matrix

import tensorflow as tf
import cv2
import sys
import os
import glob
import Misc.ImageUtils as iu
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from Network.NetworkTeacher import CIFAR10Teacher
from Misc.MiscUtils import *
import numpy as np
import time
import argparse
import shutil
from StringIO import StringIO
import string
from termcolor import colored, cprint
import math as m
from tqdm import tqdm

# Don't generate pyc codes
sys.dont_write_bytecode = True

def SetupAll(BasePath):
    """
    Inputs: 
    BasePath is the base path where Images are saved without "/" at the end
    Outputs:
    DirNames - Full path to all image files without extension
    Train/Val/Test - Idxs of all the images to be used for training/validation (held-out testing in this case)/testing
    Ratios - Ratios is a list of fraction of data used for [Train, Val, Test]
    CheckPointPath - Path to save checkpoints/model
    OptimizerParams - List of all OptimizerParams: depends on Optimizer
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    ImageSize - Size of the image
    NumTrain/Val/TestSamples - length(Train/Val/Test)
    NumTestRunsPerEpoch - Number of passes of Val data with MiniBatchSize 
    Train/Val/TestLabels - Labels corresponding to Train/Val/Test
    """
    # Setup DirNames
    DirNamesTrain, DirNamesTest =  SetupDirNames(BasePath)

    # Read and Setup Labels
    LabelsPathTrain = './TxtFiles/LabelsTrain.txt'
    LabelsPathTest = './TxtFiles/LabelsTest.txt'
    TrainLabels, TestLabels = ReadLabels(LabelsPathTrain, LabelsPathTest)


    # Setup Neural Net Params
    CheckPointPath = '../Checkpoints/' # Path to save checkpoints
    # If CheckPointPath doesn't exist make the path
    if(not (os.path.isdir(CheckPointPath))):
       os.makedirs(CheckPointPath)

    # List of all OptimizerParams: depends on Optimizer
    # For ADAM Optimizer: [LearningRate, Beta1, Beta2, Epsilion]
    UseDefaultFlag = 0 # Set to 0 to use your own params, do not change default parameters
    if UseDefaultFlag:
        # Default Parameters
        OptimizerParams = [1e-3, 0.9, 0.999, 1e-8]
    else:
        # Custom Parameters
        OptimizerParams = [1e-4, 0.9, 0.999, 1e-8]   
        
    # Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    SaveCheckPoint = 100 
    # Number of passes of Val data with MiniBatchSize 
    NumTestRunsPerEpoch = 5
    
    # Image Input Shape
    ImageSize = [32, 32]
    NumTrainSamples = len(DirNamesTrain)
    NumTestSamples = len(DirNamesTest)
    return DirNamesTrain, DirNamesTest, CheckPointPath, OptimizerParams,\
        SaveCheckPoint, ImageSize, NumTrainSamples, NumTestSamples,\
        NumTestRunsPerEpoch, TrainLabels, TestLabels

def ReadLabels(LabelsPathTrain, LabelsPathTest):
    if(not (os.path.isfile(LabelsPathTrain))):
        print('ERROR: Train Labels do not exist in '+LabelsPathTrain)
        sys.exit()
    else:
        TrainLabels = open(LabelsPathTrain, 'r')
        TrainLabels = TrainLabels.read()
        TrainLabels = map(float, TrainLabels.split())
        # TrainLabels = tf.cast(TrainLabels, tf.int32)

    if(not (os.path.isfile(LabelsPathTest))):
        print('ERROR: Test Labels do not exist in '+LabelsPathTest)
        sys.exit()
    else:
        TestLabels = open(LabelsPathTest, 'r')
        TestLabels = TestLabels.read()
        TestLabels = map(float, TestLabels.split())
        # TestLabels = tf.cast(TestLabels, tf.int32)

    return TrainLabels, TestLabels
    

def SetupDirNames(BasePath): 
    """
    Inputs: 
    BasePath is the base path where Images are saved without "/" at the end
    Outputs:
    Writes a file ./TxtFiles/DirNames.txt with full path to all image files without extension
    """
    # Don't execute if file exists
    if(not (os.path.isfile('./TxtFiles/DirNamesTrain.txt'))):
        DirNamesTrain = open('./TxtFiles/DirNamesTrain.txt', 'w')
        
        CurrPath = BasePath + '/Train/'
        NumImagesTrain = len(glob.glob(CurrPath+'*.png'))
        for ImageNum in range(1, NumImagesTrain+1):
            DirNamesTrain.write(CurrPath+str(ImageNum)+'\n')
        
        DirNamesTrain.close()

    else:
        print('WARNING: DirNamesTrain.txt File exists')
        # Read DirNames once processed
        DirNamesTrain = ReadDirNames('./TxtFiles/DirNamesTrain.txt')

    if(not (os.path.isfile('./TxtFiles/DirNamesTest.txt'))):
        DirNamesTest = open('./TxtFiles/DirNamesTest.txt', 'w')
        
        CurrPath = BasePath + '/Test/'
        NumImagesTest = len(glob.glob(CurrPath+'*.png'))
        for ImageNum in range(1, NumImagesTest+1):
            DirNamesTest.write(CurrPath+str(ImageNum)+'\n')
        
        DirNamesTest.close()

    else:
        print('WARNING: DirNamesTest.txt File exists')
        # Read DirNames once processed
        DirNamesTest = ReadDirNames('./TxtFiles/DirNamesTest.txt')
    
    
    return DirNamesTrain, DirNamesTest

    
def PerturbImage(I1, PerturbNum):
    """
    Data Augmentation
    Inputs: 
    I1 is the input image
    PerturbNum choses type of Perturbation where it ranges from 0 to 5
    0 - No perturbation
    1 - Random gaussian Noise
    2 - Random hue shift
    3 - Random saturation shift
    4 - Random gamma shift
    Outputs:
    Perturbed Image I1
    """
    if(PerturbNum == 0):
        pass
    elif(PerturbNum == 1):
        I1 = iu.GaussianNoise(I1)
    elif(PerturbNum == 2):
        I1 = iu.ShiftHue(I1)
    elif(PerturbNum == 3):
        I1 = iu.ShiftSat(I1)
    elif(PerturbNum == 4):
        I1 = iu.Gamma(I1)
        
    return I1

def ReadDirNames(Path):
    """
    Inputs: 
    Path is the path of the file you want to read
    Outputs:
    DirNames is the data loaded from ./TxtFiles/DirNames.txt which has full path to all image files without extension
    """
    # Read text files
    DirNames = open(Path, 'r')
    DirNames = DirNames.read()
    DirNames = DirNames.split()
    return DirNames
    
def GenerateBatch(DirNamesTrain, TrainLabels, ImageSize, MiniBatchSize, PerEpochCounter):
    """
    Inputs: 
    DirNames - Full path to all image files without extension
    NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
    TrainLabels - Labels corresponding to Train
    NOTE that TrainLabels can be replaced by Val/TestLabels for generating batch corresponding to validation (held-out testing in this case)/testing
    ImageSize - Size of the Image
    MiniBatchSize is the size of the MiniBatch
    Outputs:
    I1Batch - Batch of I1 images after standardization and cropping/resizing to ImageSize
    HomeVecBatch - Batch of Homing Vector labels
    """
    I1Batch = []
    LabelBatch = []
    
    ImageNum = 0
    # RandIdxAll = range(PerEpochCounter*MiniBatchSize,(PerEpochCounter+1)*MiniBatchSize)
    # count = 0
    while ImageNum < MiniBatchSize:
        # Generate random image
        RandIdx = random.randint(0, len(DirNamesTrain)-1)
        # RandIdx = RandIdxAll[count]
        
        RandImageName = DirNamesTrain[RandIdx]+'.png'
        RandImageNameWithoutExt = DirNamesTrain[RandIdx]
        RandImageNum = RandImageNameWithoutExt.split('/')
        CurrPath = '/'.join(map(str, RandImageNum[0:-1]))+'/'
        RandImageNum = int(RandImageNum[-1])    
        ImageNum += 1
    
        I1 = cv2.imread(RandImageName)
        
        # Always get a random crop to fit the size of the network
        # I1 = iu.RandomCrop(I1, ImageSize)
        
        # Apply a random perturbation
        PerturbNum = random.randint(0, 5)
        I1 = PerturbImage(I1, PerturbNum)

        # Standardize Inputs as given by Inception v3 paper
        # MAYBE: Find Mean of Dataset or use from ImageNet
        # MAYBE: Normalize Dataset
        # https://stackoverflow.com/questions/42275815/should-i-substract-imagenet-pretrained-inception-v3-model-mean-value-at-inceptio
        I1S = iu.StandardizeInputs(np.float32(I1))
        Label = convertToOneHot(TrainLabels[RandIdx], 10)

        # Append All Images and Mask
        I1Batch.append(I1S)
        LabelBatch.append(Label)
        # count += 1
        
    return I1Batch, LabelBatch


def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, NumTestSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print('Number of Epochs Training will run for ' + str(NumEpochs))
    print('Factor of reduction in training data is ' + str(DivTrain))
    print('Mini Batch Size ' + str(MiniBatchSize))
    print('Number of Training Images ' + str(NumTrainSamples))
    print('Number of Testing Images ' + str(NumTestSamples))
    if LatestFile is not None:
        print('Loading latest checkpoint with the name ' + LatestFile)              

def Accuracy(Pred, GT):
    """
    Inputs: 
    HomingVecPred is the output of the neural network
    HomingVecReal is the ground truth homing vector
    Outputs:
    NOT IMPLEMENTED YET! 
    """
    return (np.sum(Pred==GT)*100.0)
    

def TrainOperation(ImgPH, LabelPH, DirNamesTrain, DirNamesTest, TrainLabels, TestLabels, NumTrainSamples, ImageSize,
                   NumEpochs, MiniBatchSize, OptimizerParams, SaveCheckPoint, CheckPointPath, NumTestRunsPerEpoch, DivTrain, LatestFile):
    """
    Inputs: 
    ImgPH is the Input Image placeholder
    HomingVecPH is the ground truth  homing vector placeholder
    DirNames - Full path to all image files without extension
    Train/Val - Idxs of all the images to be used for training/validation (held-out testing in this case)
    Train/ValLabels - Labels corresponding to Train/Val
    NumTrain/ValSamples - length(Train/Val)
    ImageSize - Size of the image
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    OptimizerParams - List of all OptimizerParams: depends on Optimizer
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    NumTestRunsPerEpoch - Number of passes of Val data with MiniBatchSize 
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of data
    LatestFile - Latest checkpointfile to continue training
    Outputs:
    Saves Trained network in CheckPointPath
    """      
    # Predict output with forward pass
    prLogits, prSoftMax = CIFAR10Teacher(ImgPH, ImageSize, MiniBatchSize)

    with tf.name_scope('Loss'):
        # Cross-Entropy Loss
        # 10 is used because of 10 classes
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prLogits, labels=LabelPH))
        
    with tf.name_scope('Adam'):
        Optimizer = tf.train.AdamOptimizer(learning_rate=OptimizerParams[0], beta1=OptimizerParams[1],
                                           beta2=OptimizerParams[2], epsilon=OptimizerParams[3]).minimize(loss)
        #Optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-8).minimize(loss)
        #Optimizer = tf.train.MomentumOptimizer(learning_rate=1e-3, momentum=0.9, use_nesterov=True).minimize(loss)

    # Tensorboard
    LogsPath = 'Logs/'
    # Create a summary to monitor loss tensor
    tf.summary.scalar('LossEveryIter', loss)
    # Merge all summaries into a single operation
    MergedSummaryOP = tf.summary.merge_all()
    
    
    AllEpochLoss = [0.0]
    EachIterLoss = [0.0]
    # Setup Saver
    Saver = tf.train.Saver()

    # Open File
    EpochLossTxt = open('./TxtFiles/EpochLoss.txt', 'w')
    
    with tf.Session() as sess:       
        if LatestFile is not None:
            Saver.restore(sess, CheckPointPath + LatestFile + '.ckpt')
            # Extract only numbers from the name
            StartEpoch = int(''.join(c for c in LatestFile.split('a')[0] if c.isdigit()))
            print('Loaded latest checkpoint with the name ' + LatestFile + '....')
        else:
            sess.run(tf.global_variables_initializer())
            StartEpoch = 0
            print('New model initialized....')

        # Tensorboard
        Writer = tf.summary.FileWriter(LogsPath, graph=tf.get_default_graph())
            
        TotalTimeElapsed = 0.0
        TimerOverall = tic()
        for Epochs in tqdm(range(StartEpoch, NumEpochs)):
            EpochLoss = 0.0
            Timer1 = tic()
            NumIterationsPerEpoch = int(NumTrainSamples/MiniBatchSize/DivTrain)
            for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
                print('Epoch ' + str(Epochs) + ' PerEpochCounter ' + str(PerEpochCounter))
                Timer2 = tic()
                I1Batch, LabelBatch = GenerateBatch(DirNamesTrain, TrainLabels, ImageSize, MiniBatchSize, PerEpochCounter)
                FeedDict = {ImgPH: I1Batch, LabelPH: LabelBatch}
                _, LossThisBatch, Summary = sess.run([Optimizer, loss, MergedSummaryOP], feed_dict=FeedDict)
                
                # Tensorboard
                Writer.add_summary(Summary, Epochs*NumIterationsPerEpoch + PerEpochCounter)
                # If you don't flush the tensorboard doesn't update until a lot of iterations!
                Writer.flush()

                # Calculate and print Train accuracy (also called EpochLoss) every epoch
                EpochLoss += LossThisBatch

                # Save All losses
                EachIterLoss.append(LossThisBatch)

                TimeLastMiniBatch = toc(Timer2)

                # Print LossThisBatch
                print('LossThisBatch is  '+ str(LossThisBatch))
                
                # Save checkpoint every some SaveCheckPoint's iterations
                if PerEpochCounter % SaveCheckPoint == 0:
                    # Save the Model learnt in this epoch
                    SaveName =  CheckPointPath + str(Epochs) + 'a' + str(PerEpochCounter) + 'model.ckpt'
                    Saver.save(sess,  save_path=SaveName)
                    print(SaveName + ' Model Saved...')

                # Print timing information
                EstimatedTimeToCompletionThisEpoch = float(TimeLastMiniBatch)*float(NumIterationsPerEpoch-PerEpochCounter-1.0)
                EstimatedTimeToCompletionTotal = float(TimeLastMiniBatch)*float(NumIterationsPerEpoch-PerEpochCounter-1.0) +\
                                                 float(TimeLastMiniBatch)*float(NumIterationsPerEpoch-1.0)*float(NumEpochs-Epochs)
                TotalTimeElapsed = toc(TimerOverall)
                print('Percentage complete in total epochs ' + str(float(Epochs+1)/float(NumEpochs-StartEpoch+1)*100.0))
                print('Percentage complete in this Train epoch ' + str(float(PerEpochCounter)/float(NumIterationsPerEpoch)*100.0))
                print('Last MiniBatch took '+ str(TimeLastMiniBatch) + ' secs, time taken till now ' + str(TotalTimeElapsed) + \
                      ' estimated time to completion of this epoch is ' + str(EstimatedTimeToCompletionThisEpoch))
                print('Estimated Total time remaining is ' + str(EstimatedTimeToCompletionTotal))
                
            TimeLastEpoch = toc(Timer1)
            EstimatedTimeToCompletion = float(TotalTimeElapsed)/float(Epochs+1.0)*float(NumEpochs-Epochs-1.0)
                
            # Save Each Epoch loss
            AllEpochLoss.append(EpochLoss)
            
            # Save model every epoch
            SaveName = CheckPointPath + str(Epochs) + 'model.ckpt'
            Saver.save(sess, save_path=SaveName)
            print(SaveName + ' Model Saved...')
            
            # Calculate and print Test accuracy every epoch
            TestLoss = 0.0
            for PerTestEpochCounter in tqdm(range(NumTestRunsPerEpoch)):
                print('PerEpochCounter' + str(PerTestEpochCounter))
                I1Batch, LabelBatch = GenerateBatch(DirNamesTest, TestLabels, ImageSize, MiniBatchSize, PerEpochCounter)
                FeedDict = {ImgPH: I1Batch, LabelPH: LabelBatch}
                TestLossThisBatch = sess.run([loss], FeedDict)
                # Calculate and print Test accuracy (also called TestLoss) every epoch
                # TestLossThisBatch is a list so extract the value
                TestLoss += TestLossThisBatch[0]
                # Print LossThisBatch
                print('TestLossThisBatch is  ' + str(TestLossThisBatch[0]))
                # Print percentage complete in total epochs
                print('Percentage complete in Test epochs '+ str(float(PerTestEpochCounter)/float(NumTestRunsPerEpoch)*100.0))
            TestLoss /= (NumTestRunsPerEpoch)
            print('Test loss per Image ' + str(TestLoss))

            # Print timing information every epoch
            #print('Epoch ' + str(Epochs) + ' completed out of ' + str(NumEpochs) + ' loss:' + str(EpochLoss))
            PrintStatement = 'Epoch ' + str(Epochs) + ' completed out of ' + str(NumEpochs) + ' loss:' + str(EpochLoss)
            cprint(PrintStatement, 'yellow')
            print('Last Epoch took ' + str(TimeLastEpoch) +  ' secs, time taken till now, ' + str(TotalTimeElapsed) +\
                  ' estimated time to completion of this epoch is ' + str(EstimatedTimeToCompletion))
            # Save Epoch Loss to file
            # TODO: Make the filename a parameter
            # Create txt file if doesn't exist
            #if(not (os.path.isfile('./EpochLoss.txt'))):
            EpochLossTxt.write(str(EpochLoss)+'\n')
            print('------------------------------------------------')
    # Close txt file when done
    EpochLossTxt.close()

    # Tensorboard
            #tf.scalar_summary("EpochLoss", EpochLoss)    



    
def main():
    """
    Inputs: 
    None
    Outputs:
    Runs the Training and testing code based on the Flag
    """
    # TODO: Make LogDir
    # TODO: Make logging file a parameter
    # TODO: Time to complete print

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--BasePath', default='/home/ncs/Nitin/ncsdk/Nitin/SpectralCompression/CIFAR10', help='Base path of images, Default:/media/nitin/Research/Homing/DataMapping/AllFrames')
    Parser.add_argument('--NumEpochs', type=int, default=1, help='Number of Epochs to Train for, Default:200')
    Parser.add_argument('--DivTrain', type=int, default=5000, help='Factor to reduce Train data by per epoch, Default:1')
    Parser.add_argument('--MiniBatchSize', type=int, default=8, help='Size of the MiniBatch to use, Default:48')
    Parser.add_argument('--RemoveTxtFiles', type=int, default=0, help='Remove Text files and recreate them?, Default:0')
    Parser.add_argument('--LoadCheckPoint', type=int, default=0, help='Load Model from latest Checkpoint from CheckPointsPath?, Default:0')
    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    BasePath = Args.BasePath
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    RemoveTxtFiles = Args.RemoveTxtFiles
    LoadCheckPoint = Args.LoadCheckPoint
    
    if RemoveTxtFiles==1:
        shutil.copyfile('./TxtFiles/DirNamesTrain.txt', './TxtFiles/DirNamesTrainOLD.txt')
        shutil.copyfile('./TxtFiles/DirNamesTest.txt', './TxtFiles/DirNamesTestOLD.txt')
        shutil.copyfile('./TxtFiles/LabelsTrain.txt', './TxtFiles/LabelsTrainOLD.txt')
        shutil.copyfile('./TxtFiles/LabelsTest.txt', './TxtFiles/LabelsTestOLD.txt')
        os.remove('./TxtFiles/DirNamesTrain.txt')
        os.remove('./TxtFiles/DirNamesTest.txt')
        os.remove('./TxtFiles/LabelsTrain.txt')
        os.remove('./TxtFiles/LabelsTest.txt')
        print('Old DirNamesTrain, DirNamesTest, LabelsTrain, LabelsTest text files are saved as ./TxtFiles/DirNamesTrainOLD.txt, ./TxtFiles/DirNamesTestOLD.txt, ./TxtFiles/LabelsTrainOLD.txt and ./TxtFiles/LabelsTestOLD.txt')
        print('WARNING: DirNamesTrain, DirNamesTest, LabelsTrain and LabelsTest text files deleted')


    # Setup all needed parameters including file reading
    DirNamesTrain, DirNamesTest, CheckPointPath, OptimizerParams,\
    SaveCheckPoint, ImageSize, NumTrainSamples, NumTestSamples,\
    NumTestRunsPerEpoch, TrainLabels, TestLabels = SetupAll(BasePath)

    # Find Latest Checkpoint File
    if LoadCheckPoint==1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None
    
    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, NumTestSamples, LatestFile)

    # Define PlaceHolder variables for Input and Predicted output
    ImgPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, ImageSize[0], ImageSize[1], 3), name='Input')
    LabelPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, 10)) # OneHOT labels
    
    TrainOperation(ImgPH, LabelPH, DirNamesTrain, DirNamesTest, TrainLabels, TestLabels, NumTrainSamples, ImageSize,
                   NumEpochs, MiniBatchSize, OptimizerParams, SaveCheckPoint, CheckPointPath, NumTestRunsPerEpoch, DivTrain, LatestFile)
        
    
if __name__ == '__main__':
    main()
 
