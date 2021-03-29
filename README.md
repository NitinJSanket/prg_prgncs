# prg_prgncs
PRG's Setup of Intel Neural Compute Stick

# Installation
- [Step 1](https://software.intel.com/content/www/us/en/develop/articles/get-started-with-neural-compute-stick.html)
- [Step 2](https://docs.openvinotoolkit.org/2018_R5/_docs_install_guides_installing_openvino_linux.html) (Needs PC Restart after this!)
- Step 3: `sudo bash /opt/intel/openvino/deployment_tools/demo/setupvars.sh`
- Step 4: `cd /opt/intel/openvino/deployment_tools/demo && ./demo_benchmark_app.sh -d MYRIAD`
- NOTE: All tests were run at USB 2.0


# Custom Model Conversion
- Step 1: `cd NCS-2 && python tf_model.py -s [1,256,256,1] -l [4,8,16]`
- Step 2: `python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_meta_graph mnist_model.meta --input_shape [1,256,256,1]  --data_type FP16`
- Step 3: `source /opt/intel/openvino/bin/setupvars.sh`
- Step 4: `python3 movi2.py`

# Speed Test
Image size is [BWHC] [ batch size x width x height x color channels].

| Image size  |      Filters in subsequent layers      |  FPS |
|----------|:-------------:|------:|
| 1 x 256 x 256 x 1 | [8,16,32,64] | 200 ms |
| 1 x 512 x 256 x 1 | [16,32,64,128] | 320 ms |
| 1 x 512 x 256 x 1 | [32,64,128,256] | Exception: Status.ERROR |
| 1 x 256 x 128 x 1 | [32,64,128,256] | Exception: Status.ERROR |
| 1 x 256 x 128 x 1 | [16,32,64,128] | 89 ms |
| 1 x 256 x 128 x 1 | [8,16,32,64] | 57 ms |
| 1 x 256 x 128 x 1 | [4,8,16,32] | 43 ms |



In the following table filters shows that many number of convolutional layers while downsampling and same goes for upsampling also. Check if it is USB 2.0 or USB 3.0. You can check it using `lsusb`. This tests were done with USB 2.0. Image size is [BWHC] [ batch size x width x height x color channels].

| Image size  |      Filters in subsequent layers      |  Time |
|----------|:-------------:|------:|
| 1 x 512 x 256 x 1 | [8,16,32,64] | 200 ms |
| 1 x 512 x 256 x 1 | [16,32,64,128] | 320 ms |
| 1 x 512 x 256 x 1 | [32,64,128,256] | Exception: Status.ERROR |
| 1 x 256 x 128 x 1 | [32,64,128,256] | Exception: Status.ERROR |
| 1 x 256 x 128 x 1 | [16,32,64,128] | 89 ms |
| 1 x 256 x 128 x 1 | [8,16,32,64] | 57 ms |
| 1 x 256 x 128 x 1 | [4,8,16,32] | 43 ms |

## NCS1 vs NCS2
OpenVINO is compatible with both the NCS1 AND NCS2 while the NCSDK is only compatible with the NCS1. OpenVINO will be at the forefront of IntelAI solutions and works with many Intel hardware including Intel CPUs, GPUs, FPGA and the Intel Neural Compute Stick.

Some of the workflow in OpenVINO is still the same as NCSDK:

- Compile your model into a binary Intermediate Represenation (IR) format using FP16 data type (using the Model Optimizer instead of the mvNCCompile tool)
- Once you have your binary IR file, you can use it in your Python or C++ application using the Inference Engine API which is equivalent to the NCSDK MVNC API.

# NCS-1 setups


Use your computer to create a partition from free space.
Use this link to see how to resize partition.
https://raspberrypi.stackexchange.com/questions/499/how-can-i-resize-my-root-partition

Remove all the dropout layes, accuracy, loss functions and back prop related code from the saved model and create a new model checkoint files. After that use following command to create openVino graph which can be run on a movidius stick. Make sure to give "input" name to the input conv layer and "output" name to the output layer.

```
mvNCCompile mnist_model.meta -s 12 -in input -on output -o mnist_inference.graph --old-parser
```
- old parser is required because new version does not add bain in the fully connected layer and gives Error : 5.


## NCS-2
NCS-2 require OpenVino toolkit to get the optimized model and run it on NCS-2. NCSDK (Neural Compute Software Developemrnt Kit) works with NCS-1 only.
Supported layers for OpenVino are given in the following links.

https://docs.openvinotoolkit.org/2018_R5/_docs_MO_DG_prepare_model_Supported_Frameworks_Layers.html
https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_prepare_model_Supported_Frameworks_Layers.html

To see how to use NCS-2, [here](https://medium.com/analytics-vidhya/the-battle-to-run-my-custom-network-on-a-movidius-myriad-compute-stick-c7c01fb64126) is a perfect blog post.

## Install OpenVINO

Follow [this](https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_apt.html) instructions to install OpenVINO toolkit on Linux. You can use their docker for simplicity. 

## Configure NCS2
You need to configure NCS2 and set udev rules in your machine so that it can be used. Use [this](https://wiki.seeedstudio.com/ODYSSEY-X86J4105-NCS2/) instructions to configure it.

## Precodure
1] Make your model in tensorflow (or PyTorch) and train it. Freeze your model and get the .pb file. You can also use the .meta file of model but it is not recommended. 

2] Optimize your model using OpenVINO optimizer. You can use mo_tf.py file from your installed OpenVINO directory. Sample function is given below. Visit [this](https://www.intel.com/content/www/us/en/support/articles/000055228/boards-and-kits.html) page for more detailed information. 
```
python3 /opt/intel/openvino_2021.1.110/deployment_tools/model_optimizer/mo_tf.py --input_meta_graph tf_model/mnist_model.meta --input_shape [1,128,128,1]  --data_type FP16
```

3] This optimization will create ".bin", ".mapping" and ".xml" file for the optimized model.Look at the movi2.py file to see how to run this model on NCS-2. Please make sure to source setupvars.sh file from openvino installed path. Remember that you will have to source setupvars in every terminal you open. You can set this in your bashrc and you won't have to do that everytime.
```
source /opt/intel/openvino_2021/bin/setupvars.sh
python3 movi2.py
```

## Run trial setup

Use trial.sh file to check all the working. This bash file will create a tensorflow model, optimize it with OpenVINO and will run the optimized model on NCS-2. In the end it will give you the time taken to run that model for 10 times. 
```
./trial.sh input/image/size number/of/layers
./trial.sh [1,128,128,3] [16,32,64,128]
```
