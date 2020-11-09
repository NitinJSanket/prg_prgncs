# Run following command first in your terminal
# source /opt/intel/openvino_2021/bin/setupvars.sh
try: from openvino.inference_engine import IECore, IENetwork
except ImportError: print('Make sure you activated setupvars.bat!')
import sys
import numpy as np 
import time

ie = IECore()
device_list = ie.available_devices
print(device_list)
# Load any network from file
model_xml = "fp16_model/mnist_model.xml"
model_bin = "fp16_model/mnist_model.bin"
net = IENetwork(model=model_xml, weights=model_bin)

input_blob = next(iter(net.inputs))
out_blob = next(iter(net.outputs))

print('Input Shape: ' + str(net.inputs[input_blob].shape))
input_ = np.ones(net.inputs[input_blob].shape, dtype=np.float32)
# input_ = np.ones([1,128,128,12], dtype=np.float16)
# Run the model on the device
##############################################################################################
# load model to device
exec_net = ie.load_network(network=net, device_name='MYRIAD')

start = time.time()

for i in range(10):
	res = exec_net.infer(inputs={input_blob: input_})
	res = res[out_blob]
	print(i,"----",res.shape)
print("Total time taken -- ",time.time() - start)