# Run following command first in your terminal
# source /opt/intel/openvino/bin/setupvars.sh
try: from openvino.inference_engine import IECore, IENetwork
except ImportError: print('Make sure you activated setupvars.sh!')
import sys
import numpy as np 
import time
import argparse
from tqdm import tqdm

def SpeedTest(Args):
    ie = IECore()
    device_list = ie.available_devices
    print(device_list)
    # Load any network from file
    model_xml = Args.BasePath + 'mnist_model.xml'
    model_bin = Args.BasePath + 'mnist_model.bin'
    net = IENetwork(model=model_xml, weights=model_bin)

    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))

    print('Input Shape: ' + str(net.inputs[input_blob].shape))
    input_ = np.ones(net.inputs[input_blob].shape, dtype=np.float32)
    # input('q')
    # Run the model on the device
    ##############################################################################################
    # load model to device
    exec_net = ie.load_network(network=net, device_name='MYRIAD')


    for i in tqdm(range(Args.NumWarmUp + Args.NumRuns)):
            if(i == Args.NumWarmUp):
                start = time.time()
            res = exec_net.infer(inputs={input_blob: input_})
            res = res[out_blob]
            # print(i,"----",res.shape)
    print("FPS: ", Args.NumRuns/(time.time() - start))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-r', '--NumRuns', type=int, default=25, help='Number of runs to average time from')
  parser.add_argument('-w', '--NumWarmUp', type=int, default=10, help='Number of warmup runs')
  parser.add_argument('-b', '--BasePath', default='tf_model/', help='Base Path where to load model from')
  
  Args = parser.parse_args()
  SpeedTest(Args)
  
  
