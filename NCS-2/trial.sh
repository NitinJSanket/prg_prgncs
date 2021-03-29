if [ "$1" = "" ]; then
	echo Input image size not given
	exit
fi

if [ "$2" = "" ]; then
	echo Layers not defined
	exit
fi
SIZE=$1
LAYERS=$2
rm -rf tf_model/*
python3 tf_model.py -s $1 -l $2
cd tf_model
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_meta_graph mnist_model.meta --input_shape $1  --data_type FP16
cd ..
source /opt/intel/openvino/bin/setupvars.sh
python3 movi2.py
 