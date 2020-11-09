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
python3 tf_model.py -u 1 -s $1 -l $2
rm -rf fp16_model/*
python3 /opt/intel/openvino_2021.1.110/deployment_tools/model_optimizer/mo_tf.py --input_meta_graph tf_model/mnist_model.meta --input_shape $1  --data_type FP16
mv mnist_model.* fp16_model/
source /opt/intel/openvino_2021/bin/setupvars.sh
python3 movi2.py
 