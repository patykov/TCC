import argparse
import cntk as C
import json
import os
# import sys
# sys.path.append(os.path.abspath(os.path.join(sys.path[0], "..","..", "src")))
import stream_learning
import readers
import model_configuration as mc
# import evaluation
# from dataset import extractData, mapData

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-datadir', '--datadir', help='Directory for input data', required=False)
	parser.add_argument('-outputdir', '--outputdir', help='Directory where results will be saved', 
						required=False, default=None)
	parser.add_argument('-logdir', '--logdir', help='Log file', required=False, default=None)
	parser.add_argument('-device', '--device', help='Device id', required=False, default=None)
	parser.add_argument('-profile', '--profile', help="Turn on profiling", action='store_true',
						default=False)
	args = parser.parse_args()
	
	### Main characteristics ###
	dataset_name = 'UCF-101'
	model_name = 'vgg16'

	### Paths ###
	data_dir = args.datadir#os.path.join(args.datadir, dataset_name)
	output_dir = os.path.join(args.outputdir, dataset_name+'_'+model_name)
	base_model_path = "F:/TCC/Models/philly/VGG16_videoRGB_xforms" #"F:/TCC/Models/VGG16_ImageNet_CNTK.model"
	map_dir = os.path.join(data_dir, 'UCF-101_ofMapFiles_split1')
	train_map_files = [os.path.join(map_dir, f) for f in os.listdir(map_dir) if 'train' in f]
	test_map_files = [os.path.join(map_dir, f) for f in os.listdir(map_dir) if 'test' in f]
	device_id = (0 if args.device is None else args.device)
	
	# Model dimensions
	image_height = 224
	image_width	 = 224
	num_channels = 3
	num_inputs	 = 5
	num_classes	 = 101

	# Learning parameters
	max_epochs = 12 
	epoch_size = 1681171 
	mb_size	   = 256
	lr_per_mb  = [0.01]*8 + [0.001]*3 + [0.0001]
	
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)
	
	### Try to set GPU ###
	C.device.try_set_default_device(C.device.gpu(device_id))
	
	### Create Network ###
	# network = mc.get_OF_from_scratch(model_name, dropout=0.5, num_inputs=20, num_classes=101, 
						# num_channels=1, image_height=224, image_width=224)
	# network = mc.get_RGB_fine_tuning(base_model_path, feature_node_name='data', last_node_name='drop7',
						# num_classes=101, num_channels=3, image_height=224, image_width=224)
	network = mc.get_RGBdiff_fine_tuning(base_model_path, feature_node_name='conv1_1', last_node_name='fc101',
							num_inputs=5, num_classes=101, num_channels=3, image_height=224, image_width=224)
	mc.print_layers(network['model'])
	raise Exception('End of test')
	
	### Create Reader ###
	transforms = readers.get_default_transforms(image_height, image_width, num_classes, is_training=True)
	train_reader = readers.create_mb_source(train_map_files, transforms, num_classes, max_epochs, randomize=True)
	
	### Training ###
	z = stream_learning.train(model_name=model_name, 
							network=network, 
							train_reader=train_reader, 
							lr_per_mb=lr_per_mb, 
							max_epochs=max_epochs, 
							mb_size=mb_size, 
							epoch_size=epoch_size, 
							output_dir=output_dir)
	
	### Evaluation ###