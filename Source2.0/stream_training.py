import argparse
import cntk as C
import os
import model_configuration as mc
import trainer as t
import readers as r

def stream_train(stream_type, map_files, output_dir, log_file, model_name, model_config, distributed=True):
	# Model dimensions
	image_height = 224
	image_width	 = 224
	num_channels = 3
	num_classes	 = 101

	# Learning parameters
	mb_size		  = 32#256
	l2_reg_weight = 0.0001
	
	if stream_type == 'of':
		max_epochs	  = 24 #12
		epoch_size	  = 843030 #1681171
		lr_per_mb	  = [0.01]*50000 + [0.001]*20000 + [0.0001]
		transforms = r.get_crop_transfrom(crop_type='center', crop_size=min(image_height, image_width))
	elif stream_type == 'rgb':
		max_epochs	  = 3
		epoch_size	  = 1776542 
		lr_per_mb	  = [0.01]*14000 + [0.001]
		transforms = r.get_default_transforms(num_channels)
	elif stream_type == 'rgbdiff':
		max_epochs	  = 3
		epoch_size	  = 1728857
		lr_per_mb	  = [0.005]*12000 + [0.0005]*6000 + [0.00005]
		transforms = r.get_crop_transfrom(crop_type='center', crop_size=min(image_height, image_width))
	else:
		raise Exception("Stream type unknown!")
	
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)
		
	# Load network:
	network, base_model_data = mc.fetch_model(model_config)
	print('Network loaded!')
	
	# Set learning parameters
	learner = t.define_learner(network['model'], lr_per_mb, mb_size, distributed=distributed)
	print('Learner set!')
	
	# Get reader
	reader = r.create_mb_source(map_files, base_model_data['num_inputs'], transforms, 
								num_classes, max_epochs, randomize=True)
	input_map = r.get_input_map(network, base_model_data['num_inputs'], reader)
	print('Reader created!')
	
	# Train
	t.train(model_name, network, reader, input_map, 
			learner, mb_size, epoch_size, max_epochs,
			output_dir, log_file, distributed=distributed)
	
	print('Done!')
			

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-datadir', '--datadir', help='Data directory where the dataset is located', 
						required=False, default=None)
	parser.add_argument('-outputdir', '--outputdir', help='Output directory for checkpoints and models', 
						required=False, default=None)
	parser.add_argument('-logdir', '--logdir', help='Log file', 
						required=False, default=None)
	parser.add_argument('-profile', '--profile', help="Turn on profiling", action='store_true', 
						required=False, default=None)
	parser.add_argument('-modelconfig', help="Path to the model configuration file", 
						required=False, default=None)
	parser.add_argument('-modeldir', help="Model directory.",
						required=False, default=None)
	parser.add_argument('-newModelName', help="Name of the new model", 
						required=False, default=None)
	parser.add_argument('-mapdir', help="Directory with mapfiles relative to datadir",
						required=False, default=None)
	parser.add_argument('-streamType', help="Stream type to be trained.",
						required=False, default=None)
	parser.add_argument('-distrib_off', help="Turn off distributed training", action='store_true',
						required=False)
	args = parser.parse_args()
		
	# Paths
	data_dir = args.datadir
	new_model_name = args.newModelName
	output_dir = os.path.join(args.outputdir, new_model_name)
	logFile = args.logdir
	model_config = args.modelconfig
	map_dir = os.path.join(data_dir, args.mapdir)
	stream_type = args.streamType
	distributed = not args.distrib_off

	if new_model_name is None:
		new_model_name   = "VGG16_videoOF_randomCrop"
	if args.outputdir is None:
		output_dir = os.path.join(datadir, new_model_name)
	if args.logdir is None:
		logFile = os.path.join(output_dir, 'logFile.txt')
	if args.modeldir is None:
		model_config = os.path.join(data_dir, args.modelconfig)
	if args.mapdir is None:
		map_dir = os.path.join(data_dir, "OF_mapFiles_half")
	if args.streamType is None:
		stream_type = 'of'
	
	train_mapFiles = [os.path.join(map_dir, f) for f in os.listdir(map_dir) if 'train' in f]
	new_model_file = os.path.join(output_dir, new_model_name+'.model')

	### Training ###
	trained_model = stream_train(stream_type, train_mapFiles, output_dir, logFile, 
								 new_model_name, model_config, distributed)
	trained_model.save(new_model_file)

	print("Stored trained model at %s" % new_model_file)

	C.Communicator.finalize()