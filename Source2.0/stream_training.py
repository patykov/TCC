import argparse
import cntk as C
import os
import model_configuration as mc
import trainer as t
import readers as r
import create_model as cm

def stream_train(stream_type, map_files, output_dir, log_file, model_name, base_model, distributed=True):
	# Model dimensions
	image_height = 224
	image_width	 = 224
	num_channels = 3
	num_classes	 = 101

	# Learning parameters
	mb_size		  = 64
	l2_reg_weight = 0.0001

	if stream_type == 'of':
		max_epochs	  = 24 #12
		epoch_size	  = 843030 #1681171
		lr_per_mb	  = [0.01]*50000 + [0.001]*20000 + [0.0001]
		num_inputs    = 20
		network = cm.get_OF_from_scratch(base_model)
	elif stream_type == 'of_full':
		max_epochs	  = 12
		epoch_size	  = 1681171
		lr_per_mb	  = [0.01]*50000 + [0.001]*20000 + [0.0001]
		num_inputs    = 20
		network = cm.get_OF_from_scratch(base_model)
	elif stream_type == 'rgb':
		max_epochs	  = 3
		epoch_size	  = 1776542 
		lr_per_mb	  = [0.01]*14000 + [0.001]
		num_inputs = 1
		network = cm.get_RGB_fine_tuning(base_model)
	elif stream_type == 'rgb_clone':
		max_epochs	  = 3
		epoch_size	  = 1776542 
		lr_per_mb	  = [0.01]*14000 + [0.001]
		num_inputs = 1
		network = cm.get_RGB_fine_tuning(base_model, False)
	elif stream_type == 'rgbdiff':
		max_epochs	  = 3
		epoch_size	  = 1728857
		lr_per_mb	  = [0.005]*12000 + [0.0005]*6000 + [0.00005]
		num_inputs    = 5
		network = cm.get_RGBdiff_fine_tuning(base_model)
	elif stream_type == 'rgbdiff_normal':
		max_epochs	  = 3
		epoch_size	  = 1728857
		lr_per_mb	  = [0.005]*12000 + [0.0005]*6000 + [0.00005]
		num_inputs    = 5
		network = cm.get_old_RGBdiff_fine_tuning(base_model)
	else:
		raise Exception("Stream type unknown!")
		
	# Load network:
	# network, base_model_data = mc.fetch_model(model_config)
	print('Network loaded!')
	
	# Set learning parameters
	learner = t.define_learner(network['model'], lr_per_mb, mb_size, distributed=distributed)
	print('Learner set!')
	
	# Get reader
	transforms = r.get_default_transforms(num_channels)
	reader = r.create_mb_source(map_files, num_inputs, transforms, 
								num_classes, max_epochs, randomize=True)
	input_map = r.get_input_map(network, num_inputs, reader)
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
	parser.add_argument('-base_model', help="Base model path or name", 
						required=False, default=None)
	parser.add_argument('-new_model_name', help="Name of the new model", 
						required=True)
	parser.add_argument('-map_dir', help="Directory with mapfiles relative to datadir",
						required=True)
	parser.add_argument('-stream_type', help="Stream type to be trained.",
						required=True)
	parser.add_argument('-distrib_off', help="Turn off distributed training", action='store_true',
						required=False)
	args = parser.parse_args()
		
	# Paths
	data_dir = args.datadir
	new_model_name = args.new_model_name
	output_dir = args.outputdir
	logFile = args.logdir
	base_model = args.base_model
	map_dir = os.path.join(data_dir, args.map_dir)
	stream_type = args.stream_type
	distributed = not args.distrib_off


	if args.logdir is None:
		output_dir = os.path.join(output_dir, new_model_name)
		if not os.path.exists(output_dir):
			os.mkdir(output_dir)
		logFile = os.path.join(output_dir, 'logFile.txt')
	
	train_mapFiles = [os.path.join(map_dir, f) for f in os.listdir(map_dir) if 'train' in f]
	new_model_file = os.path.join(output_dir, new_model_name+'.model')

	### Training ###
	trained_model = stream_train(stream_type, train_mapFiles, output_dir, logFile, 
								 new_model_name, base_model, distributed)

	print("Stored trained model at %s" % new_model_file)

	C.Communicator.finalize()