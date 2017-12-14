import argparse
import cntk as C
from cntk import load_model, UnitType, Communicator
from cntk.learners import momentum_sgd, learning_rate_schedule, momentum_schedule
from cntk.logging import ProgressPrinter
from cntk.logging.graph import find_by_name, get_node_outputs
from cntk.losses import cross_entropy_with_softmax
from cntk.metrics import classification_error
from cntk.ops import input_variable, softmax, splice, combine, reduce_mean
from cntk.train.training_session import *
import numpy as np
import os
from models import *


# Create a minibatch source.
def create_video_mb_source(map_files, num_channels, image_height, image_width, num_classes, max_epochs):
	transforms = [
		C.io.transforms.crop(crop_type='Center', crop_size=224),
		C.io.transforms.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear')
	]
	
	map_files = sorted(map_files, key=lambda x: int(x.split('Map_')[1].split('.')[0]))

	print(map_files) 
	if len(map_files) != 20:
		raise Exception('There is a problem with the mapFiles selection.')

	# Create multiple image sources
	sources = []
	for i, map_file in enumerate(map_files):
		streams = {"feature"+str(i): C.io.StreamDef(field='image', transforms=transforms),
				   "label"+str(i): C.io.StreamDef(field='label', shape=num_classes)}
		sources.append(C.io.ImageDeserializer(map_file, C.io.StreamDefs(**streams)))

	return C.io.MinibatchSource(sources, max_sweeps=max_epochs, randomize=True)

# Create OF network model by adding a pre input
def create_OF_model(num_classes, num_channels, image_height, image_width):
	input_var = C.input_variable((num_channels, image_height, image_width))
	
	# Label Variable
	label_var = C.input_variable(num_classes)
	
	# Input Variable
	inputs = []
	for c in range(num_channels):
		inputs.append(input_variable((1, image_height, image_width), name='input_{}'.format(c)))
	flowRange  = 40.0
	imageRange = 255.0
	input_reescaleFlow = [i*(flowRange/imageRange) - flowRange/2 for i in inputs]
	input_reduceMean = [(i-reduce_mean(i, axis=[1,2])) for i in input_reescaleFlow]
	new_input = splice(*(i for i in input_reduceMean), axis=0, name='pre_input')
	
	# Clone the desired layers 
	z = create_vgg16(new_input, num_classes, 0.9)

	# Loss and metric
	ce = C.losses.cross_entropy_with_softmax(z, label_var)
	pe = C.metrics.classification_error(z, label_var)
	
	features = {}
	for i, input in enumerate(inputs):
		features['feature'+str(i)] = input
	
	return dict({
		'model': z,
		'ce' : ce,
		'pe' : pe,
		'label': label_var}, 
		**features)

	
# Trains a transfer learning model
def train_model(train_mapFiles, output_dir, log_file, model_name, profiling=False):
	# Model dimensions
	image_height = 224
	image_width	 = 224
	stack_length = 10
	num_classes	 = 101

	# Learning parameters
	max_epochs	  = 24 #12
	epoch_size	  = 843030 #1681171
	mb_size		  = 256
	lr_per_mb	  = [0.01]*50000 + [0.001]*20000 + [0.0001]
	l2_reg_weight = 0.0001
	
	# Create network:
	network = create_OF_model(num_classes, 2*stack_length, image_height, image_width)

	# Set learning parameters
	# lr_schedule = C.learners.learning_rate_schedule(lr_per_mb, unit=C.UnitType.minibatch)
	# mm_schedule = C.learners.momentum_schedule(0.9)

	# Printer
	# progress_printer = ProgressPrinter(freq=10, tag='Training', log_to_file=log_file, 
									  # num_epochs=max_epochs, gen_heartbeat=True,
									  # rank=Communicator.rank())

	# Trainer object
	# local_learner = momentum_sgd(network['model'].parameters, lr_schedule, mm_schedule, 
								# unit_gain=False, l2_regularization_weight = l2_reg_weight)
	# learner = C.data_parallel_distributed_learner(local_learner, 
												# num_quantization_bits = 32, 
												# distributed_after = 0)
	# trainer = C.Trainer(network['model'], (network['ce'], network['pe']), learner, progress_printer)
	
	# Create train reader:
	# train_reader = create_video_mb_source(train_mapFiles, 1, image_height, image_width, num_classes, max_epochs)
	# print('Reader created!')
	# define mapping from intput streams to network inputs
	input_map = {network['label']: train_reader.streams.label1}
	for i in range(20):
		input_map[network['feature'+str(i)]] = train_reader.streams["feature"+str(i)]

	# if profiling:
		# C.debugging.start_profiler(dir=output_dir, sync_gpu=True)

	# training_session(
		# trainer=trainer, mb_source=train_reader,
		# model_inputs_to_streams=input_map,
		# mb_size=mb_size,
		# progress_frequency=epoch_size,
		# checkpoint_config=CheckpointConfig(filename=os.path.join(output_dir, model_name), restore=True,
											# frequency=int(epoch_size/2))
	# ).train()
	
	# trainer.save_checkpoint(os.path.join(output_dir, "{}_last.dnn".format(model_name)))
	
	# if profiling:
		# C.debugging.stop_profiler()
	
	# return network['model']
			

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-datadir', '--datadir', help='Data directory where the dataset is located', required=False)
	parser.add_argument('-outputdir', '--outputdir', help='Output directory for checkpoints and models', required=False, default=None)
	parser.add_argument('-logdir', '--logdir', help='Log file', required=False, default=None)
	parser.add_argument('-profile', '--profile', help="Turn on profiling", action='store_true', default=False)
	args = parser.parse_args()
		
	# Paths
	data_dir = args.datadir
	# For training
	newModelName   = "VGG16_videoOF_"
	logFile = args.logdir
	map_dir = os.path.join(data_dir, "OF_mapFiles_half")
	output_dir = os.path.join(args.outputdir, newModelName)
	train_mapFiles = [os.path.join(map_dir, f) for f in os.listdir(map_dir) if 'train' in f]
	new_model_file = os.path.join(output_dir, newModelName+'.dnn')
	
	### Training ###
	# if not os.path.exists(output_dir):
		# os.mkdir(output_dir)
	
	trained_model = train_model(train_mapFiles, output_dir, logFile, newModelName)
	# trained_model.save(new_model_file)

	print("Stored trained model at %s" % new_model_file)

	
	Communicator.finalize()