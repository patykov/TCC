import os
import numpy as np
from cntk import cross_entropy_with_softmax, classification_error
from cntk.device import set_default_device, gpu
from cntk.io import MinibatchSource, ImageDeserializer, StreamDef, StreamDefs
import cntk.io.transforms as xforms
from cntk import Trainer
from cntk.learners import momentum_sgd, learning_rate_schedule, momentum_schedule, UnitType
from cntk.debugging import set_computation_network_trace_level
from cntk import load_model, placeholder
from cntk.layers import Dense
from cntk.logging.graph import find_by_name, depth_first_search, get_node_outputs
from cntk.logging import ProgressPrinter, log_number_of_parameters
from cntk.ops import input, combine, softmax
from cntk.ops.functions import CloneMethod

# Paths
base_folder = "E:\TCC"
data_dir    = os.path.join(base_folder, "Datasets")
models_dir  = os.path.join(base_folder, "Models")

# Model dimensions
image_height = 224
image_width	 = 224
num_channels = 3
num_classes	 = 101


# Define the reader for both training and evaluation action.
def create_reader(map_file, mean_file):
	transforms = [
		xforms.color(brightness_radius=0.2, contrast_radius=0.2, saturation_radius=0.4),
		xforms.crop(crop_type='randomarea', crop_size=224, jitter_type='uniratio'),
		xforms.scale(width=224, height=224, channels=num_channels, interpolations='linear'),
		xforms.mean(mean_file)
	]
	return MinibatchSource(ImageDeserializer(map_file, StreamDefs(
		features = StreamDef(field='image', transforms=transforms),
		labels	 = StreamDef(field='label', shape=num_classes))),
		randomize=False)

def print_all_node_names(loaded_model):
	node_list = depth_first_search(loaded_model, lambda x: x.is_output)
	print("printing node information in the format")
	print("node name (tensor shape)")
	for node in node_list:
		print(node.name, node.shape)
		
def find_arg_by_name(name, expression):
	vars = [i for i in expression.arguments if i.name == name]
	assert len(vars) == 1
	return vars[0]
	
def create_model(base_model, last_hidden_node_name, num_classes, input_features, input_label):
	last_node = find_by_name(base_model, last_hidden_node_name)
	
	# Clone the desired layers
	cloned_layers = combine([last_node.owner]).clone(
		CloneMethod.clone, {input_features: placeholder(name='features')})
	
	# Add new dense layer for class prediction
	cloned_out = cloned_layers(input_features)
	z          = Dense(num_classes, activation=None, name='fc101') (cloned_out)
	return z
	
# Train and evaluate the network.
def train_model(train_reader, network_path, output_dir, log_file):
	set_computation_network_trace_level(0)

	# Learning parameters
	max_epochs = 6
	epoch_size = 888271
	minibatch_size = 256 
	lr_per_mb = [0.01]*4 + [0.001] # Following article
	momentum_per_mb = 0.9
	l2_reg_weight = 0.0001

	# Input variables
	image_input = input((num_channels, image_height, image_width))
	label_input = input(num_classes)
	
	# create model
	base_model = load_model(network_path)
	z = create_model(base_model, 'pool5', num_classes, image_input, label_input)
	# node_outputs = get_node_outputs(z)
	# for out in node_outputs:
		# print("{0} {1}".format(out.name, out.shape))
		
	# loss and metric
	ce = cross_entropy_with_softmax(z, label_input)
	pe = classification_error(z, label_input)

	# Set learning parameters
	lr_per_sample = [lr/minibatch_size for lr in lr_per_mb]
	lr_schedule = learning_rate_schedule(lr_per_sample, epoch_size=epoch_size, unit=UnitType.sample)
	mm_schedule = momentum_schedule(momentum_per_mb)

	# progress writers
	progress_writers = [ProgressPrinter(tag='Training', num_epochs=max_epochs, 
						log_to_file=log_file, freq=10)]

	# trainer object
	learner = momentum_sgd(z.parameters, lr_schedule, mm_schedule,
							 l2_regularization_weight = l2_reg_weight)
	trainer = Trainer(z, (ce, pe), learner, progress_writers)

	with open(log_file, 'a') as file:
		file.write('\nlr_per_mb = {}\n'.format(lr_per_mb))
		file.write('Minibatch_size = {}\n'.format(minibatch_size))
	
	# define mapping from reader streams to network inputs
	input_map = {
		image_input: train_reader.streams.features,
		label_input: train_reader.streams.labels
	}

	log_number_of_parameters(z) ; print()

	features_si = train_reader['features']
	labels_si = train_reader['labels']
	for epoch in range(max_epochs):		    # loop over epochs
		sample_count = 0
		while sample_count < epoch_size:	# loop over minibatches in the epoch
			data = train_reader.next_minibatch(min(minibatch_size, epoch_size-sample_count), input_map=input_map)
			trainer.train_minibatch(data)							# update model with it
			sample_count += trainer.previous_minibatch_sample_count	# count samples processed so far

		trainer.summarize_training_progress()
		z.save(os.path.join(output_dir, 'Models', 'resnet34_{}.model'.format(epoch)))

	return z

if __name__=='__main__':
	set_default_device(gpu(0))

	network_path   = os.path.join(models_dir, 'ResNet_34.model') # ImageNet with 1000 output
	# train_map_file = os.path.join(data_dir, 'train_map01_RGB.txt')
	train_map_file = os.path.join(data_dir, 'compareTrainMap.txt')
	mean_file_path = os.path.join(data_dir, 'ImageNet1K_mean.xml')
	output_dir	   = os.path.join(base_folder, 'Output-resnet34-compare')
	log_file       = os.path.join(output_dir, 'resnet34.txt')
	new_model_file = os.path.join(output_dir, 'resnet34_256')
	
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)

	train_reader = create_reader(train_map_file, mean_file_path)
	trained_model = train_model(train_reader, network_path, output_dir, log_file)
	trained_model.save(new_model_file)   