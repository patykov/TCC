from __future__ import print_function
import os
import argparse
import math
import cntk
import numpy as np

from cntk.utils import *
from cntk.ops import input_variable, cross_entropy_with_softmax, classification_error
from cntk import Trainer, cntk_py
from cntk.learner import momentum_sgd, learning_rate_schedule, momentum_as_time_constant_schedule, UnitType
from _cntk_py import set_computation_network_trace_level
from cntk.device import set_default_device, gpu
from cntk.distributed import data_parallel_distributed_learner, block_momentum_distributed_learner, Communicator


abs_path   = "E:\TCC"
data_path  = os.path.join(abs_path, "DataSets")
model_path = os.path.join(abs_path, "Models", "CNTK")

image_height = 244
image_width  = 244
num_channels = 3  # RGB
num_classes  = 101


# Create network
def create_resnet_network(model_file):
	# create model
	loaded_model = cntk.load_model(model_file)
	z = cntk.combine([loaded_model.outputs[2].owner])

	# Input variables denoting the features and label data
	input_var = input_variable((num_channels, image_height, image_width), 
								dynamic_axes=z.dynamic_axes,
								name='image')
	label_var = input_variable(num_classes, dynamic_axes=z.dynamic_axes,
								name='label')
	
	# loss and metric
	ce = cross_entropy_with_softmax(z, label_var)
	pe = classification_error(z, label_var)

	return {
		'feature': input_var,
		'label': label_var,
		'ce' : ce,
		'pe' : pe,
		'output': z
	}


def train_reader(image_height, image_width, num_channels, num_classes, mean_file, map_file):
	transforms = [cntk.io.ImageDeserializer.scale(width=image_width, height=image_height, channels=num_channels, interpolations='cubic'),
					cntk.io.ImageDeserializer.crop(crop_type='Random'),
					cntk.io.ImageDeserializer.mean(mean_file)]
	return cntk.io.MinibatchSource(cntk.io.ImageDeserializer(map_file, cntk.io.StreamDefs(
		features=cntk.io.StreamDef(field='image', transforms=transforms), # first column in map file is referred to as 'image'
		labels=cntk.io.StreamDef(field='label', shape=num_classes))))     # and second as 'label'.


def test_reader(image_height, image_width, num_channels, num_classes, mean_file, map_file):
	transforms = [cntk.io.ImageDeserializer.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear'),
					cntk.io.ImageDeserializer.crop(crop_type='MultiView10'),
					cntk.io.ImageDeserializer.mean(mean_file)]
	return cntk.io.MinibatchSource(cntk.io.ImageDeserializer(map_file, cntk.io.StreamDefs(
		features=cntk.io.StreamDef(field='image', transforms=transforms), # first column in map file is referred to as 'image'
		labels=cntk.io.StreamDef(field='label', shape=num_classes))),     # and second as 'label'.
		randomize=False)


def create_trainer(network, minibatch_size, epoch_size):
	lr_per_mb = [0.01]*4+[0.001]*2
	momentum_time_constant = -minibatch_size/np.log(0.9)
	l2_reg_weight = 0.0001

	# Set learning parameters
	lr_per_sample = [lr/minibatch_size for lr in lr_per_mb]
	lr_schedule = learning_rate_schedule(lr_per_sample, epoch_size=epoch_size, unit=UnitType.sample)
	mm_schedule = momentum_as_time_constant_schedule(momentum_time_constant)
	learner = momentum_sgd(network['output'].parameters, lr_schedule, mm_schedule, l2_regularization_weight = l2_reg_weight)

	return Trainer(network['output'], network['ce'], network['pe'], learner)


def getFinalLabel(predictedLabels, labelsConfidence):
	maxCount = max(predictedLabels.values())
	top_labels = [label for label in predictedLabels.keys() if predictedLabels[label]==maxCount]
	# Only one label, return it
	if (len(top_labels) == 1):
		confidence = labelsConfidence[top_labels[0]]/maxCount
	# 2 or more labels, need to check confidence
	else:
		topConfidence = dict()
		for label in top_labels:
			topConfidence[label] = labelsConfidence[label]/maxCount
		confidence = max(topConfidence.values())
		top_labels = [label for label in topConfidence.keys() if topConfidence[label]==confidence]
	return top_labels[0], confidence


def eval_and_write(loaded_model, output_file, minibatch_source, action_id, epoch_size):
	# evaluate model and get desired node output
	features_si = minibatch_source['features']
	sample_count = 0
	predictedLabels = dict()
	labelsConfidence = dict()
	with open(output_file, 'a') as results_file:
		results_file.write('{:^15} | '.format(action_id))
		while sample_count < epoch_size:
			mb = minibatch_source.next_minibatch(1)
			output = loaded_model.eval({loaded_model.arguments[0]:mb[features_si]})
			sample_count += mb[features_si].num_samples
			predictions = softmax(np.squeeze(output)).eval()
			top_class = np.argmax(predictions)
			if top_class in predictedLabels.keys():
				predictedLabels[top_class] += 1
				labelsConfidence[top_class] += predictions[top_class] * 100
			else:
				predictedLabels[top_class] = 1
				labelsConfidence[top_class] = predictions[top_class] * 100
			#print('Predicted label: {}, Confidence: {:.2f}%'.format(top_class, predictions[top_class] * 100))
		label, confidence = getFinalLabel(predictedLabels, labelsConfidence)
		results_file.write('{:^15} | {:^15.2f}%\n'.format(label, confidence))


def train_resnet(train_data, mean_data, model_file, epoch_size, output_file, max_epochs=5):

	set_computation_network_trace_level(0)

	minibatch_size = 256
	network = create_resnet_network(model_file)

	progress_printer = ProgressPrinter(tag='Training')
	train_source = train_reader(image_height, image_width, num_channels, num_classes, mean_data, train_data)
	# define mapping from intput streams to network inputs
	input_map = {
		network['feature']: train_source.streams.features,
		network['label']: train_source.streams.labels
	}
	# Create trainer
	trainer = create_trainer(network, minibatch_size, epoch_size)

	# perform model training
	for epoch in range(max_epochs):       # loop over epochs
		sample_count = 0
		while sample_count < epoch_size:  # loop over minibatches in the epoch
			data = train_source.next_minibatch(min(minibatch_size, epoch_size-sample_count), input_map) # fetch minibatch.
			print([a.shape for a in data.keys()])
			print([a.shape for a in data.values()])
			trainer.train_minibatch(data)                                   # update model with it
			sample_count += trainer.previous_minibatch_sample_count         # count samples processed so far
			progress_printer.update_with_trainer(trainer, with_metric=True) # log progress
		progress_printer.epoch_summary(with_metric=True)
		z.save_model(os.path.join(output_path, "Models", "ResNet_34_{}.dnn".format(epoch)))


if __name__=='__main__':
	mean_data=os.path.join(data_path, 'ImageNet1K_mean.xml')
	output_path=os.path.join(abs_path, 'Output')
	output_file=os.path.join(output_path, 'ResNet_34')
	train_data=os.path.join(data_path, 'train_map.txt')
	map_dir    = os.path.join(data_path, "TestMapFiles")

	model_file = os.path.join(model_path, 'ResNet_34.0')
	epoch_size = 888271
	epochs = 6

	set_default_device(gpu(0))

	# Create distributed trainer factory
	print("Start training: epochs_size = {}, epochs = {}".format(epoch_size, epochs))

	# Create output folder
	if not os.path.exists(output_path):
		os.mkdir(output_path)

	train_resnet(train_data,
				mean_data,
				model_file,
				epoch_size,
				output_file,
				max_epochs=epochs)


	# --- Prepare for evaluation ---
	output_file = os.path.join(output_path, "evalOutput.txt")
	trained_model_path = os.path.join(output_path, "Models", "ResNet_34")
	loaded_model  = load_model(trained_model_path)
	trained_model  = combine([loaded_model.outputs[2].owner])
	#Get all test map files
	map_files = sorted(os.listdir(map_dir))
	with open(output_file, 'a') as results_file:
		results_file.write('{:<15} | {:<15} | {:<15}\n'.format('Correct label', 'Predicted label', 'Confidence'))
	for test_file in map_files:
		action_id = test_file.split('_')[-1][:-4]
		file_path = os.path.join(map_dir, test_file)
		minibatch_source = test_reader(image_height, image_width, num_channels, num_classes, mean_data, test_file, False)
		# evaluate model and write out the desired output
		# eval_and_write(trained_model, output_file, minibatch_source, action_id, epoch_size=250)    # 25 frames for that result in 250 inputs for the network

	print("Done. Wrote output to %s" % output_file)