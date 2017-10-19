import argparse
from cntk import data_parallel_distributed_learner, load_model, Trainer, UnitType, Communicator
from cntk.debugging import start_profiler, stop_profiler
from cntk.device import gpu, try_set_default_device
from cntk.io import ImageDeserializer, MinibatchSource, StreamDef, StreamDefs
import cntk.io.transforms as xforms
from cntk.learners import momentum_sgd, learning_rate_schedule, momentum_schedule
from cntk.logging import ProgressPrinter
from cntk.logging.graph import find_by_name, get_node_outputs
from cntk.losses import cross_entropy_with_softmax
from cntk.metrics import classification_error
from cntk.ops import input_variable, softmax, splice, combine, reduce_sum, element_divide, reduce_mean, element_times
from cntk.train.training_session import *
import numpy as np
import os
from models import *
import time

# Create a minibatch source.
def create_video_mb_source(map_files, num_channels, image_height, image_width, num_classes):
	transforms = [
		xforms.crop(crop_type='Center', crop_size=224),
		xforms.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear')
	]
	
	map_files = sorted(map_files, key=lambda x: int(x.split('Map_')[1].split('.')[0]))
	
	if len(map_files) != 20:
		raise Exception('There is a problem with the mapFiles selection.')

	# Create multiple image sources
	sources = []
	for i, map_file in enumerate(map_files):
		streams = {"feature"+str(i): StreamDef(field='image', transforms=transforms),
				   "label"+str(i): StreamDef(field='label', shape=num_classes)}
		sources.append(ImageDeserializer(map_file, StreamDefs(**streams)))
	
	return MinibatchSource(sources, max_sweeps=1, randomize=False)

# Create OF network model by adding a pre input
def create_OF_model(z_base, num_classes, num_channels, image_height, image_width):
	# Input variables
	input_var = input_variable((num_channels, image_height, image_width))
	label_var = input_variable(num_classes)

	# Create new input with OF needed transforms
	inputs = []
	for c in range(num_channels):
		inputs.append(input_variable((1, image_height, image_width), name='input_{}'.format(c)))
	flowRange  = 40.0
	imageRange = 255.0
	input_reescaleFlow = [i*(flowRange/imageRange) - flowRange/2 for i in inputs]
	# Splicing components
	input_uv = [splice(input_reescaleFlow[i], input_reescaleFlow[i+1], axis=0) for i in range(0, len(input_reescaleFlow)-1, 2)]
	input_reduceMean = [i-reduce_mean(i, axis=[1, 2]) for i in input_uv]
	new_input = splice(*input_reduceMean, axis=0, name='pre_input')
	z = z_base(new_input)
	# node_outputs = get_node_outputs(z)
	# for out in node_outputs: print("{0} {1}".format(out.name, out.shape))
	
	return z

# Get the video label based on its frames evaluations
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
	return top_labels[0], confidence*100
   
# Evaluate network and writes output to file
def eval_and_write(loaded_model, test_mapFiles, output_file):
	# Model dimensions
	image_height = 224
	image_width	 = 224
	num_classes	 = 101
	
	# Add pre input
	network = loaded_model #create_OF_model(loaded_model, num_classes, 20, image_height, image_width)
	
	# Create test reader
	test_reader = create_video_mb_source(test_mapFiles, 1, image_height, image_width, num_classes)
	
	OF_input_map = {}
	for i in range(20):
		OF_input_map[network.find_by_name("input_"+str(i))] = test_reader.streams["feature"+str(i)]

	with open(test_mapFiles[0], 'r') as file:
		lines = file.readlines()
	max_samples = len(lines)

	correctLabels = [0]*int(max_samples/25)
	for i in range(int(max_samples/25)):
		label = lines[i*25].replace('\n', '').split('\t')[-1]
		correctLabels[i] = int(label)
		
	sample_count = 0.0
	results = ''
	with open(output_file, 'a') as file:
		while sample_count < max_samples:
			mb = test_reader.next_minibatch(25, input_map=OF_input_map)
			predictedLabels = dict((key, 0) for key in range(num_classes))
			labelsConfidence = dict((key, 0) for key in range(num_classes))
			id_correctLabel = int(sample_count/25)
			sample_count += 25
			output = network.eval(mb)
			# import pdb
			# pdb.set_trace()
			predictions = softmax(np.squeeze(output)).eval()
			top_classes = [np.argmax(p) for p in predictions]
			for i, c in enumerate(top_classes):
				predictedLabels[c] += 1 #Melhorar
				labelsConfidence[c] += predictions[i][c]
			label, confidence = getFinalLabel(predictedLabels, labelsConfidence)
			print(correctLabels[id_correctLabel], label, top_classes)
			# results += '{:^15} | {:^15} | {:^15.2f}%\n'.format(correctLabels[id_correctLabel], label, confidence)
			# if sample_count%500 == 0:
				# print('{:.2f}% samples evaluated!'.format((sample_count/max_samples)*100))
				# file.write(results)
				# results = ''
		# file.write(results)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-datadir', '--datadir', help='Data directory where the dataset is located', required=False, default='/hdfs/pnrsy/t-pakova')
	parser.add_argument('-outputdir', '--outputdir', help='Output directory for checkpoints and models', required=False, default=None)
	parser.add_argument('-logdir', '--logdir', help='Log file', required=False, default=None)
	parser.add_argument('-device', '--device', type=int, help="Force to run the script on a specified device", required=False, default=None)
	parser.add_argument('-profile', '--profile', help="Turn on profiling", action='store_true', default=False)
	args = parser.parse_args()
	
	if args.device is not None:
		try_set_default_device(gpu(args.device))
	
	# Paths
	data_dir = args.datadir
	model_name   = "VGG16_videoOF_trainingSession_last"
	if args.logdir is not None:
		logFile = args.logdir
	map_dir = os.path.join(data_dir, "OF_mapFiles-forLaterXforms")
	output_dir = os.path.join(data_dir, model_name)
	test_mapFiles  = sorted([os.path.join(map_dir, f) for f in os.listdir(map_dir) if 'test' in f])
	output_file	   = os.path.join(data_dir, "eval_{}.txt".format(model_name))
	# output_file	   = os.path.join(output_dir, "eval_{}.txt".format(model_name))
	
	# Load Temporal VGG
	trained_model  = load_model('F:/TCC/Models/philly/{}.dnn'.format(model_name))
	# trained_model  = load_model('F:/TCC/Output-VVG16_2_videoOF_part1/Models/{}.dnn'.format(model_name))
	# trained_model  = load_model('F:/TCC/Models/VVG16_videoOF_part2')
	trained_model = combine([trained_model.outputs[0].owner])
	
	if not (os.path.exists(output_file)):
		# raise Exception('The file {} already exist.'.format(output_file))
		with open(output_file, 'w') as results_file:
			results_file.write('{:^15} | {:^15} | {:^15}\n'.format('Correct label', 'Predicted label', 'Confidence'))
	
	# evaluate model and write out the desired output
	eval_and_write(trained_model, test_mapFiles, output_file)
	
	# print("Done. Wrote output to %s" % output_file)
	
	Communicator.finalize()