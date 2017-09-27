import argparse
from cntk import data_parallel_distributed_learner, load_model, Trainer, UnitType, Communicator
from cntk.debugging import start_profiler, stop_profiler, enable_profiler
from cntk.device import gpu, try_set_default_device
from cntk.io import ImageDeserializer, MinibatchSource, StreamDef, StreamDefs
import cntk.io.transforms as xforms
from cntk.learners import momentum_sgd, learning_rate_schedule, learning_parameter_schedule, momentum_schedule
from cntk.logging import ProgressPrinter, log_number_of_parameters
from cntk.logging.graph import find_by_name, get_node_outputs
from cntk.losses import cross_entropy_with_softmax
from cntk.metrics import classification_error
from cntk.ops import input_variable, softmax, sequence, reduce_mean, splice
from cntk.train.training_session import *
import itertools
import numpy as np
import os
import re
from PIL import Image
from models import *
from multiprocessing import Process
import time

class TrainReader(object):

	def __init__(self, mapFiles, maxPartitions, network):
		self.mapFiles	   = mapFiles
		self.maxPartitions = maxPartitions
		self.trainModule   = 0
		self.network	   = network
	
	def hasNext(self):
		if (self.trainModule < self.maxPartitions):
			return True
		return False
	
	def getReader(self, image_height, image_width, num_classes):
		if not self.hasNext():
			raise Exception('There is no more training data.')
		
		print('Training {}o particion.'.format(self.trainModule+1))
		trainFiles = [f for f in self.mapFiles if 'part{}'.format(self.trainModule) in f]
		
		# Create train reader
		p = Process(target=creatingReaderPrinter, args=())
		p.start()
		train_reader = create_video_mb_source(trainFiles, 1, image_height, image_width, num_classes)
		p.terminate()

		# define mapping from intput streams to network inputs
		input_map = {
			self.network['label']: train_reader.streams.labels1,
			self.network['feature1']: train_reader.streams.features1,
			self.network['feature2']: train_reader.streams.features2,
			self.network['feature3']: train_reader.streams.features3,
			self.network['feature4']: train_reader.streams.features4,
			self.network['feature5']: train_reader.streams.features5,
			self.network['feature6']: train_reader.streams.features6,
			self.network['feature7']: train_reader.streams.features7,
			self.network['feature8']: train_reader.streams.features8,
			self.network['feature9']: train_reader.streams.features9,
			self.network['feature10']: train_reader.streams.features10,
			self.network['feature11']: train_reader.streams.features11,
			self.network['feature12']: train_reader.streams.features12,
			self.network['feature13']: train_reader.streams.features13,
			self.network['feature14']: train_reader.streams.features14,
			self.network['feature15']: train_reader.streams.features15,
			self.network['feature16']: train_reader.streams.features16,
			self.network['feature17']: train_reader.streams.features17,
			self.network['feature18']: train_reader.streams.features18,
			self.network['feature19']: train_reader.streams.features19,
			self.network['feature20']: train_reader.streams.features20
		}
		self.trainModule +=1
		
		return train_reader, input_map

def creatingReaderPrinter():
	counter = 0
	while True:
		print('Creating the reader for {:.2f} min'.format(counter/60.0))
		time.sleep(10)
		counter += 10

# Create a minibatch source.
def create_video_mb_source(map_files, num_channels, image_height, image_width, num_classes):
	transforms = [xforms.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear')]

	map_files = sorted(map_files, key=lambda x: int(x.split('Map_')[1].split('part')[0]))
	print(map_files)
	# Create multiple image sources
	sources = [
		ImageDeserializer(map_files[0], StreamDefs(
				features1 = StreamDef(field='image', transforms=transforms),
				labels1	  = StreamDef(field='label', shape=num_classes))),	 
		ImageDeserializer(map_files[1], StreamDefs(
				features2 = StreamDef(field='image', transforms=transforms),
				labels2	  = StreamDef(field='label', shape=num_classes))),
		ImageDeserializer(map_files[2], StreamDefs(
				features3 = StreamDef(field='image', transforms=transforms),
				labels3	  = StreamDef(field='label', shape=num_classes))),
		ImageDeserializer(map_files[3], StreamDefs(
				features4 = StreamDef(field='image', transforms=transforms),
				labels4	  = StreamDef(field='label', shape=num_classes))),
		ImageDeserializer(map_files[4], StreamDefs(
				features5 = StreamDef(field='image', transforms=transforms),
				labels5	  = StreamDef(field='label', shape=num_classes))),
		ImageDeserializer(map_files[5], StreamDefs(
				features6 = StreamDef(field='image', transforms=transforms),
				labels6	  = StreamDef(field='label', shape=num_classes))),
		ImageDeserializer(map_files[6], StreamDefs(
				features7 = StreamDef(field='image', transforms=transforms),
				labels7	  = StreamDef(field='label', shape=num_classes))),
		ImageDeserializer(map_files[7], StreamDefs(
				features8 = StreamDef(field='image', transforms=transforms),
				labels8	  = StreamDef(field='label', shape=num_classes))),
		ImageDeserializer(map_files[8], StreamDefs(
				features9 = StreamDef(field='image', transforms=transforms),
				labels9	  = StreamDef(field='label', shape=num_classes))),
		ImageDeserializer(map_files[9], StreamDefs(
				features10 = StreamDef(field='image', transforms=transforms),
				labels10   = StreamDef(field='label', shape=num_classes))),
		ImageDeserializer(map_files[10], StreamDefs(
				features11 = StreamDef(field='image', transforms=transforms),
				labels11   = StreamDef(field='label', shape=num_classes))),
		ImageDeserializer(map_files[11], StreamDefs(
				features12 = StreamDef(field='image', transforms=transforms),
				labels12   = StreamDef(field='label', shape=num_classes))),
		ImageDeserializer(map_files[12], StreamDefs(
				features13 = StreamDef(field='image', transforms=transforms),
				labels13   = StreamDef(field='label', shape=num_classes))),
		ImageDeserializer(map_files[13], StreamDefs(
				features14 = StreamDef(field='image', transforms=transforms),
				labels14   = StreamDef(field='label', shape=num_classes))),
		ImageDeserializer(map_files[14], StreamDefs(
				features15 = StreamDef(field='image', transforms=transforms),
				labels15   = StreamDef(field='label', shape=num_classes))),
		ImageDeserializer(map_files[15], StreamDefs(
				features16 = StreamDef(field='image', transforms=transforms),
				labels16   = StreamDef(field='label', shape=num_classes))),
		ImageDeserializer(map_files[16], StreamDefs(
				features17 = StreamDef(field='image', transforms=transforms),
				labels17   = StreamDef(field='label', shape=num_classes))),
		ImageDeserializer(map_files[17], StreamDefs(
				features18 = StreamDef(field='image', transforms=transforms),
				labels18   = StreamDef(field='label', shape=num_classes))),
		ImageDeserializer(map_files[18], StreamDefs(
				features19 = StreamDef(field='image', transforms=transforms),
				labels19   = StreamDef(field='label', shape=num_classes))),
		ImageDeserializer(map_files[19], StreamDefs(
				features20 = StreamDef(field='image', transforms=transforms),
				labels20   = StreamDef(field='label', shape=num_classes)))
	]

	return MinibatchSource(sources, max_sweeps=1, randomize=False)

# Create OF network model by adding a pre input
def create_OF_model(num_classes, num_channels, image_height, image_width):
	# Input variables
	input_var = input_variable((num_channels, image_height, image_width))
	label_var = input_variable(num_classes)

	# create base model
	z_base = create_vgg16(input_var, num_classes)
	# node_outputs = get_node_outputs(z_base)
	# for out in node_outputs: print("{0} {1}".format(out.name, out.shape))

	# Create new input with OF needed transforms
	inputs = []
	for c in range(num_channels):
		inputs.append(input_variable((1, image_height, image_width), name='input_{}'.format(c)))
	flowRange  = 40.0
	imageRange = 255.0
	input_reescaleFlow = [i*(flowRange/imageRange) - flowRange/2 for i in inputs]
	input_reduceMean = []
	for j, i in enumerate(input_reescaleFlow):
		input_reduceMean.append(i-input_reescaleFlow[(j+1) if (j%2==0) else (j-1)])
	new_input = splice(*(i for i in input_reduceMean), axis=0)
	
	z = z_base(new_input)
	# node_outputs = get_node_outputs(z)
	# for out in node_outputs: print("{0} {1}".format(out.name, out.shape))
	
	# Loss and metric
	ce = cross_entropy_with_softmax(z, label_var)
	pe = classification_error(z, label_var)
	
	return {
		'model': z,
		'ce' : ce,
		'pe' : pe,
		'label': label_var,
		'feature1': inputs[0],
		'feature2': inputs[1],
		'feature3': inputs[2],
		'feature4': inputs[3],
		'feature5': inputs[4],
		'feature6': inputs[5],
		'feature7': inputs[6],
		'feature8': inputs[7],
		'feature9': inputs[8],
		'feature10': inputs[9],
		'feature11': inputs[10],
		'feature12': inputs[11],
		'feature13': inputs[12],
		'feature14': inputs[13],
		'feature15': inputs[14],
		'feature16': inputs[15],
		'feature17': inputs[16],
		'feature18': inputs[17],
		'feature19': inputs[18],
		'feature20': inputs[19]
	}

def getLastmodel(output_dir):
	trained_models = [f for f in os.listdir(output_dir) if '.dnn' in f]
	if len(trained_models) > 0:
		trained_models = sorted(trained_models, key=lambda x: int(x.split('_')[1].split('.dnn')[0]))
		print(trained_models[-1])
		return trained_models[-1]
	return None
	
# Trains a transfer learning model
def train_model(train_mapFiles, output_dir, log_file, model_name, profiling=True):
	# Model dimensions
	image_height = 224
	image_width	 = 224
	stack_length = 10
	num_classes	 = 101

	# Learning parameters
	max_epochs	  = 1 #2147 # 9537 training videos on total
	epoch_size	  = 20475938 # 9537
	mb_size		  = 256
	lr_per_mb     = [0.01]*1341 + [0.001]*538 + [0.0001]
	l2_reg_weight = 0.0001
	
	# Create network:
	network = create_OF_model(num_classes, 2*stack_length, image_height, image_width)

	# Set learning parameters
	lr_per_sample = [lr/256 for lr in lr_per_mb]
	lr_schedule = learning_rate_schedule(lr_per_sample, epoch_size=epoch_size, 
											unit=UnitType.sample)
	mm_schedule = momentum_schedule(0.9, minibatch_size=mb_size)

	# Printer
	progress_printer = ProgressPrinter(freq=10, tag='Training', log_to_file=log_file, 
									  num_epochs=max_epochs, gen_heartbeat=True,
									  rank=Communicator.rank())

	# Trainer object
	local_learner = momentum_sgd(network['model'].parameters, lr_schedule, mm_schedule, 
								unit_gain=False, l2_regularization_weight = l2_reg_weight)
	learner = data_parallel_distributed_learner(local_learner, 
												num_quantization_bits = 32, 
												distributed_after = 0)
	trainer = Trainer(network['model'], (network['ce'], network['pe']), learner, progress_printer)
	
	# Restore training and get last sample_count and last_epoch
	last_trained_model = getLastmodel(output_dir)
	if last_trained_model is not None:
		trainer.restore_from_checkpoint(last_trained_model)
	z = trainer.model
	
	# Create reader:							  
	reader = TrainReader(train_mapFiles, 50, network)
	train_reader, input_map = reader.getReader(image_height, image_width, num_classes)

	sample_count = trainer.total_number_of_samples_seen
	last_epoch = int(sample_count/epoch_size)
	print('Total number of samples seen: {} | Last epoch: {}\n'.format(sample_count, last_epoch))

	if profiling:
		start_profiler(dir=output_dir, sync_gpu=True)
		
	for epoch in range(last_epoch, max_epochs):  # loop over epochs
		while sample_count < epoch_size:         # loop over minibatches in the epoch
			data = train_reader.next_minibatch(mb_size, input_map=input_map,
					num_data_partitions=Communicator.num_workers(), partition_index=Communicator.rank()) # fetch minibatch.
			trainer.train_minibatch(data)									# update model with it
			sample_count += trainer.previous_minibatch_sample_count			# count samples processed so far
			print('{} {} {:.4f}'.format(trainer.previous_minibatch_sample_count, sample_count, 
									(float(sample_count)/epoch_size)*100))
			if((trainer.previous_minibatch_sample_count < mb_size) and reader.hasNext()):
				trainer.save_checkpoint(os.path.join(output_dir, "{}_{}.dnn".format(model_name, 
																					reader.trainModule)))
				del train_reader
				train_reader, input_map = reader.getReader(image_height, image_width, num_classes)

		trainer.summarize_training_progress()

	if profiling:
		stop_profiler()
	
	return network['model']
			
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
	
	# Create test reader
	test_reader = create_video_mb_source(test_mapFiles, 1, image_height, image_width, num_classes)
	
	# define mapping from intput streams to network inputs
	input_map = {
		network['label']: train_reader.streams.labels1,
		network['feature1']: train_reader.streams.features1,
		network['feature2']: train_reader.streams.features2,
		network['feature3']: train_reader.streams.features3,
		network['feature4']: train_reader.streams.features4,
		network['feature5']: train_reader.streams.features5,
		network['feature6']: train_reader.streams.features6,
		network['feature7']: train_reader.streams.features7,
		network['feature8']: train_reader.streams.features8,
		network['feature9']: train_reader.streams.features9,
		network['feature10']: train_reader.streams.features10,
		network['feature11']: train_reader.streams.features11,
		network['feature12']: train_reader.streams.features12,
		network['feature13']: train_reader.streams.features13,
		network['feature14']: train_reader.streams.features14,
		network['feature15']: train_reader.streams.features15,
		network['feature16']: train_reader.streams.features16,
		network['feature17']: train_reader.streams.features17,
		network['feature18']: train_reader.streams.features18,
		network['feature19']: train_reader.streams.features19,
		network['feature20']: train_reader.streams.features20
	}
	with open(test_mapFiles[0], 'r') as file:
		max_samples = (len(file.readlines())-1)
		
	sample_count = 0
	results = ''
	with open(output_file, 'a') as file:
		while sample_count < max_samples:
			mb = test_reader.next_minibatch(25)
			predictedLabels = dict((key, 0) for key in xrange(num_classes))
			labelsConfidence = dict((key, 0) for key in xrange(num_classes))
			correctLabel = int(sample_count/25)
			sample_count += 25
			output = loaded_model.eval(mb, input_map=input_map)
			predictions = softmax(np.squeeze(output)).eval()
			top_classes = [np.argmax(p) for p in predictions]
			for i, c in enumerate(top_classes):
				predictedLabels[c] += 1 #Melhorar
				labelsConfidence[c] += predictions[i][c]
			label, confidence = getFinalLabel(predictedLabels, labelsConfidence)
			results += '{:^15} | {:^15} | {:^15.2f}%\n'.format(correctLabel, label, confidence)
			if sample_count%50 == 0:
				file.write(results)
				results = ''
		file.write(results)


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
	# For training
	newModelName   = "VVG16_videoOF"
	if args.logdir is not None:
		logFile = args.logdir
	map_dir = os.path.join(data_dir, "OF_mapFiles_dividedFifty")
	output_dir = os.path.join(data_dir, newModelName)
	
	train_mapFiles = sorted([os.path.join(map_dir, f) for f in os.listdir(map_dir) if 'train' in f])
	new_model_file = os.path.join(output_dir, newModelName)
	# For evaluation
	test_mapFiles = sorted([os.path.join(map_dir, f) for f in os.listdir(map_dir) if 'test' in f])
	output_file	   = os.path.join(output_dir, "eval_{}.txt".format(newModelName))
	
	### Training ###
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)

	trained_model = train_model(train_mapFiles, output_dir, logFile, newModelName)
	trained_model.save(new_model_file)

	print("Stored trained model at %s" % new_model_file)
	
	# test_model = os.path.join("F:\TCC\Models\ResNet34_videoOF-continuing3")
	# trained_model = load_model(test_model)
	## Evaluation ###
	if (os.path.exists(output_file)):
		raise Exception('The file {} already exist.'.format(output_file))

	with open(output_file, 'w') as results_file:
		results_file.write('{:^15} | {:^15} | {:^15}\n'.format('Correct label', 'Predicted label', 'Confidence'))
	
	# evaluate model and write out the desired output
	eval_and_write(trained_model, test_reader, output_file)
	
	print("Done. Wrote output to %s" % output_file)
	
	Communicator.finalize()