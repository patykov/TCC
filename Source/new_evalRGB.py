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
from cntk.ops import input_variable, softmax, combine, CloneMethod, placeholder
from cntk.train.training_session import *
import numpy as np
import os
from models import *
import shutil


# Create a minibatch source.
def create_video_mb_source(map_file, num_channels, image_height, image_width, num_classes, max_epochs):
	transforms = [xforms.crop(crop_type='MultiView10', crop_size=224)]
	
	return MinibatchSource(ImageDeserializer(map_file, StreamDefs(
				features = StreamDef(field='image', transforms=transforms),
				labels   = StreamDef(field='label', shape=num_classes))), 
				max_sweeps=10, randomize=False)

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
def eval_and_write(loaded_model, test_mapFile, output_file):
	# Model dimensions
	image_height = 224
	image_width	 = 224
	num_channels = 3
	num_classes	 = 101
	
	# Create test reader
	test_reader = create_video_mb_source(test_mapFile, num_channels, image_height, image_width, num_classes, 1)

	with open(test_mapFile, 'r') as file:
		lines = file.readlines()
	max_samples = len(lines)

	correctLabels = [0]*int(max_samples/25)
	for i in range(int(max_samples/25)):
		label = lines[i*25].replace('\n', '').split('\t')[-1]
		correctLabels[i] = int(label)
		
	results = '{:^15} | {:^15} | {:^15}\n'.format('Correct label', 'Predicted label', 'Confidence')
	sample_count = 0.0
	features_si = test_reader['features']
	sweeps = 0
	with open(output_file, 'a') as file:
		while (sample_count < (max_samples*10)) and (sweeps <9):
			mb = test_reader.next_minibatch(25)
			predictedLabels = dict((key, 0) for key in range(num_classes))
			labelsConfidence = dict((key, 0) for key in range(num_classes))
			if int(sample_count/25) >= len(correctLabels):
				sample_count=0
				sweeps+=1
				print(sweeps)
			id_correctLabel = int(sample_count/25)
			sample_count += mb[features_si].num_samples
			output = loaded_model.eval({loaded_model.arguments[0]:mb[features_si]})
			predictions = softmax(np.squeeze(output)).eval()
			top_classes = [np.argmax(p) for p in predictions]
			for i, c in enumerate(top_classes):
				predictedLabels[c] += 1 #Melhorar
				labelsConfidence[c] += predictions[i][c]
			label, confidence = getFinalLabel(predictedLabels, labelsConfidence)
			results += '{:^15} | {:^15} | {:^15.2f}%\n'.format(correctLabels[id_correctLabel], label, confidence)
			if sample_count%1000 == 0:
				file.write(results)
				results = ''
	with open(output_file, 'a') as file:
		file.write(results)


if __name__ == '__main__':
	try_set_default_device(gpu(0))
	
	# Paths
	data_dir = 'F:/TCC/Datasets/'
	# For training
	newModelName = "VGG16_videoRGB_small_10swe"
	map_dir = os.path.join(data_dir, "UCF-101_rgbMapFiles_split1")
	output_dir = 'F:/TCC/Results'
	# For evaluation
	test_mapFile = os.path.join(map_dir, 'testMap_1_small.txt')
	output_file = os.path.join(output_dir, "eval_{}.txt".format(newModelName))

	trained_model = load_model('F:/TCC/Models/philly/VGG16_videoRGB_final.dnn')
	trained_model = combine([trained_model.outputs[0].owner])
	
	## Evaluation ###
	eval_and_write(trained_model, test_mapFile, output_file)
	
	print("Done. Wrote output to %s" % output_file)
	
	Communicator.finalize()