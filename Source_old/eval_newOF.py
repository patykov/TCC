import argparse
from cntk import load_model
from cntk.device import gpu, try_set_default_device
from cntk.io import ImageDeserializer, MinibatchSource, StreamDef, StreamDefs
import cntk.io.transforms as xforms
from cntk.logging.graph import find_by_name, get_node_outputs
from cntk.ops import softmax, combine
import numpy as np
import os


# Create a minibatch source.
def create_video_mb_source(map_files, num_channels, image_height, image_width, num_classes):
	transforms = [
		xforms.crop(crop_type='center', crop_size=224),
		xforms.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear')
	]
	
	map_files = sorted(map_files, key=lambda x: int(x.split('Map_')[1].split('.')[0]))
	print(map_files)
	
	# Create multiple image sources
	sources = []
	for i, map_file in enumerate(map_files):
		streams = {"feature"+str(i): StreamDef(field='image', transforms=transforms),
				    "label"+str(i): StreamDef(field='label', shape=num_classes)}
		sources.append(ImageDeserializer(map_file, StreamDefs(**streams)))
	
	return MinibatchSource(sources, max_sweeps=1, randomize=False)

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
def eval_and_write(network, test_mapFiles, output_file):
	# Model dimensions
	image_height = 224
	image_width	 = 224
	num_classes	 = 101
	num_channels = 3
	num_inputs   = 20#5
	
	# Create test reader
	test_reader = create_video_mb_source(test_mapFiles, num_channels, image_height, image_width, num_classes)
	
	OF_input_map = {}
	for i in range(num_inputs):
		OF_input_map[network.find_by_name("input_"+str(i))] = test_reader.streams["feature"+str(i)]

	with open(test_mapFiles[0], 'r') as file:
		lines = file.readlines()
	max_samples = len(lines)

	correctLabels = [0]*int(max_samples/25)
	for i in range(int(max_samples/25)):
		label = lines[i*25].replace('\n', '').split('\t')[-1]
		correctLabels[i] = int(label)
		
	sample_count = 0.0
	results = '{:^15} | {:^15} | {:^15}\n'.format('Correct label', 'Predicted label', 'Confidence')
	while sample_count < max_samples:
		mb = test_reader.next_minibatch(25, input_map=OF_input_map)
		predictedLabels = dict((key, 0) for key in range(num_classes))
		labelsConfidence = dict((key, 0) for key in range(num_classes))
		id_correctLabel = int(sample_count/25)
		sample_count += 25
		output = network.eval(mb)
		predictions = softmax(np.squeeze(output)).eval()
		top_classes = [np.argmax(p) for p in predictions]
		for i, c in enumerate(top_classes):
			predictedLabels[c] += 1 #Melhorar
			labelsConfidence[c] += predictions[i][c]
		label, confidence = getFinalLabel(predictedLabels, labelsConfidence)
		results += '{:^15} | {:^15} | {:^15.2f}%\n'.format(correctLabels[id_correctLabel], label, confidence)
		if sample_count%100 ==0:
			print("{:.2f}".format(float(sample_count)/max_samples))
	with open(output_file, 'w') as file:
		file.write(results)

	
if __name__ == '__main__':

	try_set_default_device(gpu(0))
	
	# Paths
	data_dir = "E:/TCC/Datasets"
	model_name   = "vgg_caffe_of_256.model"
	map_dir = os.path.join(data_dir, "OF_mapFiles_half")#"RGBdiff_mapFiles")
	output_dir = "E:/TCC/Results/new"
	test_mapFiles  = sorted([os.path.join(map_dir, f) for f in os.listdir(map_dir) if 'test' in f])
	output_file	   = os.path.join(output_dir, "eval_{}.txt".format(model_name))
	
	# Load Temporal VGG
	trained_model  = load_model('E:/TCC/Models/new/{}'.format(model_name))
	trained_model = combine([trained_model.outputs[0].owner])

	# evaluate model and write out the desired output
	eval_and_write(trained_model, test_mapFiles, output_file)
	
	print("Done. Wrote output to %s" % output_file)
