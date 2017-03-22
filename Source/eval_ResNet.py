from __future__ import print_function
import os
import numpy as np
from cntk import load_model
from cntk import Trainer
from cntk.device import set_default_device, gpu
from cntk.graph import depth_first_search
from cntk.ops import combine, input_variable, softmax
from cntk.io import MinibatchSource, ImageDeserializer, StreamDef, StreamDefs
import cntk.io.transforms as xforms

def print_all_node_names(loaded_model):
	node_list = depth_first_search(loaded_model, lambda x: x.is_output)
	print("printing node information in the format")
	print("node name (tensor shape)")
	for node in node_list:
		print(node.name, node.shape)

def create_mb_source(image_height, image_width, num_channels, num_output_classes, mean_file, map_file):
	transforms = [xforms.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear'),
					xforms.crop(crop_type='MultiView10'),
					xforms.mean(mean_file)]
	return MinibatchSource(ImageDeserializer(map_file, StreamDefs(
		features=StreamDef(field='image', transforms=transforms),		# first column in map file is referred to as 'image'
		labels=StreamDef(field='label', shape=num_output_classes))),	# and second as 'label'.
		randomize=False)

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

if __name__ == '__main__':
	set_default_device(gpu(0))
	
	# define location of model and data and check existence
	base_folder = os.path.dirname(os.path.abspath(__file__))
	modelName = "ResNet_34_10.model"
	model_file	= os.path.join(base_folder, "..", "Models", modelName)
	map_dir	   = os.path.join(base_folder, "..", "DataSets", "TestMapFiles01_RGB")
	mean_file	= os.path.join(base_folder, "..", "DataSets", "ImageNet1K_mean.xml")
	output_file = os.path.join(base_folder, "..", "Results", "eval_{}.txt".format(modelName))
	
	if (os.path.exists(output_file)):
		raise Exception('The file {} already exist.'.format(output_file))

	# create minibatch source
	image_height = 224
	image_width	 = 224
	num_channels = 3
	num_output_classes = 101

	# load model
	loaded_model  = load_model(model_file)
	# loaded_model  = combine([loaded_model.outputs[2].owner])
	# print_all_node_names(loaded_model)

	#Get all test map files
	map_files = sorted(os.listdir(map_dir))
	with open(output_file, 'a') as results_file:
		results_file.write('{:<15} | {:<15} | {:<15}\n'.format('Correct label', 'Predicted label', 'Confidence'))
	for test_file in map_files:
		action_id = test_file.split('_')[-1][:-4]
		file_path = os.path.join(map_dir, test_file)
		minibatch_source = create_mb_source(image_height, image_width, num_channels, num_output_classes, mean_file, file_path)
		# evaluate model and write out the desired output
		eval_and_write(loaded_model, output_file, minibatch_source, action_id, epoch_size=250)	  # 25 frames for that result in 250 inputs for the network
	
	print("Done. Wrote output to %s" % output_file)