from __future__ import print_function
import numpy as np
import os
from PIL import Image
from cntk.device import set_default_device, gpu
from cntk import load_model, Trainer, UnitType, Axis
from cntk.layers import Placeholder, Constant
from cntk.graph import find_by_name, depth_first_search
from cntk.io import MinibatchSource, ImageDeserializer, StreamDef, StreamDefs
from cntk.layers import Dense
from cntk.learner import momentum_sgd, learning_rate_schedule, momentum_schedule
from cntk.ops import input_variable, cross_entropy_with_softmax, classification_error, combine, softmax, sequence
from cntk.ops.functions import CloneMethod
from cntk.utils import log_number_of_parameters, ProgressPrinter


################################################
################################################
# general settings
base_folder = "E:\TCC"
outputDir = os.path.join(base_folder, "Output2")
modelsDir = os.path.join(outputDir, "Models")
logFile = os.path.join(outputDir, "ResNet34_log.txt")
new_model_file = os.path.join(outputDir, "ResNet34_UCF101")
output_file = os.path.join(outputDir, "predOutput.txt")
features_stream_name = 'features'
label_stream_name = 'labels'

# define base model location and characteristics
_base_model_file = os.path.join(base_folder, "Models", "ResNet_34_101output")
_feature_node_name = "features"
_last_hidden_node_name = "pool5"
_image_height = 224
_image_width = 224
_num_channels = 3

# define data location and characteristics
_data_folder = os.path.join(base_folder, "DataSets")
_mean_file = os.path.join(_data_folder, "ImageNet1k_mean.xml")
_train_map_file = os.path.join(_data_folder, "train_map01_RGB.txt")
_test_map_file = os.path.join(_data_folder, "TestMapFiles01_RGB", "test_map01_RGB_0.txt")
_num_classes = 101
################################################
################################################


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

# Creates a minibatch source for training or testing
def create_mb_source(map_file, mean_file, image_width, image_height, num_channels, num_classes, is_training):
	if is_training:
		transforms = [
			ImageDeserializer.crop(crop_type='Random', jitter_type='uniratio'), # train uses jitter
			ImageDeserializer.scale(width=image_width, height=image_height, channels=num_channels, interpolations='cubic')
		]
		randomize = True
	else:
		transforms = [
			ImageDeserializer.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear'),
			ImageDeserializer.crop(crop_type='MultiView10')
		]
		randomize = False
	transforms += [ImageDeserializer.mean(mean_file)]
	return MinibatchSource(ImageDeserializer(map_file, StreamDefs(
		features=StreamDef(field='image', transforms=transforms),		# first column in map file is referred to as 'image'
		labels=StreamDef(field='label', shape=num_classes))),	# and second as 'label'.
		randomize=randomize)


# Creates the network model for transfer learning
def create_model(base_model_file, feature_node_name, last_hidden_node_name, num_classes, input_features):
	# Load the pretrained classification net and find nodes
	base_model	 = load_model(base_model_file)
	feature_node = find_by_name(base_model, feature_node_name)
	last_node	 = find_by_name(base_model, last_hidden_node_name)

	# Clone the desired layers
	cloned_layers = combine([last_node.owner]).clone(
		CloneMethod.clone, {feature_node: Placeholder(name='features')})

	# Add new dense layer for class prediction
	cloned_out = cloned_layers(input_features)
	z		   = Dense(num_classes, activation=None) (cloned_out)

	return z
	
	
# Trains a transfer learning model
def train_model(base_model_file, feature_node_name, last_hidden_node_name, mean_file, image_width, image_height, num_channels, 
				num_classes, train_map_file):
	# Learning parameters
	max_epochs = 6
	epoch_size = 888271
	mb_size = 64
	lr_per_mb = [0.01]*4 + [0.001]
	momentum_per_mb = 0.9
	l2_reg_weight = 0.0001
					
	# Set epoch_size to entire dataset size
	if epoch_size == 0:
		epoch_size = sum(1 for line in open(train_map_file))
		
	loaded_model = load_model(base_model_file)
	loaded_model = combine([loaded_model.outputs[2].owner])
		
	# Create the minibatch source and input variables
	minibatch_source = create_mb_source(train_map_file, mean_file, image_width, image_height, num_channels, num_classes, is_training=True)

	image_input = find_arg_by_name('features',loaded_model)
	label_input = input_variable(num_classes, 
								dynamic_axes=loaded_model.dynamic_axes,
								name='labels')

	# Define mapping from reader streams to network inputs
	input_map = {
		image_input: minibatch_source.streams.features,
		label_input: minibatch_source.streams.labels
	}

	ce = cross_entropy_with_softmax(loaded_model, label_input)
	pe = classification_error(loaded_model, label_input)
	
	# Instantiate the trainer object
	lr_schedule = learning_rate_schedule(lr_per_mb, unit=UnitType.minibatch)
	mm_schedule = momentum_schedule(momentum_per_mb)
	learner = momentum_sgd(loaded_model.parameters, lr_schedule, mm_schedule, l2_regularization_weight=l2_reg_weight)
	trainer = Trainer(loaded_model, ce, pe, learner)

	# Get minibatches of images and perform model training
	print("Training transfer learning model for {0} epochs (epoch_size = {1}).".format(max_epochs, epoch_size))
	log_number_of_parameters(loaded_model)
	progress_printer = ProgressPrinter(tag='Training', freq=10, num_epochs=max_epochs, log_to_file=logFile)
	for epoch in range(max_epochs):		  # loop over epochs
		sample_count = 0
		while sample_count < epoch_size:  # loop over minibatches in the epoch
			data = minibatch_source.next_minibatch(min(mb_size, epoch_size-sample_count), input_map=input_map)
			trainer.train_minibatch(data)									 # update model with it
			sample_count += trainer.previous_minibatch_sample_count			 # count samples processed so far
			progress_printer.update_with_trainer(trainer, with_metric=True)	 # log progress
			if sample_count % (100 * mb_size) == 0:
				print ("Processed {0} samples".format(sample_count))
		loaded_model.save_model(os.path.join(modelsDir, "ResNet_34_{}.model".format(epoch)))
		progress_printer.epoch_summary(with_metric=True)

	return loaded_model


# Evaluates a single image using the provided model
def eval_single_image(loaded_model, image_path, image_width, image_height):
	# load and format image (resize, RGB -> BGR, CHW -> HWC)
	img = Image.open(image_path)
	if image_path.endswith("png"):
		temp = Image.new("RGB", img.size, (255, 255, 255))
		temp.paste(img, img)
		img = temp
	resized = img.resize((image_width, image_height), Image.ANTIALIAS)
	bgr_image = np.asarray(resized, dtype=np.float32)[..., [2, 1, 0]]
	hwc_format = np.ascontiguousarray(np.rollaxis(bgr_image, 2))

	## Alternatively: if you want to use opencv-python
	# cv_img = cv2.imread(image_path)
	# resized = cv2.resize(cv_img, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
	# bgr_image = np.asarray(resized, dtype=np.float32)
	# hwc_format = np.ascontiguousarray(np.rollaxis(bgr_image, 2))

	# compute model output
	arguments = {loaded_model.arguments[0]: [hwc_format]}
	output = loaded_model.eval(arguments)

	# return softmax probabilities
	sm = softmax(output[0, 0])
	return sm.eval()


# Evaluates an image set using the provided model
def eval_test_images(loaded_model, output_file, test_map_file, image_width, image_height, column_offset=0):
	num_images = sum(1 for line in open(test_map_file))
	print('Evaluating model output for {} images.'.format(num_images))

	pred_count = 0
	correct_count = 0
	np.seterr(over='raise')
	with open(output_file, 'wb') as results_file:
		with open(test_map_file, "r") as input_file:
			for line in input_file:
				tokens = line.rstrip().split('\t')
				img_file = tokens[0 + column_offset]
				probs = eval_single_image(loaded_model, img_file, image_width, image_height)

				pred_count += 1
				true_label = int(tokens[1 + column_offset])
				predicted_label = np.argmax(probs)
				if predicted_label == true_label:
					correct_count += 1

				np.savetxt(results_file, probs[np.newaxis], fmt="%.3f")
				if pred_count % 100 == 0:
					print("Processed {0} samples ({1} correct)".format(pred_count, (correct_count / pred_count)))
				if pred_count >= num_images:
					break

	print ("{0} of {1} prediction were correct {2}.".format(correct_count, pred_count, (correct_count / pred_count)))


if __name__ == '__main__':
	set_default_device(gpu(0))
	
	if not os.path.exists(outputDir):
		os.mkdir(outputDir)
	
	trained_model = train_model(_base_model_file, _feature_node_name, _last_hidden_node_name, 
								_mean_file, _image_width, _image_height, 
								_num_channels, _num_classes, _train_map_file)
	trained_model.save_model(new_model_file)
	print("Stored trained model at %s" % new_model_file)

	# Evaluate the test set
	# eval_test_images(loaded_model, output_file, _test_map_file, _image_width, _image_height)
	# print("Done. Wrote output to %s" % output_file)
